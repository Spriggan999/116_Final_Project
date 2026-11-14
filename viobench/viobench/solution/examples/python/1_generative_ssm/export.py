"""Convert trained .pt checkpoints or notebook NPZ exports to a VIOBench solution."""
import argparse
import copy
import json
import re
from pathlib import Path
import shutil
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent


def notebook_settings(cfg):
    """Record all architectural choices needed to reload a bare state_dict."""
    return dict(
        backbone='mamba2_hybrid' if cfg.USE_HYBRID_SSM_TRANSFORMER else 'gru',
        frame_mode='embedding' if cfg.VISUAL_PREDICTION=='embedding' else
                   (('flow' if cfg.USE_GENERATIVE_FRAME_MODEL else 'affine') if cfg.USE_GENERATED_FRAMES_FOR_STATE else 'repeat'),
        visual_prediction=cfg.VISUAL_PREDICTION,
        embedding_hidden=int(cfg.EMBEDDING_HIDDEN),embedding_ema=float(cfg.EMBEDDING_EMA),
        embedding_dim=int(getattr(cfg,'EMBEDDING_DIM',48)),
        use_generated_frames=bool(cfg.USE_GENERATED_FRAMES_FOR_STATE),
        frame_student='embedding_predictor' if cfg.VISUAL_PREDICTION=='embedding' else ('flow_generative' if cfg.USE_GENERATIVE_FRAME_MODEL else 'affine'),
        frame_generator_base=int(cfg.FRAME_GENERATOR_BASE),
        learned_visual=bool(cfg.USE_LEARNED_VISUAL_ENCODER), hidden=int(cfg.STUDENT_HIDDEN),
        width=int(cfg.W), height=int(cfg.H),
        frame_width=640 if (cfg.W,cfg.H) == (640,480) else 320,
        max_flow_pixels=float(cfg.FRAME_MAX_FLOW_PIXELS),
        frame_input_stride=int(cfg.FRAME_INPUT_STRIDE),
        visual_input_stride=int(cfg.VISUAL_INPUT_STRIDE),
        mamba_layers=int(cfg.HYBRID_NUM_MAMBA_LAYERS) if cfg.USE_HYBRID_SSM_TRANSFORMER else 0,
        attention_window=int(cfg.HYBRID_ATTENTION_WINDOW),
        hybrid_d_state=int(cfg.HYBRID_D_STATE), hybrid_head_dim=int(cfg.HYBRID_HEAD_DIM),
        hybrid_attention_heads=4,
        state_visual_feature_dropout_training_only=float(cfg.STATE_VISUAL_FEATURE_DROPOUT),
    )


def normalize_settings(raw):
    """Accept both student_config.json and deployment.json from the notebook."""
    s = dict(raw)
    s['backbone'] = s.get('backbone', s.get('student_backbone', 'gru'))
    s.setdefault('frame_mode', ('flow' if s.get('frame_student') == 'flow_generative' else 'affine')
                 if s.get('generated_frames_for_state', s.get('use_generated_frames', True)) else 'repeat')
    if 'frame_mode' not in raw and (raw.get('visual_prediction')=='embedding' or raw.get('frame_student')=='embedding_predictor'):
        s['frame_mode']='embedding'
    s['visual_prediction']='embedding' if s['frame_mode']=='embedding' else 'rgb'
    s['use_generated_frames'] = bool(s.get('use_generated_frames',True)) if s['frame_mode']=='embedding' else s['frame_mode']!='repeat'
    for key, default in dict(hidden=96, width=96, height=72, mamba_layers=0,
                             attention_window=16, hybrid_d_state=16, hybrid_head_dim=32,
                             hybrid_attention_heads=4, frame_generator_base=24).items():
        s[key] = int(s.get(key, default))
    s.setdefault('learned_visual', True)
    s['embedding_hidden']=int(s.get('embedding_hidden',256))
    s['embedding_ema']=float(s.get('embedding_ema',.99))
    s['embedding_dim']=int(s.get('embedding_dim',48))
    if s['embedding_dim']<1:raise ValueError('embedding_dim must be positive')
    if s['embedding_dim']!=48 and (s['frame_mode']!='embedding' or not s['learned_visual']):
        raise ValueError('Non-48 embeddings require embedding mode and a learned visual encoder')
    if s['embedding_hidden']<1 or not 0<=s['embedding_ema']<1:raise ValueError('Invalid embedding hidden width or EMA')
    s['frame_width'] = int(s.get('frame_width', 640 if (s['width'],s['height']) == (640,480) else 320))
    s['max_flow_pixels'] = float(s.get('max_flow_pixels',24.))
    s['frame_input_stride'] = int(s.get('frame_input_stride',1))
    s['visual_input_stride'] = int(s.get('visual_input_stride',2))
    if s['frame_input_stride'] not in (1,2,4,8) or s['visual_input_stride'] not in (2,4,8):
        raise ValueError('Invalid learned input patch stride')
    if s['width'] % s['frame_input_stride'] or s['height'] % s['frame_input_stride']:
        raise ValueError('Image dimensions must be divisible by the flow input stride')
    if s['frame_width'] not in (320,640) or not np.isfinite(s['max_flow_pixels']) or s['max_flow_pixels'] <= 0:
        raise ValueError('Invalid frame_width or max_flow_pixels')
    if s['backbone'] not in ('gru', 'mamba2_hybrid'):
        raise ValueError('backbone must be gru or mamba2_hybrid')
    if s['frame_mode'] not in ('repeat', 'affine', 'flow','embedding'):
        raise ValueError('frame_mode must be repeat, affine, flow or embedding')
    if min(s['hidden'], s['height'], s['width']) < 1:
        raise ValueError('hidden, height and width must be positive')
    if s['backbone'] == 'mamba2_hybrid':
        if s['mamba_layers'] < 1 or s['attention_window'] < 1:
            raise ValueError('Mamba needs positive mamba_layers and attention_window')
        if s['hybrid_head_dim'] < 1 or 2*s['hidden'] % s['hybrid_head_dim']:
            raise ValueError('2*hidden must be divisible by hybrid_head_dim')
        if (2*s['hidden']//s['hybrid_head_dim']) % 8 or s['hybrid_d_state'] < 4 or s['hybrid_d_state'] % 4:
            raise ValueError('Official fused Mamba-2 needs head count divisible by 8 and d_state divisible by 4')
        if s['hybrid_attention_heads'] < 1 or s['hidden'] % s['hybrid_attention_heads']:
            raise ValueError('hidden must be divisible by hybrid_attention_heads')
    return s


def make_models(settings):
    from models import GenerativeFlowPredictor, MotionWarp, StateStudent, HybridStateStudent,EmbeddingPredictor
    s = normalize_settings(settings)
    state = (HybridStateStudent(hidden=s['hidden'], n_mamba=s['mamba_layers'],
              attn_window=s['attention_window'], d_state=s['hybrid_d_state'],
              headdim=s['hybrid_head_dim'], heads=s['hybrid_attention_heads'],
              learned_visual=s['learned_visual'], visual_input_stride=s['visual_input_stride'],
              visual_embeddings=s['frame_mode']=='embedding',embedding_dim=s['embedding_dim'])
             if s['backbone'] == 'mamba2_hybrid' else
             StateStudent(hidden=s['hidden'], learned_visual=s['learned_visual'],visual_input_stride=s['visual_input_stride'],
                          visual_embeddings=s['frame_mode']=='embedding',embedding_dim=s['embedding_dim']))
    frame = (EmbeddingPredictor(hidden=s['embedding_hidden'],learned_visual=s['learned_visual'],
                               visual_input_stride=s['visual_input_stride'],ema=s['embedding_ema'],embedding_dim=s['embedding_dim'])
             if s['frame_mode']=='embedding' else
             GenerativeFlowPredictor(base=s['frame_generator_base'], max_flow_pixels=s['max_flow_pixels'],input_stride=s['frame_input_stride'])
             if s['frame_mode'] == 'flow' or (s['frame_mode'] == 'repeat' and s.get('frame_student') == 'flow_generative')
             else MotionWarp(hidden=24))
    return frame.eval(), state.eval()


def portable_visual(visual, height, width):
    """Lower fixed adaptive pooling to a depthwise Conv supported by cv2.dnn.

    At 72x96 the last CNN map is 5x6; PyTorch's 3x4 adaptive bins overlap.
    A fixed grouped Conv computes those exact averages without adaptive ONNX ops.
    """
    import torch
    from torch import nn
    net = copy.deepcopy(visual).cpu().eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, height, width)
        for layer in list(net.cnn)[:8]:
            x = layer(x)
    channels, h, w = x.shape[1:]
    pooling = nn.Conv2d(channels, channels*12, (h, w), groups=channels, bias=False)
    with torch.no_grad():
        pooling.weight.zero_()
        for c in range(channels):
            for i in range(3):
                lo_y, hi_y = i*h//3, ((i+1)*h+2)//3
                for j in range(4):
                    lo_x, hi_x = j*w//4, ((j+1)*w+3)//4
                    pooling.weight[c*12+i*4+j, 0, lo_y:hi_y, lo_x:hi_x] = 1/((hi_y-lo_y)*(hi_x-lo_x))
    net.cnn[8] = pooling
    return net.eval()


def export_models(frame, state, settings, output, frame_width=None):
    """Export and numerically check before copying assets to the bundle directory."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    import cv2
    import onnx

    s = normalize_settings(settings)
    frame_width = s['frame_width'] if frame_width is None else frame_width
    if frame_width not in (320, 640):
        raise ValueError('frame_width must be 320 or 640')
    s['frame_width'] = frame_width
    frame, state = copy.deepcopy(frame).cpu().eval(), copy.deepcopy(state).cpu().eval()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    class RawFlow(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.m = model

        def forward(self, image, condition):
            a = self.m.enc1(image)
            b = self.m.enc2(a)
            c = self.m.enc3(b)
            z = self.m.bottleneck(c) + self.m.imu_condition(condition)[:, :, None, None]
            d2 = self.m.dec2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
            d1 = self.m.dec1(torch.cat([F.interpolate(d2, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
            raw=self.m.head(d1)
            if self.m.input_stride > 1:
                raw=F.interpolate(raw,size=image.shape[-2:],mode='bilinear',align_corners=False)
            return raw

    rng = np.random.default_rng(17)
    def onnx_export(model, arrays, names, path):
        inputs = tuple(torch.from_numpy(a) for a in arrays)
        with torch.no_grad():
            reference = model(*inputs).numpy()
            torch.onnx.export(model, inputs, str(path), input_names=names,
                              output_names=['output'], opset_version=13, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        net = cv2.dnn.readNetFromONNX(str(path))
        for name, value in zip(names, arrays):
            net.setInput(value, name)
        result = net.forward()
        np.testing.assert_allclose(result, reference, atol=3e-3, rtol=3e-3)
        print(f'{path.name}: OpenCV {cv2.__version__} / PyTorch max error {np.max(np.abs(result-reference)):.3g}')

    with tempfile.TemporaryDirectory(prefix='hyperdrone-export-', dir=output.parent) as temp:
        staged = Path(temp)
        np.savez(staged/'state_student.npz', **{k: v.detach().float().numpy() for k, v in state.state_dict().items()})
        if s['frame_mode'] in ('affine','embedding'):
            np.savez(staged/'frame_student.npz', **{k: v.detach().float().numpy() for k, v in frame.state_dict().items()})
        if s['learned_visual']:
            encoder=frame.encoder if s['frame_mode']=='embedding' else state.visual
            visual = portable_visual(encoder, s['height'], s['width'])
            images = rng.random((1, 3, s['height'], s['width']), dtype=np.float32)
            with torch.no_grad():
                np.testing.assert_allclose(visual(torch.from_numpy(images)).numpy(),
                    encoder(torch.from_numpy(images)).numpy(), atol=1e-6, rtol=1e-5)
            onnx_export(visual, [images], ['input'], staged/'visual.onnx')
        if s['frame_mode'] == 'flow':
            images = rng.normal(0, .3, (1, 11, s['height'], s['width'])).astype(np.float32)
            cond = rng.normal(0, .1, (1, 8)).astype(np.float32)
            onnx_export(RawFlow(frame), [images, cond], ['image', 'condition'], staged/'frame_raw.onnx')
        (staged/'deployment.json').write_text(json.dumps(s, indent=2)+'\n')
        source = (HERE/'solution'/'solution.py').read_text()
        (staged/'solution.py').write_text(re.sub(
            r'^    frame_width = (320|640)$', f'    frame_width = {frame_width}', source, flags=re.MULTILINE))
        # Exercise streaming inference through the same bundle API used by the runtime.
        from validate import validate_solution
        validate_solution(staged)
        output.mkdir(parents=True, exist_ok=True)
        # Remove only stale deployment assets managed by this exporter.
        for name in ('visual.onnx', 'frame_raw.onnx', 'frame_student.npz'):
            if (output/name).exists() and not (staged/name).exists():
                (output/name).unlink()
        for path in staged.iterdir():
            if path.is_file():
                shutil.copy2(path, output/path.name)
    print(f'Exported solution: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, help='train.py run directory with .pt weights and student_config.json')
    parser.add_argument('--assets', type=Path, help='notebook export: state_student.npz, frame_student.npz, student_config.json')
    parser.add_argument('--config', type=Path, help='explicit student_config.json/deployment.json')
    parser.add_argument('--state-checkpoint', type=Path)
    parser.add_argument('--frame-checkpoint', type=Path)
    parser.add_argument('--frame-mode', choices=['flow', 'affine', 'repeat'], help='explicit deployment ablation override')
    parser.add_argument('--frame-width', type=int, choices=[320, 640], help='override the saved ABI input width')
    parser.add_argument('--out', type=Path, default=HERE/'solution')
    args = parser.parse_args()
    root = args.checkpoints or args.assets
    if root is None and not (args.config and args.state_checkpoint):
        parser.error('give --checkpoints DIR, --assets DIR or --config JSON --state-checkpoint FILE')
    config_path = args.config or root/'student_config.json'
    settings = json.loads(config_path.read_text())
    if args.frame_mode:
        if normalize_settings(settings)['frame_mode']=='embedding':
            if args.frame_mode!='repeat':parser.error('RGB frame-mode overrides require RGB-trained checkpoints; train a fresh run with --visual-prediction rgb')
            settings.update(frame_mode='embedding',use_generated_frames=False)
        else:
            settings['frame_mode'] = args.frame_mode
    settings = normalize_settings(settings)
    frame, state = make_models(settings)
    import torch
    if args.assets:
        def load_npz(path):
            with np.load(path, allow_pickle=False) as data:
                return {k: torch.from_numpy(v.copy()) for k, v in data.items()}
        state.load_state_dict(load_npz(args.assets/'state_student.npz'), strict=True)
        if settings['frame_mode'] != 'repeat':
            frame.load_state_dict(load_npz(args.assets/'frame_student.npz'), strict=True)
    else:
        state_path = args.state_checkpoint or root/'state_student.pt'
        state.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True), strict=True)
        if settings['frame_mode'] != 'repeat':
            frame_path = args.frame_checkpoint or root/'frame_student.pt'
            frame.load_state_dict(torch.load(frame_path, map_location='cpu', weights_only=True), strict=True)
    export_models(frame, state, settings, args.out, args.frame_width)


if __name__ == '__main__':
    main()
