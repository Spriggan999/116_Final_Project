import sys
from pathlib import Path
import copy
import numpy as np
import unittest
import tempfile
from functools import partial, update_wrapper
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legacy
from gated_models import GatedModel, PROFILES
from gated_training import objective, train_student
from gated_recovery import save_latest, restore_latest

torch.set_num_threads(2)


def inputs():
    imu = torch.randn(1, 21, 6)*.02
    imu[:, :, 2] += 9.81
    observed = torch.zeros(1, 21, dtype=torch.bool)
    observed[:, [0, 8, 20]] = True
    return dict(images=torch.rand(1, 3, 3, 32, 40), imu=imu,
                ticks=torch.tensor([[0, 8, 20]]), observed=observed,
                dt=torch.tensor([.005]), gravity=torch.tensor([[0., 0., -9.81]]))


def test_streaming_causality_and_missing_images(name):
    torch.manual_seed(17)
    model = GatedModel(PROFILES[name]).eval()
    if model.innovation is not None:
        torch.nn.init.normal_(model.innovation[-1].weight, std=.01)
    batch = inputs()
    with torch.no_grad():
        prediction = model(**batch)
        assert torch.count_nonzero(prediction['sparse_visual'][:, ~batch['observed'][0]]) == 0
        assert torch.count_nonzero(prediction['innovation_update'][:, ~batch['observed'][0]]) == 0
        state = model.stream_init()
        streamed = []
        for tick in range(21):
            slot = (batch['ticks'][0] == tick).nonzero().flatten()
            image = batch['images'][:, slot[0]] if len(slot) else None
            value, state = model.stream_step(image, batch['imu'][:, tick], .005, batch['gravity'], state)
            streamed.append(value)
        for key in ('pos','vel','rot','omega','corrections'):
            actual = torch.stack([row[key] for row in streamed], 1)
            torch.testing.assert_close(actual, prediction[key], atol=2e-6, rtol=2e-5)
        future = {k:v.clone() for k,v in batch.items()}
        future['images'][:, 2] = torch.rand_like(future['images'][:, 2])
        future['imu'][:, 16:] += 1
        changed = model(**future)
        for key in ('pos','vel','rot','omega','corrections'):
            torch.testing.assert_close(changed[key][:, :16], prediction[key][:, :16])
        # Current image must not enter the prediction of its own embedding.
        future = {k:v.clone() for k,v in batch.items()}
        future['images'][:, 2] += .2
        torch.testing.assert_close(model(**future)['predicted'][:, 2], prediction['predicted'][:, 2])


def test_losses_gradients_ema_and_update(name):
    model = GatedModel(PROFILES[name])
    batch = inputs()
    truth = dict(pos=torch.zeros(1,21,3), vel=torch.zeros(1,21,3),
                 rot=torch.eye(3).expand(1,21,3,3), omega=torch.zeros(1,21,3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    row = dict(model=model, optimizer=optimizer)
    before = model.target.cnn[0].weight.detach().clone()
    result = train_student(name, row, (batch, truth, None, None), 'cpu', (0,0,1))
    assert torch.isfinite(result)
    assert model.vision.cnn[0].weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    assert model.predictor[-1].weight.grad is not None
    assert all(p.grad is None for p in model.target.parameters())
    torch.testing.assert_close(model.target.cnn[0].weight, before*.99+model.vision.cnn[0].weight.detach()*.01)
    if model.auxiliary is not None:
        assert model.auxiliary[-1].weight.grad is not None


def test_resume_roundtrip(tmp_path):
    model = GatedModel(PROFILES['combined'])
    row = dict(model=model, optimizer=torch.optim.AdamW(model.parameters()))
    save_latest(tmp_path, {'combined':row})
    expected = copy.deepcopy(model.state_dict())
    expected_random = torch.rand(5)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    restore_latest(tmp_path, {'combined':row})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    torch.testing.assert_close(torch.rand(5), expected_random)


def test_camera_coordinates_and_angular_lever_arm():
    from evaluate import camera_state
    rotation = torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    pred = dict(pos=torch.tensor([[1.,2.,3.]]), rot=rotation,
                vel=torch.tensor([[0.,1.,0.]]), omega=torch.tensor([[0.,0.,2.]]))
    transform = np.eye(4)
    transform[0, 3] = 1
    state = camera_state(pred, transform)
    np.testing.assert_allclose(state[:3], [0,3,3])
    np.testing.assert_allclose(state[7:10], [1,2,0])


def test_existing_ten_horizon_policy():
    from student_curriculum import DEFAULT_STUDENT_WINDOWS
    assert DEFAULT_STUDENT_WINDOWS == list(range(401,4002,400))


def test_gated_power_of_two_schedule_and_collection():
    import json
    from gated_curriculum import DEFAULT_HORIZON_STEPS, DEFAULT_GATED_WINDOWS
    assert DEFAULT_HORIZON_STEPS == [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,12000]
    assert DEFAULT_GATED_WINDOWS == [value+1 for value in DEFAULT_HORIZON_STEPS]
    assert (DEFAULT_GATED_WINDOWS[-1]-1)*.005 == 60
    settings=json.loads((Path(__file__).resolve().parents[1]/'collection_60s.json').read_text())
    assert settings['COLLECT_STEPS'] >= DEFAULT_GATED_WINDOWS[-1]
    assert settings['STREAM_MAX_MIB'] == 24576


def test_single_interval_has_finite_losses_and_no_false_jepa_target():
    for name in PROFILES:
        model=GatedModel(PROFILES[name])
        batch=inputs()
        batch['imu']=batch['imu'][:,:2]
        batch['images']=batch['images'][:,:1]
        batch['observed']=batch['observed'][:,:2]
        batch['ticks']=batch['ticks'][:,:1]
        truth=dict(pos=torch.zeros(1,2,3),vel=torch.zeros(1,2,3),
                   rot=torch.eye(3).expand(1,2,3,3),omega=torch.zeros(1,2,3))
        prediction=model(**batch)
        jepa=model.jepa_loss(prediction,batch['images'])
        assert jepa['embedding_prediction'] == jepa['auxiliary_prediction'] == 0
        assert not prediction['valid'].any()
        assert not prediction['innovation_update'].any()
        total,_,terms=objective(model,prediction,batch,truth)
        assert torch.isfinite(total)
        assert terms['alignment_weighted'] == 0
        total.backward()
        assert model.gru.weight_ih_l0.grad is not None


def test_uint8_capture_encoding_matches_normalized_float():
    model=GatedModel(PROFILES['combined']).eval()
    batch=inputs()
    batch['images']=(batch['images']*255).round().to(torch.uint8)
    normalized=dict(batch,images=batch['images'].float()/255.)
    with torch.no_grad():
        a,b=model(**batch),model(**normalized)
    for key in ('online','predicted','pos','rot','vel','omega'):
        torch.testing.assert_close(a[key],b[key])
    model.train()
    result=model(**batch)
    sum(model.jepa_loss(result,batch['images']).values()).backward()
    assert model.vision.cnn[0].weight.grad is not None


def test_short_and_sixty_second_recorded_windows():
    from data import StateWindows
    from gated_training import frame_inputs
    from gated_curriculum import DEFAULT_GATED_WINDOWS
    from types import SimpleNamespace
    state=torch.zeros(12001,18)
    state[:,3:12]=torch.eye(3).flatten()
    imu=torch.zeros(12001,6)
    imu[:,2]=9.81
    entry=dict(cams=torch.zeros(3001,3,8,8,dtype=torch.uint8),cam_ticks=torch.arange(0,12001,4),
               imu=imu,state=state,stride=4,dt=.005,file='synthetic.npz',scene_id=0)
    store=SimpleNamespace(entries=[entry])
    for length in DEFAULT_GATED_WINDOWS:
        window=StateWindows(store,length=length,compact_cameras=True)[0]
        assert len(window['imu']) == length
        batch={key:value[None] for key,value in window.items() if torch.is_tensor(value)}
        prepared,_=frame_inputs(batch,'cpu')
        assert prepared['ticks'][0,0] == 0
        assert prepared['ticks'].max() < length
        if length == 12001:
            assert prepared['images'].shape[1] == 1201


def test_public_full_flight_keeps_one_state():
    from evaluate import public_flight
    from types import SimpleNamespace
    frame_times = np.arange(1200)*.05
    imu = np.zeros((11991, 7))
    imu[:, 0] = np.arange(len(imu))*.005
    imu[:, 3] = 9.81
    truth = np.zeros((1200, 13))
    truth[:, 6] = 1
    class Frames:
        def __getitem__(self, index):
            return np.zeros((16, 16, 3), np.uint8)
    episode = SimpleNamespace(frame_times=frame_times, imu=imu, T_body_camera=np.eye(4),
                              TIME_TOLERANCE=1e-6, adapt=lambda size: ({}, Frames()),
                              ground_truth_state=lambda: truth)
    class Probe:
        def __init__(self):
            self.resets = self.images = self.ticks = 0
        def stream_init(self):
            self.resets += 1
            return dict(n=0)
        def stream_step(self, image, imu, dt, gravity, state):
            assert state['n'] == self.ticks
            self.ticks += 1
            self.images += image is not None
            z = torch.zeros(1, 3)
            return dict(pos=z, vel=z, omega=z, rot=torch.eye(3)[None]), dict(n=self.ticks)
    probe = Probe()
    metrics, estimates, _ = public_flight(probe, episode, 'cpu')
    assert probe.resets == 1 and probe.images == 1200 and probe.ticks == 11991
    assert estimates.shape == (1200, 13)
    assert metrics['sim3_ate_rmse_m'] == metrics['unaligned_ate_rmse_m'] == 0


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
def test_cuda_schedule_rollback_and_updates():
    from gated_schedule import MeasuredPipeline
    batch = {key:value.cuda() for key,value in inputs().items()}
    truth = dict(pos=torch.zeros(1,21,3,device='cuda'), vel=torch.zeros(1,21,3,device='cuda'),
                 rot=torch.eye(3,device='cuda').expand(1,21,3,3), omega=torch.zeros(1,21,3,device='cuda'))
    rows = {}
    for name in ('baseline','combined'):
        model = GatedModel(PROFILES[name]).cuda()
        rows[name] = dict(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4))
    expected = {name:copy.deepcopy(row['model'].state_dict()) for name,row in rows.items()}
    ready = torch.cuda.Event()
    ready.record()
    operation = partial(train_student, device='cuda')
    with MeasuredPipeline('cuda', rows, target_prefetch=False) as pipeline:
        pipeline.warmup_steps = pipeline.measure_steps = pipeline.repeats = 1
        result = pipeline.autotune(rows, ((batch, truth), ready), operation)
        assert result['candidates'] and result['max_concurrent'] >= 1
        for name,row in rows.items():
            for key,value in row['model'].state_dict().items():
                torch.testing.assert_close(value, expected[name][key], atol=0, rtol=0)
        pipeline.run(rows, ((batch,truth),ready), operation)
        assert all(torch.isfinite(row['gradient_norm_preclip']) for row in rows.values())


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in PROFILES:
        for function in (test_streaming_causality_and_missing_images, test_losses_gradients_ema_and_update):
            case = update_wrapper(partial(function, name), function)
            suite.addTest(unittest.FunctionTestCase(case, description=function.__name__+' '+name))
    def resume():
        with tempfile.TemporaryDirectory() as directory:
            test_resume_roundtrip(Path(directory))
    for function in (resume, test_camera_coordinates_and_angular_lever_arm, test_existing_ten_horizon_policy,
                     test_public_full_flight_keeps_one_state, test_cuda_schedule_rollback_and_updates,
                     test_gated_power_of_two_schedule_and_collection,
                     test_single_interval_has_finite_losses_and_no_false_jepa_target,
                     test_uint8_capture_encoding_matches_normalized_float,
                     test_short_and_sixty_second_recorded_windows):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == '__main__':
    unittest.main()
