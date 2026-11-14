"""Export an untrained embedding-width candidate and enforce all submission caps."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension',type=int,default=96)
    parser.add_argument('--frames',type=int,default=1200)
    parser.add_argument('--work-dir',type=Path,required=True,help='new directory for candidate assets/module')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if min(args.dimension,args.frames)<1:parser.error('dimension and frames must be positive')
    import torch
    import config
    from export import make_models,notebook_settings,export_models
    from verify_limits import verify
    from viobench.solution.runtime.python.bundler.bundle import bundle
    from viobench.tools.cli import run_with_large_stack
    torch.set_num_threads(1);torch.manual_seed(1701)
    settings=notebook_settings(config)
    settings.update(frame_mode='embedding',visual_prediction='embedding',learned_visual=True,
                    embedding_dim=args.dimension)
    frame,state=make_models(settings)
    with torch.no_grad():state.out.weight.normal_(0,.001)
    args.work_dir.mkdir(parents=True,exist_ok=False)
    assets=args.work_dir/'solution'
    export_models(frame,state,settings,assets)
    module=args.work_dir/'candidate.wasm'
    bundle(assets,module)
    result=run_with_large_stack(lambda:verify(module,args.frames))
    result.update(scope='Untrained seed-1701 candidate; resource feasibility only, no accuracy result.',
                  settings=settings,parameters=dict(frame=sum(p.numel() for p in frame.parameters()),
                  state=sum(p.numel() for p in state.parameters())))
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
