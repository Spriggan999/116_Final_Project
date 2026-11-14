"""Load a repaired local scene and collect short real GPU rollouts."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    from scene_assets import prepare_scene
    from collect import collect_round
    from hyperdrone import Config,HyperDrone
    result=dict(source=str(args.scene),passed=False,scope=__doc__)
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='hyperdrone-scene-check-') as temporary:
        path,removed=prepare_scene(args.scene,temporary)
        result['removed_empty_primitives']=removed
        env=None
        try:
            env=HyperDrone([path],config=Config(num_worlds=1,instances_per_world=2,
                width=640,height=480,preset='x500_fpv_imu',task='visual_inertial_localization'),seed=17)
            rows=collect_round(env,41)
            result.update(passed=any(row is not None for row in rows),
                          ticks=[len(row['state']) if row is not None else 0 for row in rows],
                          frames=[len(row['camera']) if row is not None else 0 for row in rows])
        finally:
            if env is not None:env.close()
    result['seconds']=time.perf_counter()-started
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
