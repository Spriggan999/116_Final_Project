"""CPU input-only comparison; excludes GPU augmentation, model, rendering and I/O."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config
from curriculum import AugmentedDataset,DifficultyCurriculum
from data import FramePairs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'verification/wandb_curriculum_20261113/input_benchmark.json')
    args=parser.parse_args()
    torch.manual_seed(17)
    cameras=torch.randint(0,256,(17,480,640,3),dtype=torch.uint8).permute(0,3,1,2)
    state=torch.zeros(81,18);state[:,3:12]=torch.eye(3).reshape(1,9)
    imu=torch.zeros(81,6);imu[:,2]=9.81
    store=SimpleNamespace(entries=[dict(cams=cameras,state=state,imu=imu,cam_ticks=torch.arange(0,81,5),
                                      stride=5,dt=.005,file='synthetic-flight.npz',scene_id=0)])
    results={}
    for name,threads,batch,raw,deferred in [('old',20,8,False,False),('new',2,32,True,True)]:
        torch.set_num_threads(threads)
        values={key:value for key,value in vars(config).items() if key.isupper()}
        values.update(CURRICULUM_MIX=[1.,0.,0.],CURRICULUM_DEVICE_PHOTOMETRIC=deferred)
        cfg=SimpleNamespace(**values);schedule=DifficultyCurriculum(cfg)
        schedule.stages=dict(motion=2,visual=2,imu=2)
        dataset=AugmentedDataset(FramePairs(store,max_samples=128,raw_images=raw),cfg,schedule,17)
        loader=DataLoader(dataset,batch_size=batch,num_workers=0)
        start=time.perf_counter();samples=0
        for row in loader:samples+=len(row['anchor'])
        seconds=time.perf_counter()-start
        results[name]=dict(cpu_threads=threads,batch=batch,samples=samples,seconds=seconds,
                           samples_per_second=samples/seconds)
    results['scope']=__doc__
    results['input_only_speedup']=results['old']['seconds']/results['new']['seconds']
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
