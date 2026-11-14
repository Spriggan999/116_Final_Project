"""Synthetic CUDA PPO action/reward/GAE timing with precomputed causal tokens.

Excludes image encoding, backbone inference, data collection and optimizer work.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--windows',type=int,nargs='+',default=[401,4001])
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('CUDA required')
    if min(args.windows)<3:parser.error('windows must include initial and bootstrap tokens')
    import ppo
    torch.set_num_threads(2);torch.manual_seed(1701)
    torch.cuda.set_per_process_memory_fraction(.4)
    student=SimpleNamespace(out=torch.nn.Linear(896,9).cuda())
    critic=torch.nn.Sequential(torch.nn.Linear(926,128),torch.nn.Tanh(),torch.nn.Linear(128,1)).cuda()
    cfg=SimpleNamespace(DEVICE='cuda',PPO_ROLLOUT_BATCHES=1,PPO_STD=.1,PPO_GAMMA=.9995,PPO_GAE_LAMBDA=.95)
    original_input,original_tokens=ppo.state_input_frames,ppo.frozen_tokens
    result=dict(scope=__doc__,gpu=torch.cuda.get_device_name(),cases=[])
    try:
        for length in args.windows:
            tokens=torch.randn(1,length,896,device='cuda')*.1
            ppo.state_input_frames=lambda *a:tokens
            ppo.frozen_tokens=lambda *a:tokens
            imu=torch.zeros(1,length,6,device='cuda');imu[:,:,2]=9.81
            batch=dict(imu=imu,dt=torch.tensor([.005],device='cuda'),gravity=torch.tensor([[0.,0.,-9.81]],device='cuda'),
                pos=torch.zeros(1,length,3,device='cuda'),vel=torch.zeros(1,length,3,device='cuda'),
                omega=torch.zeros(1,length,3,device='cuda'),rot=torch.eye(3,device='cuda')[None,None].expand(1,length,3,3))
            for vectorized in (False,True):
                cfg.VECTORIZED_PPO_ROLLOUT=vectorized
                ppo.collect_rollout(None,student,critic,[batch],cfg)
                gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                rollout,reward=ppo.collect_rollout(None,student,critic,[batch],cfg)
                torch.cuda.synchronize()
                row=dict(window=length,vectorized=vectorized,seconds=time.perf_counter()-started,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         transitions=len(rollout['actions']),finite=all(bool(torch.isfinite(v).all()) for v in rollout.values()))
                result['cases'].append(row);del rollout
                print(json.dumps(row),flush=True)
    finally:ppo.state_input_frames,ppo.frozen_tokens=original_input,original_tokens
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
