"""Compare FP32 parallel training physics and gradients with sequential physics."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import integrate_corrections


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(1701)
    result=dict(scope=__doc__,device=args.device,passed=False,errors=[])
    for length in (1,2,17,401,1601,4001):
        corr=torch.randn(2,length,9,device=args.device)*.05
        imu=torch.randn(2,length,6,device=args.device)*.2;imu[:,:,2]+=9.81
        dt=torch.full((2,),.005,device=args.device)
        gravity=torch.tensor([[0.,0.,-9.81]],device=args.device).repeat(2,1)
        with torch.no_grad():
            expected=integrate_corrections(corr,imu,dt,gravity)
            actual=integrate_corrections(corr,imu,dt,gravity,parallel=True)
        row=dict(length=length,max_absolute_error={k:float((actual[k]-expected[k]).abs().max()) for k in expected})
        for key in expected:
            atol=dict(pos=.01,vel=.002,rot=.0001,omega=0.,corrections=0.)[key]
            torch.testing.assert_close(actual[key],expected[key],atol=atol,rtol=1e-5)
        result['errors'].append(row)
    corr=(torch.randn(2,53,9,device=args.device)*.05).requires_grad_()
    other=corr.detach().clone().requires_grad_()
    imu=torch.randn(2,53,6,device=args.device)*.2;imu[:,:,2]+=9.81
    dt=torch.full((2,),.005,device=args.device)
    def objective(prediction):
        return sum(prediction[key].square().mean() for key in ('pos','vel','rot'))
    objective(integrate_corrections(corr,imu,dt,gravity)).backward()
    objective(integrate_corrections(other,imu,dt,gravity,parallel=True)).backward()
    torch.testing.assert_close(corr.grad,other.grad,atol=2e-8,rtol=2e-4)
    result.update(passed=True,gradient_max_absolute_error=float((corr.grad-other.grad).abs().max()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
