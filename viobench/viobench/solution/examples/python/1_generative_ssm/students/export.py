"""Embed float32 weights into a native reactor and enforce the exact fuel tier."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[3]
sys.path.insert(0,str(WORKSPACE.parents[1]))


def ordered_weights(model):
    if model.settings.get('kind')=='mamba2':
        return np.concatenate([value.detach().cpu().numpy().ravel() for value in model.state_dict().values()]).astype('<f4')
    if model.settings.get('kind')=='imu_affine':
        return np.concatenate([model.gain.detach().cpu().numpy(),model.bias.detach().cpu().numpy()]).astype('<f4')
    if model.settings.get('kind')=='visual_imu_affine':
        return np.concatenate([model.gain.detach().cpu().numpy(),model.bias.detach().cpu().numpy(),
                               model.vision_weight.detach().cpu().numpy().ravel()]).astype('<f4')
    names=(['input.weight','input.bias','recurrent.weight'] if model.settings['hidden'] else [])
    names+=['head.weight','head.bias']
    state=model.state_dict()
    return np.concatenate([state[name].detach().cpu().numpy().ravel() for name in names]).astype('<f4')


def write_sources(model,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    s=model.settings
    if s.get('kind')=='visual_imu_affine' and (s['grid_width'],s['grid_height'],s['hidden'])!=(2,1,0):
        raise ValueError('The native 500 profile requires its 2x1 visual grid and no hidden layer')
    weights=ordered_weights(model)
    if not np.isfinite(weights).all():raise ValueError('Student checkpoint contains nonfinite weights')
    (output/'weights.bin').write_bytes(weights.tobytes())
    header=f'''/* Generated float32 weights; C23 #embed is supported by wasi-sdk 34. */
#define STUDENT_SSM {int(s.get('kind')=='mamba2')}
#define GRID_WIDTH {s['grid_width']}
#define GRID_HEIGHT {s['grid_height']}
#define GRID_SIZE {s['grid_width']*s['grid_height']}
#define FEATURES {model.features}
#define HIDDEN {s['hidden']}
#define AGGREGATE {int(s['aggregate'])}
#ifdef STUDENT_HOST
static const float *student_weights;
void student_set_weights(const float *data){{student_weights=data;}}
#define WEIGHTS student_weights
#else
static const union {{ unsigned char bytes[{weights.nbytes}]; float values[{weights.size}]; }} student_weights={{.bytes={{
#embed "weights.bin"
}}}};
#define WEIGHTS student_weights.values
#endif
'''
    if s.get('kind')=='mamba2':
        offsets={};offset=0
        for name,value in model.state_dict().items():offsets[name]=offset;offset+=value.numel()
        for name,value in offsets.items():
            header+=f'#define W_{name.upper().replace(".","_")} (WEIGHTS+{value})\n'
        for key,value in dict(LAYERS=s['layers'],D_STATE=s['d_state'],HEADDIM=s['headdim'],
                ATTENTION_DIM=s['attention_dim'],ATTENTION_HEADS=s['heads'],ATTENTION_WINDOW=s['window']).items():
            header+=f'#define {key} {value}\n'
        names=['in_proj.weight','conv1d.weight','conv1d.bias','dt_bias','A_log','D','norm.weight','out_proj.weight']
        rows=[[offsets[f'ssm.{i}.{name}'] for name in names]+[offsets[f'ssm_norm.{i}.weight'],offsets[f'ssm_norm.{i}.bias']] for i in range(s['layers'])]
        header+='static const int ssm_offsets[LAYERS][10]={'+','.join('{'+','.join(map(str,row))+'}' for row in rows)+'};\n'
        shutil.copy2(HERE/'native_ssm.h',output/'native_ssm.h')
    (output/'student_weights.h').write_text(header)
    source={'imu_affine':'native_100.c','visual_imu_affine':'native_500.c'}.get(s.get('kind'),'native.c')
    shutil.copy2(HERE/source,output/'solution.c')
    native=WORKSPACE/'runtime/native'
    (output/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.20)
project(student C)
set(CMAKE_C_STANDARD 23)
set(VIOBENCH_NATIVE_DIR "{native}" CACHE PATH "VIOBench native ABI")
include(${{VIOBENCH_NATIVE_DIR}}/viobench_reactor.cmake)
viobench_solution(student solution.c)
target_link_libraries(student PRIVATE m)
target_compile_options(student PRIVATE {'-O3' if s.get('kind') in ('imu_affine','visual_imu_affine') else '-O2'})
''')
    (output/'student.json').write_text(json.dumps(s,indent=2)+'\n')
    return output


def build(model,output,sdk=None):
    output=write_sources(model,output).resolve()
    sdk=Path(sdk or WORKSPACE/'build/toolchain/wasi-sdk').resolve()
    compiler=sdk/'bin/clang'
    if not compiler.is_file():
        raise FileNotFoundError(f'wasi-sdk clang not found at {compiler}; install the VIOBench native toolchain or pass --wasi-sdk')
    module=output/'candidate.wasm'
    optimize='-O3' if model.settings.get('kind') in ('imu_affine','visual_imu_affine') else '-O2'
    subprocess.run([str(compiler),'-std=c23',optimize,'-msimd128','-mexec-model=reactor',
                    '-I'+str(WORKSPACE/'runtime/native'),str(output/'solution.c'),'-lm',
                    '-Wl,--initial-memory=67108864','-Wl,--max-memory=2147483648',
                    '-Wl,--strip-debug','-o',str(module)],check=True)
    return module


def export_student(model,output,sdk=None,frames=1200):
    from viobench.tools.cli import run_with_large_stack
    from verify_limits import verify
    output=Path(output)
    candidate=build(model,output,sdk)
    report=output/'limits.json'
    result=run_with_large_stack(lambda:verify(candidate,frames,report,step_fuel=model.settings['fuel']))
    if not result['passed']:
        raise RuntimeError(f"Student exceeded its {model.settings['fuel']:,} fuel/frame tier; see {report}. Candidate was not promoted.")
    final=output/'student.wasm'
    candidate.replace(final)
    result['module']=str(final.resolve())
    result['profile_kind']='student resource check; not an accuracy evaluation'
    result['weights_sha256']=hashlib.sha256((output/'weights.bin').read_bytes()).hexdigest()
    report.write_text(json.dumps(result,indent=2)+'\n')
    (output/'submission_limits.json').write_text(json.dumps(result['limits'],indent=2)+'\n')
    print(f"Student ready: {final}; max fuel {result['fuel_step_max']:,}/{model.settings['fuel']:,}",flush=True)
    return final


def main():
    from students.model import BudgetStudent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,required=True,help='student profile directory with student.pt/student.json')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--wasi-sdk',type=Path)
    parser.add_argument('--verify-frames',type=int,default=1200)
    args=parser.parse_args()
    model=BudgetStudent(json.loads((args.checkpoints/'student.json').read_text()))
    model.load_state_dict(torch.load(args.checkpoints/'student.pt',map_location='cpu',weights_only=True))
    export_student(model.eval(),args.out or args.checkpoints/'native',args.wasi_sdk,args.verify_frames)


if __name__=='__main__':
    main()
