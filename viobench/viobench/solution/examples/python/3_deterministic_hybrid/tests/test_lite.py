"""Lite baseline parity, profile isolation and compact inference export checks."""
import ctypes as ct
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import lite_profile
import train_fresh
from train import MLP,Tracker,export
from viobench.tools.abi.config import default_guest_config


class LiteTests(unittest.TestCase):
    def test_snapshot_is_exact_and_generation_is_reproducible(self):
        self.assertEqual((HERE/'lite/base/solution.cpp').read_bytes(),(HERE.parent/'2_deterministic/solution_lite.cpp').read_bytes())
        lite_profile.generate();before=(HERE/'lite/common/estimator.inc').read_bytes()
        lite_profile.generate();self.assertEqual(before,(HERE/'lite/common/estimator.inc').read_bytes())

    def test_lite_network_cpp_matches_pytorch(self):
        rng=np.random.default_rng(116);torch.manual_seed(116)
        names=lite_profile.NAMES
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode,name in enumerate(names,1):
                with self.subTest(name=name):
                    inputs=6 if mode==1 else 8;outputs=2 if mode in [2,3] else 3 if mode==4 else 1
                    net=Tracker(hidden=4,channels=2) if mode==2 else MLP(inputs,outputs,hidden=4)
                    count=18 if mode==2 else inputs
                    mean=np.zeros(count,np.float32);std=np.ones(count,np.float32)
                    export(name,net,mean,std,{},root)
                    source=root/(name+'.cpp');libpath=root/(name+'.so')
                    source.write_text(f'#define HYBRID_MODE {mode}\n#include "{root/name}/weights.h"\n'
                                      f'#include "{HERE}/lite/common/network.h"\n'
                                      'extern "C" void infer(const float* x,float* y){'+('neural_track' if mode==2 else 'neural_mlp')+'(x,y);}\n')
                    subprocess.run(['g++','-O2','-shared','-fPIC',str(source),'-o',str(libpath)],check=True)
                    lib=ct.CDLL(str(libpath));lib.infer.argtypes=[ct.c_void_p,ct.c_void_p]
                    for _ in range(5):
                        x=rng.normal(size=128 if mode==2 else inputs).astype(np.float32)
                        with torch.no_grad():expected=net(torch.tensor(x[None])).numpy()[0]
                        actual=np.empty_like(expected);lib.infer(x.ctypes.data,actual.ctypes.data)
                        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)

    def test_all_lite_modes_compile_as_wasm(self):
        lite_profile.generate()
        sdk=HERE.parents[2]/'build/toolchain/wasi-sdk/bin/clang++'
        if not sdk.exists():self.skipTest('WASI SDK is not installed')
        abi=HERE.parents[2]/'runtime/native'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode,name in enumerate(lite_profile.NAMES,1):
                with self.subTest(name=name):
                    inputs=6 if mode==1 else 8;outputs=2 if mode in [2,3] else 3 if mode==4 else 1
                    net=Tracker(hidden=4,channels=2) if mode==2 else MLP(inputs,outputs,hidden=4)
                    count=18 if mode==2 else inputs
                    export(name,net,np.zeros(count,np.float32),np.ones(count,np.float32),{'compile_test_only':True},root)
                    source=root/(name+'.cpp');wasm=root/(name+'.wasm')
                    source.write_text(f'#define HYBRID_MODE {mode}\n#include "{root/name}/weights.h"\n'
                                      f'#include "{HERE}/lite/common/estimator.inc"\n')
                    subprocess.run([str(sdk),'-O2','-msimd128','-fno-exceptions','-I'+str(abi),str(source),
                                    '-mexec-model=reactor','-Wl,--initial-memory=67108864','-o',str(wasm)],check=True)
                    self.assertEqual(wasm.read_bytes()[:4],b'\x00asm')

    def test_collector_preserves_lite_baseline_trajectory(self):
        import json
        lite_profile.generate();abi=HERE.parents[2]/'runtime/native'
        rng=np.random.default_rng(116)
        with tempfile.TemporaryDirectory() as folder:
            libs=[]
            for label,source in [('original',HERE/'lite/base/solution.cpp'),('collector',HERE/'lite/common/collect.cpp')]:
                path=Path(folder)/(label+'.so')
                subprocess.run(['g++','-O2','-shared','-fPIC','-I'+str(abi),str(source),'-o',str(path)],check=True)
                lib=ct.CDLL(str(path));lib.init.argtypes=[ct.c_char_p,ct.c_uint32]
                lib.step.argtypes=[ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_int32,ct.c_void_p]
                config=json.dumps(default_guest_config()).encode();self.assertEqual(lib.init(config,len(config)),0)
                libs.append(lib)
            texture=rng.integers(0,256,(240,320,3),dtype=np.uint8)
            for i in range(100):
                frame=np.ascontiguousarray(np.roll(texture,max(0,i-20)//2,axis=1))
                times=np.arange(i*10-9,i*10+1)*.005
                imu=np.column_stack([times,rng.normal(0,.01,(10,3))+[0,0,9.81],rng.normal(0,.001,(10,3))]).astype(np.float64)
                states=[]
                for lib in libs:
                    out=np.empty(13,np.float64)
                    lib.step(i*.05,frame.ctypes.data,imu.ctypes.data,len(imu),out.ctypes.data)
                    states.append(out)
                np.testing.assert_allclose(states[0],states[1],rtol=0,atol=1e-10)

    def test_profile_configuration_is_isolated(self):
        try:
            train_fresh.configure_profile('lite')
            self.assertEqual(train_fresh.HERE,HERE/'lite')
            self.assertEqual(train_fresh.fit,lite_profile.fit_lite)
            self.assertEqual(train_fresh.TASK_FACTORY,lite_profile.make_tasks_lite)
        finally:train_fresh.configure_profile('full')
        self.assertEqual(train_fresh.HERE,HERE)

if __name__=='__main__':unittest.main()
