import sys, glob, pickle, numpy as np
from multiprocessing import Pool
from viobench.tools.runner.sandbox import Sandbox
def one(a):
    w,pk=a; cfg,data=pickle.load(open(pk,"rb")); sb=Sandbox(w); sb.init(cfg)
    F=np.array([sb.step(t,np.ascontiguousarray(f),imu).fuel for t,f,imu,g in data])
    k0=int(np.argmax(F>20000)); C=9
    return [int(F[k0+C+p::C].max()) for p in range(C)]
if __name__=="__main__":
    pks=sorted(glob.glob("/home/claude/tests/real_*.pkl"))
    for w in sys.argv[1:]:
        with Pool(10) as P: R=np.array(P.map(one,[(w,p) for p in pks]))
        print(w.split('/')[-1], "per-phase max over episodes:", R.max(0).tolist())
