import sys, glob, pickle, numpy as np
from multiprocessing import Pool
from viobench.tools.runner.sandbox import Sandbox
from scipy.spatial.transform import Rotation as Rot
W=9
def one(a):
    w,pk=a; cfg,data=pickle.load(open(pk,"rb")); sb=Sandbox(w); sb.init(cfg)
    E=[];G=[]
    for t,f,imu,gt in data:
        r=sb.step(t,np.ascontiguousarray(f),imu); E.append(np.r_[r.pose,r.twist]); G.append(gt)
    E=np.array(E);G=np.array(G)
    Re=Rot.from_quat(E[:,3:7]); Rg=Rot.from_quat(G[:,3:7])
    out=[]
    for a in range(20,len(E)-W,W):
        b=a+W
        de=E[b,:3]-E[a,:3]; dg=G[b,:3]-G[a,:3]
        # both in the current camera frame (removes attitude drift)
        ce=Re[b].inv().apply(de); cg=Rg[b].inv().apply(dg)
        ng=np.linalg.norm(cg); ne=np.linalg.norm(ce)
        if ng<0.05 or ne<1e-6: continue
        ang=np.degrees(np.arccos(np.clip(ce@cg/(ne*ng),-1,1)))
        out.append((ang,ng, ce/ne, cg/ng))
    return out
if __name__=="__main__":
    pks=sorted(glob.glob("/home/claude/tests/real_*.pkl"))
    with Pool(10) as P: R=P.map(one,[(sys.argv[1],p) for p in pks])
    A=np.array([o[0] for r in R for o in r]); N=np.array([o[1] for r in R for o in r])
    print("windows",len(A)," median angle",np.median(A).round(1)," mean",A.mean().round(1))
    for th in [10,20,45,90,135]: print(f" >{th} deg: {np.mean(A>th)*100:.1f}%")
    for lo,hi in [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,9)]:
        m=(N>=lo)&(N<hi); print(f" |d| {lo}-{hi} m: n={m.sum()} median {np.median(A[m]):.1f} deg, >90: {np.mean(A[m]>90)*100:.1f}%")
    # axis breakdown of error: which camera-frame components
    CE=np.array([o[2] for r in R for o in r]); CG=np.array([o[3] for r in R for o in r])
    print(" mean |gt dir| per axis (x,y,z):",np.abs(CG).mean(0).round(2)," mean |err| per axis:",np.abs(CE-CG).mean(0).round(2))
    # temporal structure of the lateral error per episode
    for pk,r in zip(pks,R):
        ex=np.array([o[2][0]-o[3][0] for o in r])
        if len(ex)<10: continue
        ac=np.corrcoef(ex[:-1],ex[1:])[0,1]
        print(pk.split('_')[-1][:6], f"lateral err mean {ex.mean():+.3f} rms {np.sqrt((ex**2).mean()):.3f} lag-1 autocorr {ac:+.2f}")
