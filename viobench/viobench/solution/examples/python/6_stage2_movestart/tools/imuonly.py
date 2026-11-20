import zipfile,glob,numpy as np,io,sys,os
from multiprocessing import Pool
def load(zp):
    z=zipfile.ZipFile(zp); n=[x for x in z.namelist() if x.endswith('imu0/data.csv')][0]; p=n[:-len('mav0/imu0/data.csv')]
    imu=np.loadtxt(io.BytesIO(z.read(n)),delimiter=',',skiprows=1); gt=np.loadtxt(io.BytesIO(z.read(p+'mav0/state_groundtruth_estimate0/data.csv')),delimiter=',',skiprows=1)
    m=min(len(imu),len(gt))//10*10; return imu[:m],gt[:m]
def run(imu,gt,ta,tv,tz,kyaw=1.0):
    # 20 Hz: mean gyro / mean a_z per frame. world = start body frame (z up). csv: t, w(3), a(3)
    n=len(imu)//10; w=imu[:,1:4].reshape(n,10,3).mean(1); az=imu[:,6].reshape(n,10).mean(1); DT=0.05
    bg=w[:20].mean(0); baz=az[:20].mean()-9.81
    e3=np.array([0,0,1.0]); yaw=0.0; v=np.zeros(3); p=np.zeros(3); out=np.zeros((n,3))
    Rm=np.eye(3)
    for k in range(20,n):
        wm=(w[k]-bg)*DT; a=np.linalg.norm(wm); K=np.array([[0,-wm[2],wm[1]],[wm[2],0,-wm[0]],[-wm[1],wm[0],0]])
        Rm=Rm@(np.eye(3)+K+0.5*K@K)
        if ta>0:   # leak the tilt toward level: rotate about (e3 x z) by a fraction
            e=Rm[:,2]; ax=np.cross(e,[0,0,1.0]); th=ax*(DT/ta); Kc=np.array([[0,-th[2],th[1]],[th[2],0,-th[0]],[-th[1],th[0],0]]); Rm=(np.eye(3)+Kc)@Rm
        if k%20==0:
            u,_,vt=np.linalg.svd(Rm); Rm=u@vt
        acc=Rm[:,2]*(az[k]-baz)+np.array([0,0,-9.81])
        v+=acc*DT
        if tv>0: v[:2]*=(1-DT/tv)
        if tz>0: v[2]*=(1-DT/tz)
        elif tz<0: v[2]=0
        p+=v*DT; out[k]=p
    pt=gt[::10,1:4][:n]
    e=out-pt; z0=np.sqrt((pt**2).sum(1).mean())
    return np.sqrt((e**2).sum(1).mean()),np.sqrt((e[:,:2]**2).sum(1).mean()),z0,np.sqrt((pt[:,:2]**2).sum(1).mean())
def job(zp):
    imu,gt=load(zp); r=[]
    for ta,tv,tz in CFG: r.append(run(imu,gt,ta,tv,tz))
    return r
CFG=[(0,0,-1),(5,2,-1),(5,4,-1),(10,2,-1),(10,4,-1),(10,8,-1),(20,4,-1),(20,8,-1),(3,2,-1),(3,4,-1),(10,4,1.0),(10,4,3.0)]
if __name__=='__main__':
    D=os.path.expanduser('~/mnt/Final_project/data/')
    for name,pat in [('agile med',D+'agile_dev_med/*.zip'),('agile hard',D+'agile_dev_hard/*.zip'),('heldout',D+'heldout/*.zip'),('public',D+'*.zip')]:
        zs=sorted(glob.glob(pat))[:12]
        with Pool(2) as pool: R=np.array(pool.map(job,zs))
        print(f"{name:10s} n={len(zs)} zero-output {R[:,0,2].mean():.2f} (horiz {R[:,0,3].mean():.2f})")
        for i,c in enumerate(CFG): print(f"   tilt-leak {c[0]:>4}s vel-leak {c[1]:>3}s z {c[2]:>4}: ATE {R[:,i,0].mean():6.2f}  horiz {R[:,i,1].mean():6.2f}")
