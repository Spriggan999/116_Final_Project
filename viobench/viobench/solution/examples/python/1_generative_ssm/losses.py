"""Supervised state losses from the notebook."""
import math
import torch
import torch.nn.functional as F
from performance import fp32_physics


def geodesic(rot_gt,rot_pred):
    """Rotation angle of R_gtᵀ R_pred, accurate and differentiable near zero.

    atan2(|vee(D-Dᵀ)|/2, (tr D-1)/2). The former clamped acos had a 0.256° floor
    with zero gradient below it; the 1e-12 keeps identity (tick 0) finite.
    """
    d=rot_gt.transpose(-1,-2)@rot_pred
    cosine=(d.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    skew=torch.stack((d[...,2,1]-d[...,1,2],d[...,0,2]-d[...,2,0],d[...,1,0]-d[...,0,1]),-1)
    sine=torch.sqrt(skew.square().sum(-1)/4+1e-12)
    return torch.atan2(sine,cosine)


@fp32_physics
def losses(pred,b,device,metrics=True,components=False):
    pos_gt=b['pos'].to(device,non_blocking=True);vel_gt=b['vel'].to(device,non_blocking=True)
    rot_gt=b['rot'].to(device,non_blocking=True);omega_gt=b['omega'].to(device,non_blocking=True)
    pos=F.smooth_l1_loss(pred['pos'],pos_gt,beta=.2)
    vel=F.smooth_l1_loss(pred['vel'],vel_gt,beta=.2)
    rot=(geodesic(rot_gt,pred['rot'])**2).mean()
    omega=F.smooth_l1_loss(pred['omega'],omega_gt,beta=.2)
    total=3*pos+vel+rot+0.1*omega
    terms={}
    if components:
        from loss_tracking import weighted_terms
        terms=weighted_terms(dict(position=pos,velocity=vel,rotation=rot,angular_velocity=omega),
                             dict(position=3.,velocity=1.,rotation=1.,angular_velocity=.1))
    if not metrics:return total,terms
    return total,terms|dict(pos_rmse=float(torch.sqrt(((pred['pos']-pos_gt)**2).mean()).detach()),
                      vel_rmse=float(torch.sqrt(((pred['vel']-vel_gt)**2).mean()).detach()),
                      rot_deg=float((geodesic(rot_gt,pred['rot']).mean()*180/math.pi).detach()),
                      omega_rmse=float(torch.sqrt(((pred['omega']-omega_gt)**2).mean()).detach()))
