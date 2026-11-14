"""Supervised state losses from the notebook."""
import math
import torch
import torch.nn.functional as F
from performance import fp32_physics


def geodesic(rot_gt,rot_pred):
    d=rot_gt.transpose(-1,-2)@rot_pred
    cc=((d.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1+1e-5,1-1e-5)
    return torch.acos(cc)


@fp32_physics
def mean_ate(predicted_position, true_position):
    """Mean Euclidean position error in metres, in the existing local frame.

    No scale/rigid alignment is fitted. This is a mean distance, rather than
    coordinate-wise position RMSE or the commonly reported ATE RMSE.
    """
    return torch.linalg.vector_norm(predicted_position-true_position,dim=-1).mean()


@fp32_physics
def losses(pred,b,device,metrics=True):
    pos_gt=b['pos'].to(device,non_blocking=True);vel_gt=b['vel'].to(device,non_blocking=True)
    rot_gt=b['rot'].to(device,non_blocking=True);omega_gt=b['omega'].to(device,non_blocking=True)
    pos=F.smooth_l1_loss(pred['pos'],pos_gt,beta=.2)
    vel=F.smooth_l1_loss(pred['vel'],vel_gt,beta=.2)
    rot=(geodesic(rot_gt,pred['rot'])**2).mean()
    omega=F.smooth_l1_loss(pred['omega'],omega_gt,beta=.2)
    total=3*pos+vel+rot+0.1*omega
    if not metrics:return total,{}
    return total,dict(mean_ate_m=float(mean_ate(pred['pos'],pos_gt).detach()),
                      pos_rmse=float(torch.sqrt(((pred['pos']-pos_gt)**2).mean()).detach()),
                      vel_rmse=float(torch.sqrt(((pred['vel']-vel_gt)**2).mean()).detach()),
                      rot_deg=float((geodesic(rot_gt,pred['rot']).mean()*180/math.pi).detach()),
                      omega_rmse=float(torch.sqrt(((pred['omega']-omega_gt)**2).mean()).detach()))
