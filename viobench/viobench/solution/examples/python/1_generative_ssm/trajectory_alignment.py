"""Stable Sim(3)-aligned camera-position RMSE auxiliary objective."""
import torch
from performance import fp32_physics


@fp32_physics
def alignment_objective(prediction,truth,epsilon=1e-12):
    position=prediction['pos'].float()
    target=truth['pos'].to(position).float()
    lever=truth.get('camera_position_in_body')
    if lever is not None:
        lever=lever.to(position)[:,None,:,None]
        eye=torch.eye(3,device=position.device,dtype=position.dtype)
        position=position+((prediction['rot'].float()-eye)@lever).squeeze(-1)
        target=target+((truth['rot'].to(position)-eye)@lever).squeeze(-1)
    if position.shape[1]<3:
        zero=position.sum()*0
        return zero,dict(alignment_scale_mean=zero.detach()+1,alignment_scale_min=zero.detach()+1,
                         alignment_scale_max=zero.detach()+1,alignment_collapsed_fraction=zero.detach(),
                         alignment_insufficient_samples_fraction=zero.detach()+1)
    # Fit the minimizing transform without differentiating SVD singular vectors.
    # For a minimizing fit, the envelope theorem gives the same residual gradient
    # away from degeneracies, while repeated singular values remain finite.
    with torch.no_grad():
        source=position.detach().double();goal=target.detach().double()
        source_mean=source.mean(1,keepdim=True);goal_mean=goal.mean(1,keepdim=True)
        x=source-source_mean;y=goal-goal_mean
        variance=x.square().sum(-1).mean(-1)
        u,s,vh=torch.linalg.svd(y.transpose(1,2)@x/source.shape[1])
        sign=torch.ones_like(s)
        sign[:,-1]=torch.where(torch.linalg.det(u)*torch.linalg.det(vh)<0,-1.,1.)
        rotation=(u*sign[:,None,:])@vh
        scale=(s*sign).sum(-1)/variance.clamp_min(epsilon)
        collapsed=variance<=epsilon
        identity=torch.eye(3,device=source.device,dtype=source.dtype).expand_as(rotation)
        rotation=torch.where(collapsed[:,None,None],identity,rotation)
        scale=torch.where(collapsed,torch.ones_like(scale),scale)
        translation=goal_mean-scale[:,None,None]*(source_mean@rotation.transpose(1,2))
        rotation=rotation.to(position);scale=scale.to(position);translation=translation.to(position)
    aligned=scale[:,None,None]*(position@rotation.transpose(1,2))+translation
    per_trajectory=torch.sqrt((aligned-target).square().sum(-1).mean(-1)+epsilon)-epsilon**.5
    loss=per_trajectory.mean()
    return loss,dict(alignment_scale_mean=scale.detach().mean(),
                     alignment_scale_min=scale.detach().min(),alignment_scale_max=scale.detach().max(),
                     alignment_collapsed_fraction=collapsed.float().mean(),
                     alignment_insufficient_samples_fraction=loss.detach().new_zeros(()))
