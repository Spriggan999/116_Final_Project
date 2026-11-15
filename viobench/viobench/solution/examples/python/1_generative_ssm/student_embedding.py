"""Training-only EMA targets and JEPA-style objectives for budget students."""
from copy import copy

import torch
import torch.nn.functional as F
from performance import amp_disabled

PREFIX='_jepa_target_'


def target_key(name):
    return PREFIX+name.replace('.','__')


def initialize_target(model):
    # Buffers travel with checkpoints and reversible scheduling probes, but
    # neither become optimizer parameters nor enter the deployment weight blob.
    for name,value in model.vision.state_dict().items():
        model.register_buffer(target_key(name),value.detach().clone())


def hydrate_target_state(model,state):
    state=copy(state)
    for name in model.vision.state_dict():
        key=target_key(name)
        if key not in state:state[key]=state['vision.'+name].detach().clone()
    return state


@torch.no_grad()
def update_target(model,decay):
    for name,value in model.vision.state_dict().items():
        getattr(model,target_key(name)).lerp_(value.detach(),1-decay)


def objective(model,inputs,std_target=.1,variance_weight=1.,covariance_weight=.01):
    images,imu,ticks=inputs['images'],inputs['imu'],inputs['ticks']
    gray=model.image_features(images)
    if model.settings.get('kind')=='visual_imu_affine':
        sensor=imu[:,ticks]
        interval=imu.new_full(sensor.shape[:2],.05);interval[:,0]=0.
    else:
        means=torch.stack([imu[:,lo+1:hi+1].mean(1) for lo,hi in zip(ticks[:-1],ticks[1:])],1)
        sensor=torch.cat([torch.zeros_like(means[:,:1]),means],1)
        spans=imu.new_tensor([0,*[hi-lo for lo,hi in zip(ticks[:-1],ticks[1:])]])
        interval=spans[None]*inputs['dt'][:,None]
    online,innovation=model.predicted_visual_features(gray,sensor,interval)
    predicted=online-innovation
    with torch.no_grad(),amp_disabled():
        state={name:getattr(model,target_key(name)) for name in model.vision.state_dict()}
        target=torch.func.functional_call(model.vision,state,(gray.float(),))
    # The initial capture has no prediction history and is never a target.
    prediction=F.smooth_l1_loss(predicted[:,1:],target[:,1:],beta=.1)
    content=F.smooth_l1_loss(online[:,1:],target[:,1:],beta=.1)
    values=online.reshape(-1,online.shape[-1]).float()
    centered=values-values.mean(0)
    variance=F.relu(std_target-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
    covariance=centered.T@centered/max(1,len(values)-1)
    off=covariance-torch.diag_embed(covariance.diagonal())
    covariance=off.square().sum()/values.shape[-1]
    total=prediction+content+variance_weight*variance+covariance_weight*covariance
    return total,dict(embedding_prediction=prediction,embedding_content=content,
                      embedding_variance=variance,embedding_covariance=covariance,embedding_loss=total)
