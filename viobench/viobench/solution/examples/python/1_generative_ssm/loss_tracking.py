"""Detached, sample-weighted raw losses, coefficients and objective contributions."""
import json
from pathlib import Path


def weighted_terms(raw,weights,prefix='',scale=1.):
    result={}
    for name,value in raw.items():
        weight=weights[name]*scale
        result[prefix+name+'_raw']=value.detach()
        result[prefix+name+'_weight']=weight
        result[prefix+name+'_weighted']=value.detach()*weight
    return result


class LossAverages:
    def __init__(self):self.sums={};self.count=0

    def update(self,terms,count):
        self.count+=count
        for key,value in terms.items():
            if hasattr(value,'detach'):value=value.detach()
            self.sums[key]=self.sums.get(key,0)+value*count

    def result(self):
        return {key:float(value/self.count) for key,value in self.sums.items()} if self.count else {}


def record_components(output,phase,epoch,training,validation=None):
    train=training.result();val=validation.result() if validation is not None else {}
    path=Path(output)/'loss_components.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps(dict(phase=phase,epoch=epoch,train=train,validation=val))+'\n')
    return {phase+'/train_'+key:value for key,value in train.items()}|{phase+'/val_'+key:value for key,value in val.items()}


def embedding_terms(components,variance_weight=1.,covariance_weight=.01,scale=1.):
    raw={key:components[key] for key in ('prediction','content','variance','covariance')}
    return weighted_terms(raw,dict(prediction=1.,content=1.,variance=variance_weight,covariance=covariance_weight),scale=scale)


def rgb_objective(prediction,target,config,flow=None):
    import torch.nn.functional as F
    l1=F.l1_loss(prediction,target)
    dx=(prediction[:,:,:,1:]-prediction[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1])
    dy=(prediction[:,:,1:,:]-prediction[:,:,:-1,:])-(target[:,:,1:,:]-target[:,:,:-1,:])
    raw=dict(rgb=l1,edge_x=dx.abs().mean(),edge_y=dy.abs().mean(),
             flow_x=l1.new_zeros(()),flow_y=l1.new_zeros(()))
    if flow is not None:
        raw['flow_x']=(flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()
        raw['flow_y']=(flow[:,:,1:,:]-flow[:,:,:-1,:]).abs().mean()
    edge=config.FRAME_EDGE_LOSS_WEIGHT if config.USE_FRAME_EDGE_LOSS else 0.
    smooth=config.FRAME_FLOW_SMOOTHNESS_WEIGHT if config.USE_GENERATIVE_FRAME_MODEL and config.USE_FLOW_SMOOTHNESS_LOSS else 0.
    weights=dict(rgb=1.,edge_x=edge,edge_y=edge,flow_x=smooth,flow_y=smooth)
    # Preserve the original grouped addition and objective gradients.
    total=l1
    if edge:total=total+edge*(raw['edge_x']+raw['edge_y'])
    if smooth:total=total+smooth*(raw['flow_x']+raw['flow_y'])
    return total,weighted_terms(raw,weights)
