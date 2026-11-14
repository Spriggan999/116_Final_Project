"""Exact sliding-window self-attention without a full long-sequence mask."""
import torch
import torch.nn.functional as F


def local_attention(module,x,window,chunk_size=512):
    """Reuse MultiheadAttention weights; bound query/key blocks during training."""
    if window<1 or type(chunk_size) is not int or chunk_size<0:
        raise ValueError('Invalid local attention window/chunk')
    batch,ticks,width=x.shape
    if not chunk_size or ticks<=chunk_size:
        positions=torch.arange(ticks,device=x.device)
        lag=positions[:,None]-positions[None,:]
        return module(x,x,x,attn_mask=(lag<0)|(lag>=window),need_weights=False)[0]
    if not module.batch_first or module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError('Local attention requires ordinary batch-first self-attention')
    heads=module.num_heads
    q,k,v=F.linear(x,module.in_proj_weight,module.in_proj_bias).chunk(3,-1)
    q,k,v=(value.reshape(batch,ticks,heads,width//heads).transpose(1,2) for value in (q,k,v))
    parts=[]
    for start in range(0,ticks,chunk_size):
        end=min(ticks,start+chunk_size)
        first=max(0,start-window+1)
        lag=torch.arange(start,end,device=x.device)[:,None]-torch.arange(first,end,device=x.device)[None,:]
        # SDPA True means allowed; MultiheadAttention True means masked out.
        allowed=(lag>=0)&(lag<window)
        parts.append(F.scaled_dot_product_attention(q[:,:,start:end],k[:,:,first:end],v[:,:,first:end],
                     attn_mask=allowed,dropout_p=module.dropout if module.training else 0.))
    merged=torch.cat(parts,2).transpose(1,2).reshape(batch,ticks,width)
    return module.out_proj(merged)


def observation_ages(observed,dt,dtype):
    positions=torch.arange(observed.shape[1],device=observed.device)[None]
    latest=torch.where(observed,positions,0).cummax(1).values
    return (positions-latest).to(dtype)*dt[:,None]
