/* Official Mamba-2 equations, streamed at camera rate. No growing history.
 * Weight layout is generated from the official module's state_dict.
 */
#define INNER (2*HIDDEN)
#define NHEADS (INNER/HEADDIM)
#define CONV_DIM (INNER+2*D_STATE)
#define PROJ_DIM (2*INNER+2*D_STATE+NHEADS)
static float conv_cache[LAYERS][CONV_DIM][4];
static float state_cache[LAYERS][INNER][D_STATE];
static float decay_rates[LAYERS][NHEADS];
#if ATTENTION_DIM > 0
static float keys[ATTENTION_WINDOW][ATTENTION_DIM],values[ATTENTION_WINDOW][ATTENTION_DIM];
static int cache_count,cache_cursor;
#endif
static void linear_f(const float *w,const float *bias,const float *x,float *y,int rows,int cols){
    for(int i=0;i<rows;i++){
        float sum=bias?bias[i]:0;
        for(int j=0;j<cols;j++)sum+=w[i*cols+j]*x[j];
        y[i]=sum;
    }
}
static void norm_f(float *x,const float *w,const float *bias,int n){
    float mean=0,variance=0;
    for(int i=0;i<n;i++)mean+=x[i];mean/=n;
    for(int i=0;i<n;i++){float z=x[i]-mean;variance+=z*z;}
    float scale=1.f/sqrtf(variance/n+1e-5f);
    for(int i=0;i<n;i++)x[i]=(x[i]-mean)*scale*w[i]+bias[i];
}
static float silu_f(float x){return x/(1.f+expf(-x));}
static void ssm_reset(void){
    memset(conv_cache,0,sizeof(conv_cache));memset(state_cache,0,sizeof(state_cache));
    for(int l=0;l<LAYERS;l++)for(int h=0;h<NHEADS;h++)decay_rates[l][h]=-expf(WEIGHTS[ssm_offsets[l][4]+h]);
#if ATTENTION_DIM > 0
    memset(keys,0,sizeof(keys));memset(values,0,sizeof(values));cache_count=cache_cursor=0;
#endif
}
static void ssm_prediction(const float *features,float *correction){
    float token[HIDDEN],projected[PROJ_DIM],xbc[CONV_DIM],y[INNER],out[HIDDEN];
    linear_f(W_INPUT_WEIGHT,W_INPUT_BIAS,features,token,HIDDEN,FEATURES);
    for(int i=0;i<HIDDEN;i++)token[i]=tanhf(token[i]);
    for(int l=0;l<LAYERS;l++){
        const int *o=ssm_offsets[l];
        linear_f(WEIGHTS+o[0],NULL,token,projected,PROJ_DIM,HIDDEN);
        for(int c=0;c<CONV_DIM;c++){
            float *cache=conv_cache[l][c];cache[0]=cache[1];cache[1]=cache[2];cache[2]=cache[3];cache[3]=projected[INNER+c];
            float sum=WEIGHTS[o[2]+c];for(int j=0;j<4;j++)sum+=cache[j]*WEIGHTS[o[1]+4*c+j];
            xbc[c]=silu_f(sum);
        }
        for(int h=0;h<NHEADS;h++){
            float raw=projected[INNER+CONV_DIM+h]+WEIGHTS[o[3]+h];
            float dt=fmaxf(raw,0.f)+log1pf(expf(-fabsf(raw)));
            float decay=expf(dt*decay_rates[l][h]);
            for(int p=0;p<HEADDIM;p++){
                int i=h*HEADDIM+p;float sum=0;
                for(int n=0;n<D_STATE;n++){
                    float state=state_cache[l][i][n]*decay+dt*xbc[i]*xbc[INNER+n];
                    state_cache[l][i][n]=state;sum+=state*xbc[INNER+D_STATE+n];
                }
                y[i]=(sum+WEIGHTS[o[5]+h]*xbc[i])*silu_f(projected[i]);
            }
        }
        float square=0;for(int i=0;i<INNER;i++)square+=y[i]*y[i];
        float scale=1.f/sqrtf(square/INNER+1e-5f);
        for(int i=0;i<INNER;i++)y[i]*=scale*WEIGHTS[o[6]+i];
        linear_f(WEIGHTS+o[7],NULL,y,out,HIDDEN,INNER);
        for(int i=0;i<HIDDEN;i++)token[i]+=out[i];
        norm_f(token,WEIGHTS+o[8],WEIGHTS+o[9],HIDDEN);
    }
#if ATTENTION_DIM > 0
    float a[ATTENTION_DIM],qkv[3*ATTENTION_DIM],attended[ATTENTION_DIM],merged[ATTENTION_DIM];
#if ATTENTION_DIM != HIDDEN
    linear_f(W_ATTN_INPUT_WEIGHT,W_ATTN_INPUT_BIAS,token,a,ATTENTION_DIM,HIDDEN);
#else
    memcpy(a,token,sizeof(a));
#endif
    linear_f(W_ATTN_IN_PROJ_WEIGHT,W_ATTN_IN_PROJ_BIAS,a,qkv,3*ATTENTION_DIM,ATTENTION_DIM);
    memcpy(keys[cache_cursor],qkv+ATTENTION_DIM,sizeof(a));memcpy(values[cache_cursor],qkv+2*ATTENTION_DIM,sizeof(a));
    if(cache_count<ATTENTION_WINDOW)cache_count++;
    cache_cursor=(cache_cursor+1)%ATTENTION_WINDOW;
    const int dim=ATTENTION_DIM/ATTENTION_HEADS;
    for(int h=0;h<ATTENTION_HEADS;h++){
        float scores[ATTENTION_WINDOW],maximum=-INFINITY,sum=0;
        for(int t=0;t<cache_count;t++){
            int slot=cache_count==ATTENTION_WINDOW?(cache_cursor+t)%ATTENTION_WINDOW:t;
            float v=0;for(int j=0;j<dim;j++)v+=qkv[h*dim+j]*keys[slot][h*dim+j];
            scores[t]=v/sqrtf((float)dim);maximum=fmaxf(maximum,scores[t]);
        }
        for(int t=0;t<cache_count;t++){scores[t]=expf(scores[t]-maximum);sum+=scores[t];}
        for(int j=0;j<dim;j++){
            float v=0;for(int t=0;t<cache_count;t++){
                int slot=cache_count==ATTENTION_WINDOW?(cache_cursor+t)%ATTENTION_WINDOW:t;
                v+=(scores[t]/sum)*values[slot][h*dim+j];
            }
            attended[h*dim+j]=v;
        }
    }
    linear_f(W_ATTN_OUT_PROJ_WEIGHT,W_ATTN_OUT_PROJ_BIAS,attended,merged,ATTENTION_DIM,ATTENTION_DIM);
#if ATTENTION_DIM != HIDDEN
    linear_f(W_ATTN_OUTPUT_WEIGHT,W_ATTN_OUTPUT_BIAS,merged,out,HIDDEN,ATTENTION_DIM);
#else
    memcpy(out,merged,sizeof(out));
#endif
    for(int i=0;i<HIDDEN;i++)token[i]+=out[i];
    norm_f(token,W_ATTN_NORM_WEIGHT,W_ATTN_NORM_BIAS,HIDDEN);
#endif
    linear_f(W_HEAD_WEIGHT,W_HEAD_BIAS,token,correction,9,HIDDEN);
    for(int i=0;i<9;i++)correction[i]=.25f*tanhf(correction[i]);
}
