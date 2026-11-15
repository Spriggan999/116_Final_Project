/* Learned encoder -> causal residual embedding forecast -> state features. */
static float anchor_embedding[EMBEDDING_DIM],older_embedding[EMBEDDING_DIM],prior_interval;
static void frontend_linear(const float *w,const float *bias,const float *x,float *y,int rows,int cols){
    for(int i=0;i<rows;i++){
        float sum=bias[i];for(int j=0;j<cols;j++)sum+=w[i*cols+j]*x[j];y[i]=sum;
    }
}
static void frontend_reset(void){
    memset(anchor_embedding,0,sizeof(anchor_embedding));memset(older_embedding,0,sizeof(older_embedding));prior_interval=0;
}
#if !MICRO_FRONTEND
#define V1_H ((GRID_HEIGHT+2*VISION_PADDING-3)/2+1)
#define V1_W ((GRID_WIDTH+2*VISION_PADDING-3)/2+1)
#if VISION_SECOND_KERNEL == 1
#define V2_H V1_H
#define V2_W V1_W
#else
#define V2_H ((V1_H+1)/2)
#define V2_W ((V1_W+1)/2)
#endif
static void frontend_conv(const float *w,const float *bias,const float *x,float *y,
                          int ci,int co,int ih,int iw,int oh,int ow,int k,int stride,int padding){
    for(int c=0;c<co;c++)for(int yy=0;yy<oh;yy++)for(int xx=0;xx<ow;xx++){
        float sum=bias[c];
        for(int i=0;i<ci;i++)for(int ky=0;ky<k;ky++)for(int kx=0;kx<k;kx++){
            int sy=yy*stride+ky-padding,sx=xx*stride+kx-padding;
            if(sy>=0&&sy<ih&&sx>=0&&sx<iw)sum+=w[((c*ci+i)*k+ky)*k+kx]*x[(i*ih+sy)*iw+sx];
        }
        y[(c*oh+yy)*ow+xx]=fmaxf(0.f,sum);
    }
}
#endif
static void frontend_encode(const float *gray,float *encoded){
#if MICRO_CNN
    float pooled[VISION_C1]={0};
    for(int c=0;c<VISION_C1;c++)for(int y=0;y<GRID_HEIGHT;y+=2)for(int x=0;x<GRID_WIDTH;x+=2){
        const float *w=W_VISION_CONV1_WEIGHT+4*c;
        float v=W_VISION_CONV1_BIAS[c];
        v+=w[0]*gray[y*GRID_WIDTH+x];v+=w[1]*gray[y*GRID_WIDTH+x+1];
        v+=w[2]*gray[(y+1)*GRID_WIDTH+x];v+=w[3]*gray[(y+1)*GRID_WIDTH+x+1];
        pooled[c]+=(v<0.f?v*.1f:v);
    }
    for(int c=0;c<VISION_C1;c++)pooled[c]/=(GRID_HEIGHT/2)*(GRID_WIDTH/2);
    frontend_linear(W_VISION_PROJECT_WEIGHT,W_VISION_PROJECT_BIAS,pooled,encoded,EMBEDDING_DIM,VISION_C1);
    for(int i=0;i<EMBEDDING_DIM;i++)if(encoded[i]<0.f)encoded[i]*=.1f;
#elif MICRO_FRONTEND
    frontend_linear(W_VISION_PROJECT_WEIGHT,W_VISION_PROJECT_BIAS,gray,encoded,EMBEDDING_DIM,GRID_SIZE);
    for(int i=0;i<EMBEDDING_DIM;i++)if(encoded[i]<0.f)encoded[i]*=.1f;
#else
    float first[VISION_C1*V1_H*V1_W],second[VISION_C2*V2_H*V2_W],pooled[VISION_C2];
    frontend_conv(W_VISION_CONV1_WEIGHT,W_VISION_CONV1_BIAS,gray,first,1,VISION_C1,
                  GRID_HEIGHT,GRID_WIDTH,V1_H,V1_W,3,2,VISION_PADDING);
    frontend_conv(W_VISION_CONV2_WEIGHT,W_VISION_CONV2_BIAS,first,second,VISION_C1,VISION_C2,
                  V1_H,V1_W,V2_H,V2_W,VISION_SECOND_KERNEL,VISION_SECOND_KERNEL==1?1:2,VISION_SECOND_KERNEL==1?0:1);
    for(int c=0;c<VISION_C2;c++){
        float sum=0;for(int i=0;i<V2_H*V2_W;i++)sum+=second[c*V2_H*V2_W+i];
        pooled[c]=sum/(V2_H*V2_W);
    }
    frontend_linear(W_VISION_PROJECT_WEIGHT,W_VISION_PROJECT_BIAS,pooled,encoded,EMBEDDING_DIM,VISION_C2);
    for(int i=0;i<EMBEDDING_DIM;i++)encoded[i]=tanhf(encoded[i]);
#endif
}
static void frontend_pair(const float *gray,const float *sensor,float interval,int ready,float *encoded,float *innovation){
    frontend_encode(gray,encoded);
    float predicted[EMBEDDING_DIM];
    if(!ready)memcpy(predicted,encoded,sizeof(predicted));
    else{
#if MICRO_FRONTEND
        const int condition=3;
#else
        const int condition=6;
#endif
        float features[3*EMBEDDING_DIM+8],hidden[PREDICTOR_HIDDEN],residual[EMBEDDING_DIM];
        for(int i=0;i<EMBEDDING_DIM;i++){
            features[i]=older_embedding[i];features[EMBEDDING_DIM+i]=anchor_embedding[i];
            features[2*EMBEDDING_DIM+i]=anchor_embedding[i]-older_embedding[i];
        }
        for(int i=0;i<condition;i++)features[3*EMBEDDING_DIM+i]=sensor[i]/10.f;
        features[3*EMBEDDING_DIM+condition]=interval;features[3*EMBEDDING_DIM+condition+1]=prior_interval;
        frontend_linear(W_EMBEDDING_PREDICTOR_0_WEIGHT,W_EMBEDDING_PREDICTOR_0_BIAS,features,hidden,
                        PREDICTOR_HIDDEN,3*EMBEDDING_DIM+condition+2);
        for(int i=0;i<PREDICTOR_HIDDEN;i++){
#if MICRO_FRONTEND
            if(hidden[i]<0.f)hidden[i]*=.1f;
#else
            hidden[i]=tanhf(hidden[i]);
#endif
        }
        frontend_linear(W_EMBEDDING_PREDICTOR_2_WEIGHT,W_EMBEDDING_PREDICTOR_2_BIAS,hidden,residual,
                        EMBEDDING_DIM,PREDICTOR_HIDDEN);
        float factor=fmaxf(0.f,fminf(2.f,interval/.05f));
        for(int i=0;i<EMBEDDING_DIM;i++){
#if MICRO_FRONTEND
            float bounded=fmaxf(-1.f,fminf(1.f,residual[i]));
#else
            float bounded=tanhf(residual[i]);
#endif
            predicted[i]=anchor_embedding[i]+.25f*bounded*factor;
        }
    }
    for(int i=0;i<EMBEDDING_DIM;i++)innovation[i]=encoded[i]-predicted[i];
    if(!ready)memcpy(older_embedding,encoded,sizeof(older_embedding));
    else memcpy(older_embedding,anchor_embedding,sizeof(older_embedding));
    memcpy(anchor_embedding,encoded,sizeof(anchor_embedding));prior_interval=interval;
}
