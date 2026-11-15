/* Tiny tiers: learned scalar/2D-CNN encoder, causal forecast and affine head.
 * Fixed attitude and 20 Hz integration; only the last accelerometer is used.
 * Camera calibration is folded into the state-head weights at init.
 * Keep the step kernel small and meter every compiled export before publishing.
 */
#include "viobench_abi.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "student_weights.h"
#include "native_vision.h"
typedef float vec __attribute__((vector_size(16)));
static vec p,v,m0,m1,m2,offset;
#define VISUAL_FEATURES (2*EMBEDDING_DIM)
static vec visual[VISUAL_FEATURES];
static int ready;

int32_t abi_version(void){return VIOBENCH_ABI_VERSION;}
void *alloc(uint32_t n){return malloc(n?n:1);}
int32_t frame_width(void){return 320;}
int32_t init(const char *json,uint32_t length){
    if(length>=16384)return -1;
    char text[16384];memcpy(text,json,length);text[length]=0;
    const char *s=strstr(text,"\"T_cam_imu\"");if(!s)return -1;
    s=strchr(s,':');if(!s)return -1;s++;
    float R[16];
    for(int i=0;i<16;i++){
        while(*s&&(*s=='['||*s==']'||*s==','||*s==' '||*s=='\n'||*s=='\r'||*s=='\t'))s++;
        char *end;double value=strtod(s,&end);if(end==s||!isfinite(value))return -1;R[i]=(float)value;s=end;
    }
    const float *w=W_GAIN;
    m0=(vec){R[0]*w[0],R[4]*w[0],R[8]*w[0],0};
    m1=(vec){R[1]*w[1],R[5]*w[1],R[9]*w[1],0};
    m2=(vec){R[2]*w[2],R[6]*w[2],R[10]*w[2],0};
    offset=(vec){R[0]*w[3]+R[1]*w[4]+R[2]*w[5],
                 R[4]*w[3]+R[5]*w[4]+R[6]*w[5],
                 R[8]*w[3]+R[9]*w[4]+R[10]*w[5],0};
    for(int j=0;j<VISUAL_FEATURES;j++){
        float x=W_VISION_WEIGHT[j],y=W_VISION_WEIGHT[VISUAL_FEATURES+j],z=W_VISION_WEIGHT[2*VISUAL_FEATURES+j];
        visual[j]=(vec){R[0]*x+R[1]*y+R[2]*z,
                        R[4]*x+R[5]*y+R[6]*z,
                        R[8]*x+R[9]*y+R[10]*z,0};
    }
    ready=0;p=v=(vec){0,0,0,0};frontend_reset();return 0;
}
int32_t step(double t,const uint8_t *frame,const double *imu,int32_t count,double *__restrict out){
    float gray[GRID_SIZE];int index=0;
    for(int y=0;y<GRID_HEIGHT;y++)for(int x=0;x<GRID_WIDTH;x++){
        const uint8_t *pixel=frame+3*(((2*y+1)*240/(2*GRID_HEIGHT))*320+(2*x+1)*320/(2*GRID_WIDTH));
        gray[index++]=(pixel[0]+pixel[1]+pixel[2])*(1.f/765.f);
    }
    if(ready&&count!=10)return -1;
    const double *a=ready?imu+64:NULL;
    float sensor[3]={ready?(float)a[0]:0.f,ready?(float)a[1]:0.f,ready?(float)a[2]:0.f};
    float encoded[EMBEDDING_DIM],innovation[EMBEDDING_DIM];
    frontend_pair(gray,sensor,ready?.05f:0.f,ready,encoded,innovation);
    if(ready){
        vec acc=m0*(float)a[0]+m1*(float)a[1]+m2*(float)a[2]+offset;
        for(int i=0;i<EMBEDDING_DIM;i++)acc+=visual[i]*encoded[i]+visual[EMBEDDING_DIM+i]*innovation[i];
        v+=acc*.05f;p+=v*.05f;
    }
    ready=1;
    out[0]=p[0];out[1]=p[1];out[2]=p[2];out[3]=out[4]=out[5]=0;out[6]=1;
    out[7]=v[0];out[8]=v[1];out[9]=v[2];out[10]=out[11]=out[12]=0;
    return 0;
}
