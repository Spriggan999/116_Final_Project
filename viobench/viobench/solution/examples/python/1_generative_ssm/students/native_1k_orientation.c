/* Camera-rate CNN + learned acceleration/gyro heads and normalized attitude.
 * Camera calibration is folded into heads; lever arm is applied to outputs. */
#include "viobench_abi.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "student_weights.h"
#include "native_vision.h"
typedef float vec __attribute__((vector_size(16)));
static vec p,v,q,m0,m1,m2,offset,gravity,lever,g0,g1,g2,gb;
static vec cross(vec a,vec b){
    return (vec){a[1],a[2],a[0],0}*(vec){b[2],b[0],b[1],0}
          -(vec){a[2],a[0],a[1],0}*(vec){b[1],b[2],b[0],0};
}
static vec rotate(vec attitude,vec value){
    vec xyz=attitude;xyz[3]=0;
    vec twice=2.f*cross(xyz,value);
    return value+attitude[3]*twice+cross(xyz,twice);
}
#define VISUAL_FEATURES (2*EMBEDDING_DIM)
static vec visual[VISUAL_FEATURES],gyro_visual[VISUAL_FEATURES];
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
    gravity=(vec){-9.81f*R[2],-9.81f*R[6],-9.81f*R[10],0};
    offset-=gravity;
    lever=(vec){-R[3],-R[7],-R[11],0};
    const float *gg=W_GYRO_GAIN,*bg=W_GYRO_BIAS;
    g0=(vec){R[0]*gg[0],R[4]*gg[0],R[8]*gg[0],0};
    g1=(vec){R[1]*gg[1],R[5]*gg[1],R[9]*gg[1],0};
    g2=(vec){R[2]*gg[2],R[6]*gg[2],R[10]*gg[2],0};
    gb=(vec){R[0]*bg[0]+R[1]*bg[1]+R[2]*bg[2],
             R[4]*bg[0]+R[5]*bg[1]+R[6]*bg[2],
             R[8]*bg[0]+R[9]*bg[1]+R[10]*bg[2],0};
    for(int j=0;j<VISUAL_FEATURES;j++){
        float x=W_VISION_WEIGHT[j],y=W_VISION_WEIGHT[VISUAL_FEATURES+j],z=W_VISION_WEIGHT[2*VISUAL_FEATURES+j];
        visual[j]=(vec){R[0]*x+R[1]*y+R[2]*z,
                        R[4]*x+R[5]*y+R[6]*z,
                        R[8]*x+R[9]*y+R[10]*z,0};
        x=W_GYRO_VISION_WEIGHT[j];y=W_GYRO_VISION_WEIGHT[VISUAL_FEATURES+j];z=W_GYRO_VISION_WEIGHT[2*VISUAL_FEATURES+j];
        gyro_visual[j]=(vec){R[0]*x+R[1]*y+R[2]*z,R[4]*x+R[5]*y+R[6]*z,R[8]*x+R[9]*y+R[10]*z,0};
    }
    ready=0;p=v=(vec){0,0,0,0};q=(vec){0,0,0,1};frontend_reset();return 0;
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
    vec omega=(vec){0,0,0,0};
    if(ready){
        vec acc=m0*(float)a[0]+m1*(float)a[1]+m2*(float)a[2]+offset;
        for(int i=0;i<EMBEDDING_DIM;i++)acc+=visual[i]*encoded[i]+visual[EMBEDDING_DIM+i]*innovation[i];
        omega=g0*(float)a[3]+g1*(float)a[4]+g2*(float)a[5]+gb;
        for(int i=0;i<EMBEDDING_DIM;i++)omega+=gyro_visual[i]*encoded[i]+gyro_visual[EMBEDDING_DIM+i]*innovation[i];
        acc=rotate(q,acc)+gravity;
        v+=acc*.05f;p+=v*.05f;
        vec delta=omega*.025f,xyz=q;xyz[3]=0;
        vec next=xyz+q[3]*delta+cross(xyz,delta);
        next[3]=q[3]-(xyz[0]*delta[0]+xyz[1]*delta[1]+xyz[2]*delta[2]);
        vec sq=next*next;
        q=next/sqrtf(sq[0]+sq[1]+sq[2]+sq[3]);
    }
    ready=1;
    vec position=p+rotate(q,lever)-lever;
    vec inverse=-q;inverse[3]=q[3];
    vec velocity=rotate(inverse,v)+cross(omega,lever);
    out[0]=position[0];out[1]=position[1];out[2]=position[2];
    out[3]=q[0];out[4]=q[1];out[5]=q[2];out[6]=q[3];
    out[7]=velocity[0];out[8]=velocity[1];out[9]=velocity[2];
    out[10]=omega[0];out[11]=omega[1];out[12]=omega[2];
    return 0;
}
