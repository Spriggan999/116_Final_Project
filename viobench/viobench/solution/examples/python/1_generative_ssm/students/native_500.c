/* 500-fuel tier: two sparse grayscale pixels + last accelerometer sample.
 * Eighteen learned coefficients; fixed attitude and 20 Hz integration.
 * Calibration and grayscale normalization are folded into weights at init.
 * Keep the step kernel small and meter every compiled export before publishing.
 */
#include "viobench_abi.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "student_weights.h"
typedef float vec __attribute__((vector_size(16)));
static vec p,v,m0,m1,m2,offset;
static vec visual[4];
static float previous_left,previous_right;
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
    const float *w=WEIGHTS;
    m0=(vec){R[0]*w[0],R[4]*w[0],R[8]*w[0],0};
    m1=(vec){R[1]*w[1],R[5]*w[1],R[9]*w[1],0};
    m2=(vec){R[2]*w[2],R[6]*w[2],R[10]*w[2],0};
    offset=(vec){R[0]*w[3]+R[1]*w[4]+R[2]*w[5],
                 R[4]*w[3]+R[5]*w[4]+R[6]*w[5],
                 R[8]*w[3]+R[9]*w[4]+R[10]*w[5],0};
    for(int j=0;j<4;j++){
        float x=w[6+j],y=w[10+j],z=w[14+j];
        visual[j]=(vec){(R[0]*x+R[1]*y+R[2]*z)/765.f,
                        (R[4]*x+R[5]*y+R[6]*z)/765.f,
                        (R[8]*x+R[9]*y+R[10]*z)/765.f,0};
    }
    ready=0;previous_left=previous_right=0;p=v=(vec){0,0,0,0};return 0;
}
int32_t step(double t,const uint8_t *frame,const double *imu,int32_t count,double *__restrict out){
    const uint8_t *left=frame+3*(120*320+80),*right=frame+3*(120*320+240);
    float l=(float)(left[0]+left[1]+left[2]),r=(float)(right[0]+right[1]+right[2]);
    if(ready){
        if(count!=10)return -1;
        const double *a=imu+64;
        vec acc=m0*(float)a[0]+m1*(float)a[1]+m2*(float)a[2]+offset;
        acc+=visual[0]*l+visual[1]*r+visual[2]*(l-previous_left)+visual[3]*(r-previous_right);
        v+=acc*.05f;p+=v*.05f;
    }
    previous_left=l;previous_right=r;ready=1;
    out[0]=p[0];out[1]=p[1];out[2]=p[2];out[3]=out[4]=out[5]=0;out[6]=1;
    out[7]=v[0];out[8]=v[1];out[9]=v[2];out[10]=out[11]=out[12]=0;
    return 0;
}
