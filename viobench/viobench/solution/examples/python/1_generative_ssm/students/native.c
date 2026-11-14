/* Frame-rate student. No Python, OpenCV, filesystem reads or dynamic inference allocations. */
#include "viobench_abi.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "student_weights.h"

static double position[3],velocity[3],rotation[4],gravity[3],omega[3];
static double Rci[9],lever[3],qci[4],last_t;
static float previous[GRID_SIZE],hidden[HIDDEN+1];
static int ready,gravity_ready,width,height;

static void cross(const double *a,const double *b,double *r){
    r[0]=a[1]*b[2]-a[2]*b[1];r[1]=a[2]*b[0]-a[0]*b[2];r[2]=a[0]*b[1]-a[1]*b[0];
}
static void multiply(const double *a,const double *b,double *r){
    double c[3];cross(a,b,c);
    for(int i=0;i<3;i++)r[i]=a[3]*b[i]+b[3]*a[i]+c[i];
    r[3]=a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2];
}
static void rotate(const double *q,const double *v,double *r){
    double t[3],s[3];cross(q,v,t);for(int i=0;i<3;i++)t[i]*=2;
    cross(q,t,s);for(int i=0;i<3;i++)r[i]=v[i]+q[3]*t[i]+s[i];
}
static void matvec(const double *m,const double *v,double *r){
    for(int i=0;i<3;i++)r[i]=m[3*i]*v[0]+m[3*i+1]*v[1]+m[3*i+2]*v[2];
}
static void normalize(double *q){
    double n=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for(int i=0;i<4;i++)q[i]/=n;
}
static void from_matrix(const double *m,double *q){
    double trace=m[0]+m[4]+m[8];
    if(trace>0){
        double s=2*sqrt(1+trace);q[0]=(m[7]-m[5])/s;q[1]=(m[2]-m[6])/s;q[2]=(m[3]-m[1])/s;q[3]=s/4;
    }else{
        int i=m[4]>m[0]?1:0;if(m[8]>m[4*i])i=2;int j=(i+1)%3,k=(i+2)%3;
        double s=2*sqrt(fmax(1e-12,1+m[4*i]-m[4*j]-m[4*k]));
        q[i]=s/4;q[j]=(m[3*i+j]+m[3*j+i])/s;q[k]=(m[3*i+k]+m[3*k+i])/s;
        q[3]=(m[3*k+j]-m[3*j+k])/s;
    }normalize(q);
}
static const char *field(const char *json,const char *name){
    const char *p=strstr(json,name);if(!p)return NULL;p=strchr(p,':');return p?p+1:NULL;
}
int32_t abi_version(void){return VIOBENCH_ABI_VERSION;}
void *alloc(uint32_t n){return malloc(n?n:1);}
int32_t frame_width(void){return 320;}
int32_t init(const char *json,uint32_t length){
    if(length>=16384)return -1;char text[16384];memcpy(text,json,length);text[length]=0;
    const char *a=field(text,"\"width\""),*b=field(text,"\"height\"");
    if(!a||!b)return -1;width=atoi(a);height=atoi(b);if(width!=320||height!=240)return -1;
    const char *p=field(text,"\"T_cam_imu\"");if(!p)return -1;
    double extrinsic[16];
    for(int i=0;i<16;i++){
        while(*p&&(*p=='['||*p==']'||*p==','||*p==' '||*p=='\n'||*p=='\r'||*p=='\t'))p++;
        char *end;extrinsic[i]=strtod(p,&end);if(p==end||!isfinite(extrinsic[i]))return -1;p=end;
    }
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)Rci[3*i+j]=extrinsic[4*i+j];
    for(int j=0;j<3;j++)lever[j]=-(Rci[j]*extrinsic[3]+Rci[3+j]*extrinsic[7]+Rci[6+j]*extrinsic[11]);
    from_matrix(Rci,qci);
    memset(position,0,sizeof(position));memset(velocity,0,sizeof(velocity));
    memset(rotation,0,sizeof(rotation));rotation[3]=1;
    memset(gravity,0,sizeof(gravity));memset(omega,0,sizeof(omega));
    memset(hidden,0,sizeof(hidden));memset(previous,0,sizeof(previous));ready=gravity_ready=0;last_t=0;
    return 0;
}

static void prediction(const float *features,float *correction){
    const float *w=WEIGHTS;
#if HIDDEN > 0
    float next[HIDDEN];
    const float *input=w,*bias=input+HIDDEN*FEATURES,*recurrent=bias+HIDDEN;
    for(int i=0;i<HIDDEN;i++){
        float v=bias[i];
        for(int j=0;j<FEATURES;j++)v+=input[i*FEATURES+j]*features[j];
        for(int j=0;j<HIDDEN;j++)v+=recurrent[i*HIDDEN+j]*hidden[j];
        next[i]=tanhf(v);
    }
    memcpy(hidden,next,sizeof(next));w=recurrent+HIDDEN*HIDDEN;
    const float *x=hidden;const int n=HIDDEN;
#else
    const float *x=features;const int n=FEATURES;
#endif
    const float *bias_out=w+9*n;
    for(int i=0;i<9;i++){
        float v=bias_out[i];for(int j=0;j<n;j++)v+=w[i*n+j]*x[j];
        correction[i]=.25f*tanhf(v);
    }
}
static void physical(const double *imu,const float *correction,double dt){
    double acc[3],world[3],dq[4],next[4];
    for(int i=0;i<3;i++){acc[i]=imu[i]+correction[3+i];omega[i]=imu[3+i]+correction[i];dq[i]=omega[i]*dt*.5;}
    rotate(rotation,acc,world);
    for(int i=0;i<3;i++){
        double value=world[i]+gravity[i]+correction[6+i];
        position[i]+=velocity[i]*dt+.5*value*dt*dt;velocity[i]+=value*dt;
    }
    dq[3]=1;multiply(rotation,dq,next);normalize(next);memcpy(rotation,next,sizeof(next));
}
static void output(double *out){
    double offset[3],point[3],vbody[3],twist[3],crossed[3],inverse[4],tmp[4],qc[4];
    rotate(rotation,lever,offset);
    for(int i=0;i<3;i++){point[i]=position[i]+offset[i]-lever[i];inverse[i]=-rotation[i];}
    inverse[3]=rotation[3];matvec(Rci,point,out);
    multiply(qci,rotation,tmp);for(int i=0;i<3;i++)qc[i]=-qci[i];qc[3]=qci[3];
    multiply(tmp,qc,out+3);normalize(out+3);
    rotate(inverse,velocity,vbody);cross(omega,lever,crossed);
    for(int i=0;i<3;i++)twist[i]=vbody[i]+crossed[i];matvec(Rci,twist,out+7);matvec(Rci,omega,out+10);
}
int32_t step(double t,const uint8_t *frame,const double *imu,int32_t count,double *out){
    if(!isfinite(t)||count<0||count>10||(ready&&t<=last_t))return -1;
    double mean[6]={0},interval=ready?t-last_t:0,prior=last_t;
    for(int i=0;i<count;i++){
        const double *row=imu+7*i;
        if(!isfinite(row[0])||(ready&&row[0]<=prior)||row[0]>t+1e-8)return -1;
        prior=row[0];for(int j=0;j<6;j++){if(!isfinite(row[1+j]))return -1;mean[j]+=row[1+j];}
    }
    if(count)for(int j=0;j<6;j++)mean[j]/=count;
    if(!gravity_ready&&count){
        double norm=0;for(int j=0;j<3;j++){gravity[j]=fmax(-20,fmin(20,mean[j]));norm+=gravity[j]*gravity[j];}
        norm=sqrt(norm);if(norm>1e-5){for(int j=0;j<3;j++)gravity[j]*=-9.81/norm;gravity_ready=1;}
    }
    float features[FEATURES],correction[9];int n=0;
    for(int y=0;y<GRID_HEIGHT;y++)for(int x=0;x<GRID_WIDTH;x++){
        int yy=(2*y+1)*height/(2*GRID_HEIGHT),xx=(2*x+1)*width/(2*GRID_WIDTH);
        const uint8_t *rgb=frame+3*(yy*width+xx);
        float value=(rgb[0]+rgb[1]+rgb[2])*(1.f/765.f);
        features[n]=value;features[GRID_SIZE+n]=ready?value-previous[n]:0;previous[n++]=value;
    }
    for(int j=0;j<6;j++)features[2*GRID_SIZE+j]=(float)(mean[j]/10);
    features[FEATURES-1]=(float)fmin(4,interval*20);
    prediction(features,correction);
    if(ready&&count){
#if AGGREGATE
        physical(mean,correction,interval);
#else
        double previous_t=last_t;
        for(int i=0;i<count;i++){physical(imu+7*i+1,correction,imu[7*i]-previous_t);previous_t=imu[7*i];}
        if(t-previous_t>1e-8)physical(imu+7*(count-1)+1,correction,t-previous_t);
#endif
    }
    output(out);if(!ready){for(int i=0;i<13;i++)out[i]=0;out[6]=1;}
    ready=1;last_t=t;return 0;
}
