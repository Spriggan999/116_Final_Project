#pragma once
#include <cmath>
inline float bounded(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
#if HYBRID_MODE > 0
// Weight arrays are emitted by train.py. No framework or host inference is used.
inline void neural_mlp(const float* input, float* output) {
    float h[NET_H];
    for (int j=0;j<NET_H;j++) {
        float s=net_b1[j];
        for(int i=0;i<NET_IN;i++) s += net_w1[j*NET_IN+i] * ((input[i]-net_mean[i])*net_invstd[i]);
        h[j]=s>0?s:0;
    }
    for(int j=0;j<NET_OUT;j++) {
        float s=net_b2[j];
        for(int i=0;i<NET_H;i++) s+=net_w2[j*NET_H+i]*h[i];
        output[j]=s;
    }
}
#if HYBRID_MODE == 2
// 2x8x8 -> Conv(2,4,3,stride=2) -> ReLU -> 36 -> 8 -> 2.
inline void neural_track(const float* input, float* output) {
    float feature[36];
    for(int c=0;c<4;c++) for(int y=0;y<3;y++) for(int x=0;x<3;x++) {
        float s=conv_b[c];
        for(int k=0;k<2;k++) for(int v=0;v<3;v++) for(int u=0;u<3;u++)
            s+=conv_w[((c*2+k)*3+v)*3+u]*input[k*64+(2*y+v)*8+2*x+u];
        feature[c*9+y*3+x]=s>0?s:0;
    }
    neural_mlp(feature,output);
}
#endif
#endif
