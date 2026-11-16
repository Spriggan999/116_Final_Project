"""Lite-specific neural hooks: smaller networks, 11-patch budgets, isolated weights."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HERE=ROOT/'lite'
NAMES=['patch_quality','patch_tracker','measurement_confidence','compute_controller','translation_scale']

def generate():
    s=(HERE/'base/solution.cpp').read_text()
    def edit(old,new):
        nonlocal s
        if s.count(old)!=1: raise RuntimeError(f'Hook anchor changed: {old[:70]}')
        s=s.replace(old,new)
    edit('namespace {','''#include "network.h"
#ifndef HYBRID_COLLECT
#define HYBRID_COLLECT 0
#endif
namespace {
int hybrid_budget=11, hybrid_iterations=3;
float hybrid_previous[8]={0};
double t_clone=0;
#if HYBRID_COLLECT
float patch_records[11][138]; int patch_record_count=0;
double window_record[18]; bool window_ready=false;
#endif
''')
    edit('void kf_clone() {\n    p_kf = p_w;', 'void kf_clone() {\n    t_clone=last_t;\n    p_kf = p_w;')
    edit('    int sub;                        // template subset (cut / tracked on frame offset sub)', '''    float quality_features[6];
    int sub;                        // template subset (cut / tracked on frame offset sub)''')
    edit('// cut templates for subset `si` from the current frame at the next ranked cells, until `want` more are accepted', '''inline void quality_features(int x,int y,float score,float* f) {
    f[0]=std::log1p(std::fmax(0.0f,score))/8.0f; f[1]=(float)x/SW; f[2]=(float)y/SH;
    f[3]=(float)norm(omega_b); f[4]=(float)norm(v_w)/5.0f; f[5]=(float)((samp_n?samp_s/samp_n:CONTRAST_REF)/CONTRAST_REF);
}
// cut templates for subset `si` from the current frame at the next ranked cells, until `want` more are accepted''')
    edit('        p.sub = si;', '''        p.sub = si;
#if HYBRID_COLLECT
        quality_features(cx_[top[i]],cy_[top[i]],cscore[top[i]],p.quality_features);
#endif''')
    edit('bool track(const Patch& p, double& px, double& py) {','bool track_classic(const Patch& p, double& px, double& py) {')
    edit('for (int it = 0; it < LK_ITERS; it++) {', '''for (int it = 0; it <
#if HYBRID_MODE == 2
         1
#elif HYBRID_MODE == 4
         hybrid_iterations
#else
         LK_ITERS
#endif
         ; it++) {''')
    edit('// ================= small solvers =================', '''// Pair input uses separate mean removal, matching training exactly.
#if HYBRID_MODE == 2 || HYBRID_COLLECT
bool patch_pair(const Patch& p,double px,double py,float* pair) {
    int ix=(int)std::floor(px),iy=(int)std::floor(py);
    if(ix<0||iy<0||ix+PS+1>SW||iy+PS+1>SH) return false;
    float ax=px-ix,ay=py-iy; float a[PS+2],b[PS+2];
    float means[2]={0,0};
    for(int v=0;v<PS;v++) {
        fetch_row(ix,iy+v,PS+1,a);fetch_row(ix,iy+v+1,PS+1,b);
        for(int u=0;u<PS;u++) {
            int k=v*PS+u;
            pair[k]=p.T[k];
            pair[64+k]=(1-ay)*((1-ax)*a[u]+ax*a[u+1])+ay*((1-ax)*b[u]+ax*b[u+1]);
            means[0]+=pair[k]/64;means[1]+=pair[64+k]/64;
        }
    }
    for(int c=0;c<2;c++) for(int k=0;k<64;k++) pair[c*64+k]=bounded((pair[c*64+k]-means[c])/64.0f,-2,2);
    return true;
}
#endif
bool track(const Patch& p,double& px,double& py) {
    double x0=px,y0=py;
#if HYBRID_MODE == 2 || HYBRID_COLLECT
    float pair[128];bool valid_pair=patch_pair(p,px,py,pair);
#endif
#if HYBRID_MODE == 2
    if(valid_pair) {
        float shift[2];neural_track(pair,shift);
        px+=bounded(shift[0],-1,1)*1.5f;py+=bounded(shift[1],-1,1)*1.5f;
    }
#endif
    bool ok=track_classic(p,px,py);
#if HYBRID_MODE == 2
    if((px-x0)*(px-x0)+(py-y0)*(py-y0)>MAX_CORR_PX*MAX_CORR_PX) ok=false;
#endif
#if HYBRID_COLLECT
    if(valid_pair && patch_record_count<11) {
        float* row=patch_records[patch_record_count++];
        for(int j=0;j<6;j++)row[j]=p.quality_features[j];
        row[6]=ok?1:0;row[7]=(px-x0)/1.5;row[8]=(py-y0)/1.5;row[9]=p.sub;
        for(int j=0;j<128;j++)row[10+j]=pair[j];
    }
#endif
    return ok;
}
// ================= small solvers =================''')
    # Shared window features are evaluated before the two independent measurement hooks.
    anchor='    Vec3 a = std::fabs(u.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};'
    edit(anchor, '''    float features[8]={(float)(last_parallax/fx_s*100), (float)(std::sqrt(q)*3),
        (float)nt/N_TRACK,(float)kept/N_TRACK,(float)dn,(float)(norm(v_w)/5),
        (float)norm(omega_b),(float)((last_t-t_clone)/0.45)};
    for(int j=0;j<8;j++) hybrid_previous[j]=features[j];
    float dir_noise_factor=1,disp_noise_factor=1,scale_factor=1;
#if HYBRID_MODE == 3
    float noise[2];neural_mlp(features,noise);
    dir_noise_factor=std::exp(bounded(noise[0],-0.693147f,1.386294f));
    disp_noise_factor=std::exp(bounded(noise[1],-0.693147f,1.386294f));
#elif HYBRID_MODE == 5
    float correction[1];neural_mlp(features,correction);
    scale_factor=std::exp(bounded(correction[0],-0.287682f,0.405465f));
#endif
#if HYBRID_COLLECT
    for(int j=0;j<8;j++)window_record[j]=features[j];
    Vec3 u0=mulT(R_wc0,u);
    window_record[8]=u0.x;window_record[9]=u0.y;window_record[10]=u0.z;
    window_record[11]=last_parallax/fx_s*PAR_SPEED;
    window_record[12]=t_clone;window_record[13]=last_t;
    double ss=std::sqrt(q/kept)*SIG_DIR_SCALE+SIG_DIR_PX/(last_parallax*std::sqrt((double)kept))+SIG_DIR_FLOOR;
    window_record[14]=ss;window_record[15]=PAR_SPEED_REL*window_record[11]+0.01;
    window_record[16]=nt;window_record[17]=kept;window_ready=true;
#endif
''' + anchor)
    edit('    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR);', '    double Rn = sig_dir * sig_dir * (dn * dn + D_FLOOR * D_FLOOR)*dir_noise_factor;')
    edit('    double sd = PAR_SPEED_REL * dmeas + 0.01;', '    dmeas *= scale_factor;\n    double sd = (PAR_SPEED_REL * dmeas + 0.01)*std::sqrt(disp_noise_factor);')
    # Selector replaces top-K ranking scores, preserving raw texture thresholds.
    edit('        float v = score_cell(cx_[k], cy_[k]);', '        float v = score_cell(cx_[k], cy_[k]); cscore[k]=v;')
    edit('    select_top();\n    {   // texture threshold', '''    select_top();
#if HYBRID_MODE == 1
    // Score only the 15 texture-shortlisted cells, not all 96 candidates.
    for(int k=0;k<nk;k++) {
        int index=top[k];float f[6],out[1];
        quality_features(cx_[index],cy_[index],cscore[index],f);neural_mlp(f,out);
        topv[k]=cscore[index]*bounded(0.5f+out[0]*0.1f,0.1f,1.0f);
    }
    for(int k=1;k<nk;k++) {
        int index=top[k];float value=topv[k];int j=k;
        while(j>0 && topv[j-1]<value) {top[j]=top[j-1];topv[j]=topv[j-1];j--;}
        top[j]=index;topv[j]=value;
    }
#endif
    {   // texture threshold''')
    edit('    rank_slice(0, NCAND);', '''#if HYBRID_MODE == 4
    float action[3];neural_mlp(hybrid_previous,action);
    int best=0;for(int j=1;j<3;j++)if(action[j]>action[best])best=j;
    hybrid_budget=best==0?7:(best==1?9:11);hybrid_iterations=best+1;
    // Insufficient tracks or fast rotation always triggers the full budget.
    if(hybrid_previous[2]<0.4f || norm(omega_b)>0.6) { hybrid_budget=11;hybrid_iterations=3; }
#endif
    rank_slice(0, NCAND);''')
    edit('        int want = (N_TRACK - n_patches + (SPLIT - si) - 1) / (SPLIT - si);', '''        int target_patches=N_TRACK;
#if HYBRID_MODE == 4
        target_patches=hybrid_budget;
#endif
        int want = (target_patches - n_patches + (SPLIT - si) - 1) / (SPLIT - si);''')
    edit('    frame_px = frame;', '''    frame_px = frame;
#if HYBRID_COLLECT
    patch_record_count=0;window_ready=false;
#endif''')
    edit('    initialized = false; have_kf = false;', '''    initialized = false; have_kf = false;
    hybrid_budget=11;hybrid_iterations=3;
#if HYBRID_COLLECT
    // The host collector reuses a library; each WASM sandbox starts fresh.
    acc_T=0;acc_n=0;acc_R=Mat3{{0}};acc_f={0,0,0};
    prop_debt=0;prop_ahead_next=false;inject_pending=false;
#endif
    for(int j=0;j<8;j++)hybrid_previous[j]=0;''')
    # Collection exports exist only in host builds, never benchmark modules.
    s+='''\n#if HYBRID_COLLECT
extern "C" int hybrid_patches(float* out) {
    std::memcpy(out,patch_records,patch_record_count*138*sizeof(float));return patch_record_count;
}
extern "C" int hybrid_window(double* out) {
    if(!window_ready)return 0;std::memcpy(out,window_record,sizeof window_record);return 1;
}
#endif
'''
    (HERE/'common').mkdir(parents=True,exist_ok=True)
    (HERE/'common/estimator.inc').write_text(s)
    for mode,name in enumerate(NAMES,1):
        (HERE/'models'/name).mkdir(parents=True,exist_ok=True)
        (HERE/'models'/name/'solution.cpp').write_text(f'#define HYBRID_MODE {mode}\n#include "weights.h"\n#include "../../common/estimator.inc"\n')
    (HERE/'common/collect.cpp').write_text('#define HYBRID_MODE 0\n#define HYBRID_COLLECT 1\n#include "estimator.inc"\n')

def fit_lite(*args,**kwargs):
    from train import fit
    kwargs.update(hidden_size=4,tracker_channels=2,tracker_shift_scale=1.5,
                  variance_bounds=(.5,4.),scale_bounds=(.75,1.5))
    return fit(*args,**kwargs)


def make_tasks_lite(training,validation):
    from train_fresh import make_tasks
    tasks=make_tasks(training,validation)
    x,y,scene,kind=tasks['patch_tracker']
    # Predict a conservative initial shift; the following LK step refines it.
    tasks['patch_tracker']=(x,y.clip(-1,1),scene,kind)
    x,y,scene,kind=tasks['translation_scale']
    import numpy as np
    tasks['translation_scale']=(x,y.clip(np.log(.75),np.log(1.5)),scene,kind)
    return tasks

if __name__=='__main__':generate()
