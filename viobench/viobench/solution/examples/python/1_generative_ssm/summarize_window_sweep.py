"""Verify completed window coverage, export rates and draw an LR heatmap."""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from lr_sweep import write_json
from lr_sweep_windows import ORDER,WINDOWS,horizon,common_seed_ranking


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    results=json.loads((root/'window_results.json').read_text())
    records=json.loads((root/'all_trials.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    teacher_embedding=json.loads((Path(protocol['parent'])/'recommendations.json').read_text())['teacher_embedding']['lr']
    for seconds in WINDOWS:
        group=results[str(seconds)]
        assert set(group['state'])==set(ORDER),f'Incomplete state coverage at {seconds}s'
        assert set(group['embedding'])==set(ORDER[:-1]),f'Incomplete embedding coverage at {seconds}s'
        for name in ORDER:
            subset=[row for row in records if row['model']==name and horizon(row)==seconds]
            state_rows=[row for row in subset if 'embedding_lr' not in row]
            rank,_=common_seed_ranking(state_rows)
            assert rank[0]==group['state'][name]['lr']
            if name!='teacher':
                embedding=group['embedding'][name]
                embedding_rows=[row for row in subset if 'embedding_lr' in row and row['lr']==embedding['state_lr']]
                for metric,field in [('score','embedding_lr'),('embedding_score','embedding_loss_winner_lr')]:
                    rank,_=common_seed_ranking(embedding_rows,'embedding_lr',metric)
                    assert rank[0]==embedding[field]
    hashes=collections.defaultdict(set)
    for row in records:hashes[row['model'],row['seed']].add(row['initial_sha256'])
    assert all(len(values)==1 for values in hashes.values()),'Mismatched initial weights'
    verification=dict(state_model_windows=90,student_embedding_model_windows=80,
        complete=True,matched_initial_weights=True,state_trials=sum('embedding_lr' not in row for row in records),
        embedding_group_trials=sum('embedding_lr' in row for row in records),
        failed_trials=sum(row['status']!='complete' for row in records),
        teacher_embedding='one reused completed frame-pair study; no window parameter')
    trace=root/'gpu_samples.csv'
    if trace.exists():
        with trace.open() as stream:samples=list(csv.DictReader(stream))
        if samples:
            used=[float(row['memory_used_mib']) for row in samples]
            util=[float(row['gpu_utilization_pct']) for row in samples]
            verification['gpu_samples']=dict(count=len(samples),peak_used_mib=max(used),
                median_utilization_pct=float(np.median(util)),mean_utilization_pct=float(np.mean(util)))
    write_json(root/'coverage_verification.json',verification)
    fields=['model','window_seconds','state_batch','state_lr','embedding_lr_ranked_by_state',
            'embedding_lr_ranked_by_jepa','teacher_global_embedding_lr',
            'embedding_state_winner_at_edge','embedding_jepa_winner_at_edge']
    with (root/'window_rates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name in ORDER:
            for seconds in WINDOWS:
                group=results[str(seconds)];embedding=group['embedding'].get(name,{})
                state_lr=group['state'][name]['lr']
                trial=next(row for row in records if row['model']==name and horizon(row)==seconds
                           and row['seed']==29 and row['lr']==state_lr and 'embedding_lr' not in row)
                writer.writerow(dict(model=name,window_seconds=seconds,state_batch=trial['batch'],state_lr=state_lr,
                    embedding_lr_ranked_by_state=embedding.get('embedding_lr'),
                    embedding_lr_ranked_by_jepa=embedding.get('embedding_loss_winner_lr'),
                    teacher_global_embedding_lr=teacher_embedding if name=='teacher' else None,
                    embedding_state_winner_at_edge=embedding.get('state_winner_at_tested_edge'),
                    embedding_jepa_winner_at_edge=embedding.get('jepa_winner_at_tested_edge')))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print('Coverage verified and CSV saved. Install matplotlib for the optional heatmap.')
        return
    fig,axes=plt.subplots(3,1,figsize=(13,14),layout='constrained')
    specifications=[('state','lr',ORDER,'State LR (shared during initial state sweep)'),
        ('embedding','embedding_lr',ORDER[:-1],'Embedding-group LR ranked by state loss'),
        ('embedding','embedding_loss_winner_lr',ORDER[:-1],'Embedding-group LR ranked by JEPA loss')]
    all_values=[row[key] for _,key,_,_ in specifications for group in results.values()
                for row in group['state' if key=='lr' else 'embedding'].values()]
    lower=min(all_values);upper=max(all_values)
    if lower==upper:upper=lower*10
    for ax,(kind,field,names,title) in zip(axes,specifications):
        data=np.array([[results[str(seconds)][kind][name][field] for seconds in WINDOWS] for name in names])
        im=ax.imshow(data,aspect='auto',cmap='viridis',norm=LogNorm(vmin=lower,vmax=upper))
        ax.set_xticks(range(len(WINDOWS)),WINDOWS);ax.set_yticks(range(len(names)),names)
        ax.set_xlabel('Training window (seconds)');ax.set_title(title)
        for y in range(len(names)):
            for x in range(len(WINDOWS)):
                value=data[y,x];color='black' if im.norm(value)>.6 else 'white'
                ax.text(x,y,f'{value:.0e}',ha='center',va='center',fontsize=8,color=color)
    fig.colorbar(im,ax=axes,label='Learning rate (log scale)',shrink=.8)
    fig.suptitle('Fresh-checkpoint LR pilots — provisional candidates\n'
                 f'Teacher embedding: {teacher_embedding:g}, global frame-pair sweep, not window-dependent',fontsize=13)
    fig.savefig(root/'window_lr_heatmap.png',dpi=160);fig.savefig(root/'window_lr_heatmap.svg')
    plt.close(fig)
    print(json.dumps(verification));print(root/'window_lr_heatmap.png')


if __name__=='__main__':main()
