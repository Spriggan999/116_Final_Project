"""Summarize downloaded read-only W&B exports and matched local training logs."""
import argparse
import json
from pathlib import Path
import re
import statistics


def stats(values):
    values=sorted(values)
    return dict(samples=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p95=values[int(.95*(len(values)-1))],maximum=values[-1])


def analyze(folder,run):
    performance=[json.loads(line) for line in (run/'performance.jsonl').read_text().splitlines()]
    summary={}
    for filename,stage in (('c3ikypyw.json','embedding'),('yxkxjez9.json','state')):
        data=json.loads((folder/filename).read_text())
        system=data['system'];history=data['history']
        hardware={}
        for name in ('gpu.0.gpu','gpu.0.smActive','gpu.0.pipeTensorActive','gpu.0.memoryAllocated','gpu.0.powerWatts','cpu'):
            values=[row['system.'+name] for row in system if isinstance(row.get('system.'+name),(float,int))]
            if values:hardware[name]=stats(values)
        training=[row for row in performance if row['stage']==stage]
        wall=[b['_timestamp']-a['_timestamp'] for a,b in zip(history,history[1:])]
        summary[stage]=dict(name=data['name'],url=data['url'],epochs=len(history),system=hardware,
            training_seconds=stats([row['seconds'] for row in training]),epoch_wall_seconds=stats(wall),
            fresh_data_wait_seconds=sum(row['fresh_data_wait_seconds'] for row in training),
            pytorch_peak_allocated_mib=max(row['peak_allocated_mib'] for row in training))
    log=(run/'train.log').read_text()
    times=[float(seconds) for count,seconds in re.findall(r'DONE (\d+) episodes in ([\d.]+)s',log) if count=='8']
    summary['collection_seconds']=stats(times)
    curriculum=[json.loads(line) for line in (run/'curriculum_metrics.jsonl').read_text().splitlines()]
    summary['state_windows']=sorted({row['window_samples'] for row in curriculum if row['phase']=='state'})
    summary['state_promotions']=sum(bool(row['promoted'] or row['window_promoted']) for row in curriculum if row['phase']=='state')
    summary['launcher_status']=json.loads((run/'run_status.json').read_text())
    (folder/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for column,(filename,label) in enumerate((('c3ikypyw.json','Embedding SL'),('yxkxjez9.json','State SL'))):
        data=json.loads((folder/filename).read_text());system=data['system'];start=system[0]['_timestamp']
        for key,name in (('system.gpu.0.gpu','GPU busy'),('system.cpu','CPU utilization')):
            rows=[row for row in system if key in row]
            axes[0,column].plot([(row['_timestamp']-start)/60 for row in rows],[row[key] for row in rows],label=name,linewidth=1)
        axes[0,column].set(title=label,xlabel='Minutes within W&B run',ylabel='Utilization (%)',ylim=(0,100))
        axes[0,column].legend()
        prefix='embedding' if column==0 else 'student'
        history=data['history']
        loss_key=prefix+('/validation_loss' if column==0 else '/val_loss')
        axes[1,column].plot([row[prefix+'/epoch'] for row in history],
                            [row[loss_key] for row in history],marker='o')
        axes[1,column].set(xlabel='Epoch',ylabel='Fixed validation loss')
        if column==0:axes[1,column].set_yscale('log')
        else:axes[1,column].set_title('Trained on 0.5 s; validated on 20 s; no promotions')
    figure.suptitle('2026-11-13 W&B analysis — sampled utilization, not a CUDA kernel profile')
    figure.savefig(folder/'gpu_and_learning.png',dpi=150)
    plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--exports',type=Path,default=root/'verification/wandb_curriculum_20261113')
    parser.add_argument('--run',type=Path,default=root/'checkpoints/20261113-011547-curriculum-23bnej8g')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();analyze(args.exports,args.run)
    if args.plot:plot(args.exports)
