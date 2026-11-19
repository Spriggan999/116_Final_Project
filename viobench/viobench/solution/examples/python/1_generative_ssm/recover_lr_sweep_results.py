"""CPU-only recovery of the stopped LR studies. Uses Python's standard library.

No torch, CUDA, W&B SDK, subprocesses, training or data collection is imported.
Cleanup is limited to these two studies, after their numerical results are saved.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
OLD=HERE/'checkpoints/lr-sweep-20261115-200326'
NEW=HERE/'checkpoints/lr-windows-20261115-211044'
ORDER=['1k','10k','100k','1m','1m_ssm','1m_hybrid','100m','100m_hybrid','teacher']
WINDOWS=list(range(2,21,2))


def load(path):return json.loads(path.read_text())


def write(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def seconds(row):return int(re.search(r'(?:^|-)(\d+)s(?:-|$)',row['phase']).group(1))


def rank(rows,key='lr',metric='score'):
    seeds={seed:{} for seed in [17,29]}
    for row in rows:
        if row['status']!='complete' or row.get(metric) is None or not math.isfinite(row[metric]):continue
        group=seeds[row['seed']]
        if row[key] in group:raise ValueError('Duplicate candidate in matched-seed comparison')
        group[row[key]]=row[metric]
    common=set(seeds[17])&set(seeds[29])
    if not common:return None
    scores={rate:sum(seeds[seed][rate]/max(min(seeds[seed][r] for r in common),1e-12)/2
                     for seed in [17,29]) for rate in common}
    winner=min(common,key=lambda rate:(scores[rate],rate))
    return dict(lr=winner,normalized_scores=scores)


def digest(path,compressed=False):
    value=hashlib.sha256()
    opener=gzip.open if compressed else open
    with opener(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def compress_verified(path):
    if path.is_symlink() or not path.exists():return None
    before=path.stat();target=path.with_name(path.name+'.gz');temp=target.with_name(target.name+'.tmp')
    if target.exists():return None
    original_hash=digest(path)
    with path.open('rb') as source,temp.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as output:
            shutil.copyfileobj(source,output,1024*1024)
        raw.flush();os.fsync(raw.fileno())
    if digest(temp,compressed=True)!=original_hash:raise ValueError(f'Archive verification failed: {path}')
    after=temp.stat()
    if after.st_size>=before.st_size or after.st_blocks>before.st_blocks:
        temp.unlink();return None
    temp.replace(target);path.unlink()
    return dict(action='gzip_verified',source=str(path.relative_to(HERE)),archive=str(target.relative_to(HERE)),
                original_bytes=before.st_size,retained_bytes=after.st_size,
                allocated_bytes_freed=(before.st_blocks-after.st_blocks)*512,sha256=original_hash)


def recover():
    rows=load(NEW/'all_trials.json');old_rows=load(OLD/'confirmation/all_trials.json')
    new_rows=load(NEW/'trials.json');assert len(rows)==len(old_rows)+len(new_rows)
    assert all(row['status']=='complete' for row in rows)
    teacher=load(OLD/'recommendations.json')['teacher_embedding']
    frame_rows=load(OLD/'teacher_embedding_trials.json')
    retained={row['trial']['checkpoint']:OLD/'confirmation/selected'/name/('state_student.pt' if name=='teacher' else 'student.pt')
              for name,row in load(OLD/'confirmation/recommendations.json').items()}
    retained[teacher['checkpoint']]=OLD/'confirmation/selected/teacher/frame_student.pt'
    result=dict(status='stopped_by_user',gpu_used_for_recovery=False,old_state_trials=len(old_rows),
        new_completed_trials=len(new_rows),total_state_and_group_trials=len(rows),teacher_embedding_trials=len(frame_rows),
        total_completed_trials=len(rows)+len(frame_rows),
        last_phase=load(NEW/'progress_at_stop.json') if (NEW/'progress_at_stop.json').exists() else load(NEW/'progress.json'),
        temporary_recordings_and_new_trial_weights='Already absent from /tmp at start of recovery.',
        teacher_embedding_lr=teacher['lr'],windows={})
    for window in WINDOWS:
        groups=dict(state={},embedding={});result['windows'][str(window)]=groups
        for name in ORDER:
            candidates=[row for row in rows if row['model']==name and seconds(row)==window]
            state=rank([row for row in candidates if 'embedding_lr' not in row])
            if state is None:continue
            groups['state'][name]=state
            embedding=[row for row in candidates if 'embedding_lr' in row and row['lr']==state['lr']]
            if embedding:
                by_state=rank(embedding,'embedding_lr','score')
                by_jepa=rank(embedding,'embedding_lr','embedding_score')
                groups['embedding'][name]=dict(state_lr=state['lr'],embedding_lr=by_state['lr'],
                    embedding_loss_winner_lr=by_jepa['lr'],state_scores=by_state['normalized_scores'],
                    jepa_scores=by_jepa['normalized_scores'])
    snapshot=load(NEW/'window_results.json')
    for window,groups in snapshot.items():
        for name,row in groups['state'].items():assert result['windows'][window]['state'][name]['lr']==row['lr']
        for name,row in groups['embedding'].items():
            for key in ['embedding_lr','embedding_loss_winner_lr']:
                assert result['windows'][window]['embedding'][name][key]==row[key]
    result['state_windows_recovered']=sum(len(group['state']) for group in result['windows'].values())
    result['embedding_windows_recovered']=sum(len(group['embedding']) for group in result['windows'].values())
    write(HERE/'recovered_lr_results.json',result)
    with (HERE/'recovered_lr_by_window.csv').open('w',newline='') as stream:
        fields=['model','window_seconds','state_lr','embedding_lr_ranked_by_state',
                'embedding_lr_ranked_by_jepa','teacher_global_embedding_lr','state_status','embedding_status']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name in ORDER:
            for window in WINDOWS:
                group=result['windows'][str(window)];state=group['state'].get(name,{});embedding=group['embedding'].get(name,{})
                writer.writerow(dict(model=name,window_seconds=window,state_lr=state.get('lr'),
                    embedding_lr_ranked_by_state=embedding.get('embedding_lr'),
                    embedding_lr_ranked_by_jepa=embedding.get('embedding_loss_winner_lr'),
                    teacher_global_embedding_lr=teacher['lr'] if name=='teacher' else None,
                    state_status='tested' if state else 'not_run',
                    embedding_status='global_frame_pairs' if name=='teacher' else 'tested' if embedding else 'not_run'))
    with (HERE/'recovered_lr_trials.csv').open('w',newline='') as stream:
        fields=['model','component','window_seconds','seed','updates','batch','state_lr','embedding_lr',
                'tail_state_loss','tail_embedding_loss','best_update','best_validation_state_loss',
                'best_validation_sim3_ate_m','best_validation_embedding_loss',
                'checkpoint_path','retained_checkpoint_path','checkpoint_present']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            best=row.get('best_validation') or {}
            writer.writerow(dict(model=row['model'],component='embedding_group' if 'embedding_lr' in row else 'state',
                window_seconds=seconds(row),seed=row['seed'],updates=row['steps'],batch=row['batch'],
                state_lr=row['lr'],embedding_lr=row.get('embedding_lr'),tail_state_loss=row['score'],
                tail_embedding_loss=row.get('embedding_score'),best_update=row['best_step'],
                best_validation_state_loss=best.get('state_loss'),best_validation_sim3_ate_m=best.get('mean_ate_m'),
                checkpoint_path=row['checkpoint'],retained_checkpoint_path=retained.get(row['checkpoint']),
                checkpoint_present=Path(row['checkpoint']).is_file() or
                    (row['checkpoint'] in retained and retained[row['checkpoint']].is_file())))
        for row in frame_rows:
            writer.writerow(dict(model='teacher',component='teacher_embedding',seed=17,updates=256,batch=8,
                embedding_lr=row['lr'],tail_embedding_loss=row['score'],
                best_validation_embedding_loss=row['best_validation'],checkpoint_path=row['checkpoint'],
                retained_checkpoint_path=retained.get(row['checkpoint']),
                checkpoint_present=Path(row['checkpoint']).is_file() or
                    (row['checkpoint'] in retained and retained[row['checkpoint']].is_file())))
    return result


def cleanup():
    manifest_path=HERE/'lr_sweep_cleanup_manifest.json'
    manifest=load(manifest_path) if manifest_path.exists() else dict(actions=[])
    write(manifest_path,manifest)
    # Retain the confirmed selected models; remove superseded first-pass weights.
    for path in (OLD/'selected').rglob('*.pt'):
        if path.is_symlink():continue
        kept=OLD/'confirmation/selected'/path.relative_to(OLD/'selected')
        if not kept.is_file():raise ValueError(f'No confirmed replacement preserved for {path}')
        before=path.stat();original_hash=digest(path);kept_hash=digest(kept)
        path.unlink()
        identical=original_hash==kept_hash
        if identical:path.symlink_to(os.path.relpath(kept,path.parent))
        manifest['actions'].append(dict(action='deduplicated_identical_weights' if identical else 'removed_superseded_initial_weights',
            path=str(path.relative_to(HERE)),original_bytes=before.st_size,retained_bytes=0,
            allocated_bytes_freed=before.st_blocks*512,sha256=original_hash,
            confirmed_checkpoint_retained=str(kept.relative_to(HERE)),confirmed_sha256=kept_hash))
        write(manifest_path,manifest)
    for directory in [OLD,NEW]:
        for path in list(directory.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.name.endswith(('.log','.jsonl','.wandb')):
                action=compress_verified(path)
                if action:
                    manifest['actions'].append(action);write(manifest_path,manifest)
    for filename in ['run.sh','boundary.sh','wait_for_boundaries.py']:
        path=NEW/filename
        backup=NEW/(filename+'.disabled-original.gz')
        if not backup.exists():
            with gzip.open(backup,'wb') as stream:stream.write(path.read_bytes())
        if filename.endswith('.sh'):
            path.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "Disabled: user stopped this sweep. See LR_SWEEP_RECOVERY_README.md." >&2\nexit 1\n')
        else:path.write_text('raise SystemExit("Disabled: user stopped this sweep; no automatic restart.")\n')
    if not (NEW/'progress_at_stop.json').exists():shutil.copy2(NEW/'progress.json',NEW/'progress_at_stop.json')
    write(NEW/'progress.json',dict(phase='stopped_by_user',completed_new_trials=477,last_phase='training_cache',
        last_window_seconds=10,automatic_resume_disabled=True,recovered_results='../../LR_SWEEP_RECOVERY_README.md'))
    write(NEW/'USER_STOPPED.json',dict(gpu_authorized=False,automatic_resume_disabled=True,
        reason='User forced the process to stop after a freeze and requested recovery/cleanup only.'))
    manifest['logical_bytes_freed']=sum(row['original_bytes']-row['retained_bytes'] for row in manifest['actions'])
    manifest['allocated_bytes_freed_observed']=sum(row['allocated_bytes_freed'] for row in manifest['actions'])
    manifest['tmp_at_recovery']='Already cleared; 32 GiB free. No temporary trial weights/recordings remained to recover.'
    manifest['confirmed_models_preserved']=str((OLD/'confirmation/selected').relative_to(HERE))
    write(manifest_path,manifest)
    return manifest


def readme(result,manifest):
    lines=['# Recovered LR sweep results — stopped by user','',
        '**No GPU was used for recovery. No sweep or training process was restarted.**',
        'The saved per-study launchers and queued-boundary launcher have been disabled.',
        '', '## Recovered coverage','',
        '- 477 completed trials from the interrupted extension; all have complete numerical records.',
        '- 190 earlier state trials plus five teacher embedding trials: 672 completed trials across both studies.',
        '- State LR results: 2, 4, 6, 8, 10 and 20 seconds for all nine state models (54/90 model-window settings).',
        '- Separate student embedding-group results: 2, 4, 6 and 8 seconds for all eight students (32/80 settings).',
        '- The 10/20-second state results were recovered from the earlier study; they were not rerun.',
        '- No results exist for 12/14/16/18 seconds. The new 10-second embedding sweep never started.',
        '- The planned higher/lower embedding-rate boundary checks never started.',
        '', 'An em dash means **not tested**, not an interpolated or recommended rate.',
        'All values below are best tested pilot candidates, not proven optimal learning rates.',
        '', '## State learning rates','']
    def table(kind,field,names):
        out=['| Model | '+' | '.join(f'{window} s' for window in WINDOWS)+' |','|---|'+'---:|'*len(WINDOWS)]
        for name in names:
            values=[result['windows'][str(window)][kind].get(name,{}).get(field) for window in WINDOWS]
            out.append('| '+name+' | '+' | '.join(f'{value:g}' if value is not None else '—' for value in values)+' |')
        return out
    lines+=table('state','lr',ORDER)
    lines+=['', '## Embedding-group LR ranked by state loss','',
        'The group includes the online visual encoder and causal embedding predictor. State LR is held at the value above.', '']
    lines+=table('embedding','embedding_lr',ORDER[:-1])
    lines+=['', '## Embedding-group LR ranked by JEPA loss','']
    lines+=table('embedding','embedding_loss_winner_lr',ORDER[:-1])
    lines+=['', '**Teacher embedding LR: 3e-4**, from its earlier frame-pair sweep. That stage has no trajectory-window parameter.',
        '', '## Method and limitations','',
        '- Rates rank the average normalized tail validation loss across fresh seeds 17 and 29.',
        '- Seed 17 used 512 updates; seed 29 used 1,024, matched within each comparison. Teacher embedding used a separate 256-update pilot.',
        '- Validation used eight complete 20-second flights; training used 16 flights from eight different ProcTHOR scenes.',
        '- Existing 2/10/20-second candidate pools differ from the added 4/6/8-second screens. These are shortlisted, bounded searches.',
        '- Student embedding sweeps retain state + 0.1 JEPA + 0.1 alignment, holding state LR fixed. This is not an exhaustive two-LR grid.',
        '- State and JEPA losses can prefer different rates. Several embedding winners are at the highest tested rate (1e-3), without the planned extension.',
        '- The known mid-flight position/velocity target inconsistency was not fixed. Production defaults were not changed.',
        '- These are tuning-validation results, not new independent-test or website-score improvements.',
        '', '## Stop and storage incident','',
        'The last durable progress record was preparation of the 10-second training cache. The shell recorded exit code 137;',
        'that alone cannot distinguish a manual kill from an OOM kill. The user reports force-stopping the process after a freeze.',
        '`/tmp` is a 32 GiB RAM-backed filesystem here, so recordings, caches and trial checkpoints there also consumed host memory.',
        'At the start of recovery `/tmp` was already almost empty (32 GiB free), and about 58 GiB of RAM was available.',
        'No remaining sweep/worker/queued-boundary processes or tmux sessions were found.',
        'Future execution needs bounded storage/RAM usage and disk-backed scratch; the sweep was not restarted or repaired during recovery.',
        '', '## What survived','',
        '- [Recovered structured results](recovered_lr_results.json).',
        '- [Recovered LR table CSV](recovered_lr_by_window.csv), with explicit tested/not-run statuses.',
        '- [All 672 trial summaries](recovered_lr_trials.csv), including the five teacher-embedding trials and retained checkpoint locations.',
        '- [Original combined trial records](checkpoints/lr-windows-20261115-211044/all_trials.json).',
        '- [Teacher embedding trial records](checkpoints/lr-sweep-20261115-200326/teacher_embedding_trials.json).',
        '- [Earlier independent-scene comparison](checkpoints/lr-sweep-20261115-200326/confirmation/independent_check.json).',
        '- [Retained confirmed selected pilot checkpoints](checkpoints/lr-sweep-20261115-200326/confirmation/selected/).',
        '- New trial weights and raw recordings in `/tmp` were already gone; the scalar results survive, but those missing weights cannot be recovered from logs.',
        '', '## Cleanup','']
    if manifest:
        lines+=[f'- Removed/deduplicated preliminary checkpoints and compacted verified logs: {manifest["logical_bytes_freed"]/1024**2:.1f} MiB logical space reclaimed.',
                f'- Filesystem-reported allocated space reclaimed: {manifest["allocated_bytes_freed_observed"]/1024**2:.1f} MiB.',
                '- Confirmed selected pilot checkpoints and their architecture JSON files were retained. Main production checkpoints were untouched.',
                '- Compressed logs have `.gz` suffixes and were decompressed/hash-verified before their originals were removed.',
                '- Original per-study launchers were archived as `.disabled-original.gz`; active copies now exit without starting work.',
                '- [Exact cleanup manifest](lr_sweep_cleanup_manifest.json).']
    lines+=['', '## Existing W&B runs','',
        '- [Interrupted window study](https://forge.coreweave.com/wandb/ysaxena_personal/hyperdrone-vio-fuel-aware/runs/86bc54pq).',
        '- [Earlier confirmation study](https://forge.coreweave.com/wandb/ysaxena_personal/hyperdrone-vio-fuel-aware/runs/xoa55tlu).',
        'Recovery used local records only; no W&B run was restarted or remotely modified.','']
    (HERE/'LR_SWEEP_RECOVERY_README.md').write_text('\n'.join(lines))
    (NEW/'comparison.md').write_text('# Stopped by user — partial study recovered\n\n'
        'No automatic restart. See [the consolidated recovery README](../../LR_SWEEP_RECOVERY_README.md) '
        'for all recovered rates, including the earlier 10/20-second state results, missing settings and cleanup details.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cleanup',action='store_true')
    args=parser.parse_args();result=recover()
    manifest=cleanup() if args.cleanup else load(HERE/'lr_sweep_cleanup_manifest.json') if (HERE/'lr_sweep_cleanup_manifest.json').exists() else None
    readme(result,manifest)
    print(json.dumps(dict(state_windows=result['state_windows_recovered'],embedding_windows=result['embedding_windows_recovered'],
        new_completed_trials=result['new_completed_trials'],cleanup_logical_bytes_freed=manifest['logical_bytes_freed'] if manifest else 0)))


if __name__=='__main__':main()
