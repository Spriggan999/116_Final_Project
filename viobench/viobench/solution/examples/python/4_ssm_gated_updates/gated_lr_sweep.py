"""Matched, staged horizon LR sweeps. Planning never starts GPU training."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
RATES=[1e-5,3e-5,1e-4,3e-4]
HORIZONS=[.16,2.56,10.24,20.48]
PHYSICAL=['position_rmse_m','velocity_rmse_mps','orientation_rmse_deg','angular_velocity_rmse_radps']


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def summary(run,profile,expected_checks=None):
    run=Path(run);path=run/profile/'metrics.jsonl'
    if not path.exists():return None
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    fresh=[r for r in rows if r.get('procthor_curriculum_val/fresh_data_check')]
    config=json.loads((run/'distillation.json').read_text())
    # Continuations contribute only their new checks; consult their ancestry.
    if config.get('warm_start_stage'):
        parent=Path(config['warm_start'])/profile/'metrics.jsonl'
        while parent.exists():
            ancestor=Path(parent).parent.parent
            older=[json.loads(line) for line in parent.read_text().splitlines()]
            fresh=[r for r in older if r.get('procthor_curriculum_val/fresh_data_check')]+fresh
            cfg=json.loads((ancestor/'distillation.json').read_text())
            if not cfg.get('warm_start_stage'):break
            parent=Path(cfg['warm_start'])/profile/'metrics.jsonl'
    by_epoch={r['epoch']:r for r in fresh};fresh=[by_epoch[k] for k in sorted(by_epoch)]
    completed=rows[-1].get('fresh_checks',len(fresh)) if rows else 0
    if not rows or (expected_checks is not None and completed<expected_checks):return None
    tail=fresh[-10:]
    def mean(key):return statistics.mean(r[key] for r in tail)
    skips=statistics.mean(r.get('train_optimizer_update_skipped',0) for r in fresh)
    values=dict(ate=mean('procthor_val/unaligned_ate_rmse_m'),skip_fraction=skips,
                physical={k:mean('procthor_curriculum_val/'+k) for k in PHYSICAL},checks=completed)
    values['numerically_valid']=skips<.01 and all(math.isfinite(v) for v in [values['ate'],*values['physical'].values()])
    return values


def rank(results,control=None):
    """Reject physical regressions; ties within 2% favor fewer skips then lower LR."""
    valid=[r for r in results if r['summary']['numerically_valid'] and
           (control is None or all(r['summary']['physical'][k]<=1.1*control['physical'][k] for k in PHYSICAL))]
    ranked=[]
    while valid:
        best=min(r['summary']['ate'] for r in valid)
        tied=[r for r in valid if r['summary']['ate']<=best*1.02]
        winner=min(tied,key=lambda r:(r['summary']['skip_fraction'],r['lr']))
        ranked.append(winner);valid.remove(winner)
    return ranked


def available_sources(root):
    result=[]
    for p in Path(root).iterdir():
        try:
            state=json.loads((p/'recovery_state.json').read_text())
            curriculum=state['curriculum']
            horizon=(curriculum['windows'][curriculum['window_stage']]-1)/200
            if (p/'latest.pt').exists():result.append((horizon,state['completed_epochs'],p.resolve()))
        except (OSError,KeyError,TypeError):continue
    return result


def choose_source(sources,horizon,overrides=None):
    """Nearest saved horizon on a log scale; ties favor the more-trained checkpoint.

    Only one snapshot per run survives, so no checkpoint was saved at most
    requested horizons. A 2x mismatch adapts better than a 512x one.
    """
    if overrides and horizon in overrides:
        run=Path(overrides[horizon]).resolve()
        match=[s for s in sources if s[2]==run]
        if not match:raise ValueError(f'--source {horizon:g}: {run} has no recoverable latest.pt')
        return run,match[0][0]
    chosen=min(sources,key=lambda s:(round(abs(math.log(s[0]/horizon)),6),-s[1]))
    return chosen[2],chosen[0]


def task(plan,profile,horizon,lr,checks,parent=None,architecture='legacy',label=None):
    tag=label or f'{profile}-h{horizon:g}-lr{lr:g}-n{checks}'
    source=Path(parent) if parent else Path(plan['sources'][str(horizon)]['run'])
    out=Path(plan['out'])/tag
    command=[sys.executable,str(HERE/'train.py'),'--no-public60','--warm-start',str(source),
        '--training-policy','corrected_v1','--architecture',architecture,'--profiles',profile,
        '--config',str(source/'training_config.json'),'--validation-manifest',plan['validation_manifest'],
        '--fixed-horizon','--window',str(round(horizon*200)+1),'--max-fresh-checks',str(checks),
        '--max-passes-per-batch','1','--curriculum-budget-action','stop','--resume-lr',str(lr),
        '--epochs','10000','--seed','17','--device','cuda','--out',str(out),
        '--wandb-mode',plan.get('wandb_mode','online')]
    if parent:command+=['--warm-start-stage']
    else:command+=['--init-weights','latest']
    if architecture!='legacy' and not parent:command+=['--allow-architecture-transfer']
    return dict(id=tag,profile=profile,horizon=horizon,lr=lr,checks=checks,run=str(out),command=command)


def option(command,name,value=None):
    """Return a copy of an argv list with NAME set to VALUE, or removed if None."""
    command=list(command)
    if name in command:
        i=command.index(name)
        has_value=i+1<len(command) and not command[i+1].startswith('--')
        del command[i:i+2 if has_value else i+1]
    if value is not None:command+=[name] if value is True else [name,str(value)]
    return command


def resume_task(t):
    """Continue an interrupted task into a fresh directory; never reuse or delete one.

    With a completed epoch, resume exactly (weights, optimizer, check count);
    otherwise restart the original command. summary() follows the ancestry.
    """
    old=Path(t['run']);k=1
    while old.with_name(f'{old.name}-r{k}').exists():k+=1
    new=old.with_name(f'{old.name}-r{k}')
    if (old/'recovery_state.json').exists() and (old/'latest.pt').exists():
        command=option(t['command'],'--init-weights')
        command=option(command,'--allow-architecture-transfer')
        command=option(command,'--warm-start',old)
        command=option(command,'--config',old/'training_config.json')
        command=option(command,'--warm-start-stage',True)
    else:command=t['command']
    t.setdefault('attempts',[]).append(str(old))
    t['run']=str(new);t['command']=option(command,'--out',new)


def failed_summary(reason):
    """A task that kept failing ranks as numerically invalid instead of halting the sweep."""
    return dict(ate=math.inf,skip_fraction=1.,physical={k:math.inf for k in PHYSICAL},checks=0,
                numerically_valid=False,failed=reason)


def result_for(tasks,profile,horizon,lr,checks):
    matches=[t for t in tasks if t['profile']==profile and t['horizon']==horizon and t['lr']==lr and t['checks']==checks]
    if not matches:return None
    t=matches[-1]
    if t.get('failed'):return dict(lr=lr,summary=failed_summary(t['failed']),task=t)
    value=summary(t['run'],profile,checks)
    return dict(lr=lr,summary=value,task=t) if value else None


def failed(result):
    return bool(result and result['summary'].get('failed'))


def run_task(t,save_plan,attempts=3):
    """Train one task to its check budget, resuming after interruptions.

    Output goes to <run>.log. After `attempts` failed launches the task is
    marked failed (ranked invalid) so one diverging candidate cannot stop the sweep.
    """
    for attempt in range(attempts):
        if Path(t['run']).exists():
            previous=t['run'];resume_task(t);save_plan()
            print(f'Interrupted run {previous}; continuing as {t["run"]}',flush=True)
        log=Path(t['run']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        print(f'Launching {t["id"]} (attempt {attempt+1}); log: {log}',flush=True)
        with log.open('a') as stream:
            code=subprocess.run(t['command'],cwd=HERE.parent,stdout=stream,stderr=subprocess.STDOUT).returncode
        if summary(t['run'],t['profile'],t['checks']):return True
        print(f'{t["id"]} ended before its budget (exit {code}); see {log}',flush=True)
    t['failed']=f'no complete result after {attempts} launches; last exit {code}; see {log}'
    save_plan()
    return False


def pending(plan):
    return [t for t in plan['tasks'] if not t.get('failed') and not summary(t['run'],t['profile'],t['checks'])]


def refine(plan):
    """Add only tasks whose prerequisite comparisons are complete."""
    tasks=plan['tasks'];added=[];decisions={}
    def add(t):
        if t['id'] not in {v['id'] for v in tasks}:tasks.append(t);added.append(t['id'])
    for h in list(plan['horizons']):
        rates=plan.get('rates_by_horizon',{}).get(str(h),RATES)
        initial=[result_for(tasks,'baseline',h,lr,30) for lr in rates]
        if not all(initial):continue
        control=next((r['summary'] for r in initial if r['lr']==3e-4),None)
        finalists=rank(initial,control)[:2]
        if not finalists:
            decisions[str(h)]=dict(status='no_eligible_candidate');continue
        def final(profile,lr):
            # A failed 30-check run is final; there is nothing to continue.
            first=result_for(tasks,profile,h,lr,30)
            return first if failed(first) else result_for(tasks,profile,h,lr,100)
        for r in finalists:
            lr=r['lr'];add(task(plan,'baseline',h,lr,100,parent=r['task']['run']))
            add(task(plan,'combined',h,lr,30))
            first=result_for(tasks,'combined',h,lr,30)
            if first and not failed(first):add(task(plan,'combined',h,lr,100,parent=first['task']['run']))
        finals=[final('baseline',r['lr']) for r in finalists]
        combined=[final('combined',r['lr']) for r in finalists]
        if not all(finals+combined):continue
        # Always confirm a 3e-4 reference, even if eliminated in the first round.
        refs={}
        for profile in ('baseline','combined'):
            r=result_for(tasks,profile,h,3e-4,30)
            if r is None:add(task(plan,profile,h,3e-4,30));continue
            if failed(r):refs[profile]=r;continue
            add(task(plan,profile,h,3e-4,100,parent=r['task']['run']))
            refs[profile]=result_for(tasks,profile,h,3e-4,100)
        if len(refs)<2 or not all(refs.values()):continue
        # Normalize by the 3e-4 reference; if it diverged, by the best finalist.
        scale={}
        for i,p in enumerate(('baseline','combined')):
            ref=refs[p]['summary']
            fallback=[x['summary']['ate'] for x in (finals,combined)[i] if math.isfinite(x['summary']['ate'])]
            scale[p]=ref['ate'] if ref['numerically_valid'] else (min(fallback) if fallback else 1.)
        eligible=[]
        for b,c in zip(finals,combined):
            pair={'baseline':b,'combined':c}
            if all(r['summary']['numerically_valid'] and all(r['summary']['physical'][k]<=1.1*refs[p]['summary']['physical'][k] for k in PHYSICAL) for p,r in pair.items()):
                ratio=statistics.mean(r['summary']['ate']/max(scale[p],1e-12) for p,r in pair.items())
                eligible.append((ratio,b['lr']))
        if eligible:decisions[str(h)]=dict(status='selected',lr=min(eligible)[1])
        elif all(refs[p]['summary']['numerically_valid'] for p in refs):decisions[str(h)]=dict(status='control_retained',lr=3e-4)
        else:decisions[str(h)]=dict(status='no_eligible_candidate')
    plan['decisions']=decisions
    # Resolve only the two explicitly planned intermediate horizons.
    if all(str(h) in decisions for h in HORIZONS):
        for lo,hi,mid in [(.16,2.56,.64),(2.56,10.24,5.12)]:
            if all('lr' in decisions[str(h)] for h in (lo,hi)) and decisions[str(lo)]['lr']!=decisions[str(hi)]['lr'] and mid not in plan['horizons']:
                source,source_h=choose_source(available_sources(plan['checkpoint_root']),mid,
                    {float(k):v for k,v in plan.get('source_overrides',{}).items()})
                plan['sources'][str(mid)]=dict(run=str(source),checkpoint_horizon=source_h)
                plan['horizons'].append(mid)
                rates=sorted({decisions[str(lo)]['lr'],decisions[str(hi)]['lr'],3e-4})
                plan.setdefault('rates_by_horizon',{})[str(mid)]=rates
                for lr in rates:add(task(plan,'baseline',mid,lr,30))
    return added


def schedule(plan):
    decisions=plan.get('decisions',{})
    if any(str(h) not in decisions or 'lr' not in decisions[str(h)] for h in plan['horizons']):
        raise ValueError('Finish all required horizon comparisons before exporting a schedule')
    rows=[]
    for h in sorted(plan['horizons']):
        lr=decisions[str(h)]['lr']
        if not rows or lr!=rows[-1]['lr']:rows.append(dict(horizon_seconds=h,lr=lr))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run','refine','auto','report','schedule','pilot'])
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--checkpoint-root',type=Path,default=HERE/'checkpoints')
    parser.add_argument('--validation-manifest',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--source',action='append',default=[],metavar='HORIZON=RUN',
                        help='plan: initialize a horizon from this run instead of the nearest saved horizon')
    parser.add_argument('--limit',type=int,default=1,help='maximum training subprocesses in this invocation')
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    args=parser.parse_args()
    if args.action=='plan':
        if args.plan.exists():parser.error('Plan already exists; use refine to preserve results')
        if not args.validation_manifest or not args.out:parser.error('plan requires --validation-manifest and --out')
        import legacy
        from gated_data import validation_files
        validation_files(args.validation_manifest)
        sources=available_sources(args.checkpoint_root)
        try:overrides={float(k):v for k,v in (item.split('=',1) for item in args.source)}
        except ValueError:parser.error('--source expects HORIZON=RUN')
        if not sources:parser.error('No recoverable checkpoints found')
        plan=dict(version=1,checkpoint_root=str(args.checkpoint_root.resolve()),validation_manifest=str(args.validation_manifest.resolve()),
                  out=str(args.out.resolve()),horizons=list(HORIZONS),source_overrides={str(k):v for k,v in overrides.items()},wandb_mode=args.wandb_mode,sources={},tasks=[])
        for h in HORIZONS:
            source,source_h=choose_source(sources,h,overrides)
            plan['sources'][str(h)]=dict(run=str(source),checkpoint_horizon=source_h,
                note='Adaptation from available checkpoint; not a historical snapshot at the requested horizon')
            for lr in RATES:plan['tasks'].append(task(plan,'baseline',h,lr,30))
        save(args.plan,plan);print(f'Prepared {len(plan["tasks"])} tasks; no training started.');return
    plan=json.loads(args.plan.read_text())
    if args.action=='refine':
        print('Added tasks:',refine(plan));save(args.plan,plan)
    elif args.action=='schedule':
        save(args.plan.with_name('lr_schedule.json'),schedule(plan))
    elif args.action=='pilot':
        rates=schedule(plan);lr=next(r['lr'] for r in reversed(rates) if r['horizon_seconds']<=20.48)
        if not all(str(h) in plan.get('decisions',{}) for h in HORIZONS):parser.error('Complete the LR sweep first')
        for profile in ('baseline','combined'):
            for architecture in ('legacy','persistent_visual_v2'):
                t=task(plan,profile,20.48,lr,100,architecture=architecture,label=f'pilot-{architecture}-{profile}')
                if t['id'] not in {r['id'] for r in plan['tasks']}:plan['tasks'].append(t)
        save(args.plan,plan)
    elif args.action=='report':
        for t in plan['tasks']:print(t['id'],json.dumps(summary(t['run'],t['profile'],t['checks'])))
        print('Decisions:',json.dumps(plan.get('decisions',{}),indent=2))
        comparisons={}
        for profile in ('baseline','combined'):
            paired={}
            for architecture in ('legacy','persistent_visual_v2'):
                ts=[t for t in plan['tasks'] if t['id']==f'pilot-{architecture}-{profile}']
                paired[architecture]=summary(ts[0]['run'],profile,100) if ts else None
            if all(paired.values()):
                old,new=paired['legacy'],paired['persistent_visual_v2']
                comparisons[profile]=new['numerically_valid'] and new['ate']<=.8*old['ate'] and all(new['physical'][k]<=1.1*old['physical'][k] for k in PHYSICAL)
        print('Redesign accepted:',len(comparisons)==2 and all(comparisons.values()),comparisons)
    elif args.action=='run':
        if args.limit<1:parser.error('--limit must be positive')
        for t in pending(plan)[:args.limit]:run_task(t,lambda:save(args.plan,plan))
    elif args.action=='auto':
        # Unattended: run everything runnable, refine, repeat until decided.
        while True:
            for t in pending(plan):run_task(t,lambda:save(args.plan,plan))
            added=refine(plan);save(args.plan,plan)
            done=len([t for t in plan['tasks'] if not t.get('failed') and summary(t['run'],t['profile'],t['checks'])])
            print(f'Refined: {len(added)} new tasks {added}; {done}/{len(plan["tasks"])} complete; '
                  f'decisions {json.dumps(plan.get("decisions",{}))}',flush=True)
            if not added and not pending(plan):break
        try:
            save(args.plan.with_name('lr_schedule.json'),schedule(plan))
            print('LR sweep complete; schedule written to',args.plan.with_name('lr_schedule.json'),flush=True)
        except ValueError as error:print('LR sweep finished without a full schedule:',error,flush=True)


if __name__=='__main__':main()
