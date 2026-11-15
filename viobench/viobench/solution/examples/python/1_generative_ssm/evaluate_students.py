"""Evaluate deployed students + teacher and reproduce the website's area score.

No uploads. A supplied baseline snapshot is a proxy unless its episode hashes
and scoring configuration match the reports. Server comparison JSON is exact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import isfinite,log
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[4]))
from viobench.tools.server.evaluations import dominated_area,FUEL_FLOOR,ERROR_FLOOR
from viobench.tools.runner.leaderboard import pareto_front


def score_portfolio(own,baselines,step_fuel=20_000_000_000):
    """Standalone gain and leave-one-out gain share a fixed baseline reference."""
    def xy(row):return max(row['fuel_per_frame_max'],FUEL_FLOOR),max(row['sim3_ate_rmse_mean'],ERROR_FLOOR)
    for row in own+baselines:
        if not all(isfinite(row[key]) and row[key]>=0 for key in ('fuel_per_frame_max','sim3_ate_rmse_mean')):
            raise ValueError('Comparison requires finite nonnegative metrics')
    points=[xy(row) for row in own];base=[xy(row) for row in baselines]
    front=set(pareto_front(base+points))
    result=dict(points=[dict(row,on_frontier=len(base)+i in front) for i,row in enumerate(own)])
    if not base:
        result.update(claimed_percent=None,error='No eligible baselines; website area is unavailable')
        return result
    x=1.2*step_fuel;y=1.2*max(error for _,error in base)
    denominator=log(x/FUEL_FLOOR)*log(y/ERROR_FLOOR)
    area=lambda values:dominated_area(values,x,y)
    baseline_area=area(base);combined=area(base+points)
    for i,row in enumerate(result['points']):
        row['standalone_gain_percent']=100*max(0,area(base+[points[i]])-baseline_area)/denominator
        row['unique_gain_percent']=100*max(0,combined-area(base+points[:i]+points[i+1:]))/denominator
    result.update(claimed_percent=100*max(0,combined-baseline_area)/denominator,
                  fuel_reference=x,error_reference=y,
                  ranked_standalone=sorted(result['points'],key=lambda row:row['standalone_gain_percent'],reverse=True))
    return result


def plot_comparison(result,baselines,path):
    """Draw the website's log-space reference rectangle and claimed area."""
    if not baselines:return True
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'hyperdrone-mpl'))
    try:import matplotlib
    except ImportError:return False
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=[(max(row['fuel_per_frame_max'],FUEL_FLOOR),max(row['sim3_ate_rmse_mean'],ERROR_FLOOR)) for row in baselines]
    own=[(max(row['fuel_per_frame_max'],FUEL_FLOOR),max(row['sim3_ate_rmse_mean'],ERROR_FLOOR)) for row in result['points']]
    xref,yref=result['fuel_reference'],result['error_reference']
    xs=sorted({FUEL_FLOOR,xref,*[x for x,y in base+own if x<xref]})
    def envelope(points):return [min([yref]+[y for x,y in points if x<=at]) for at in xs]
    figure,axis=plt.subplots(figsize=(10,6))
    axis.step(xs,envelope(base),where='post',linestyle='--',color='grey',label='Baseline frontier')
    axis.step(xs,envelope(base+own),where='post',color='C0',label='Combined frontier')
    axis.fill_between(xs,envelope(base+own),envelope(base),step='post',alpha=.2,color='C0')
    axis.scatter(*zip(*base),marker='s',color='grey',label='Website baselines')
    for row,(x,y) in zip(result['points'],own):
        axis.scatter(x,y,color='C0' if row['on_frontier'] else 'C3')
        axis.annotate(row['label'],(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    axis.set(xscale='log',yscale='log',xlim=(FUEL_FLOOR,xref),ylim=(ERROR_FLOOR,yref),
             xlabel='Measured maximum fuel per frame',ylabel='Mean Sim(3) ATE (m)',
             title='Exact website scores' if result['exact_website_comparison'] else 'Approximate baseline-snapshot comparison')
    axis.legend();axis.grid(True,which='both',alpha=.2);figure.tight_layout();figure.savefig(path,dpi=160);plt.close(figure)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--students',type=Path,help='distillation root with profile/native/student.wasm')
    parser.add_argument('--teacher',type=Path,help='exported teacher WASM')
    parser.add_argument('--episodes',nargs='*',help='same VIOBench episode paths/hashes for every model; default public')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=2,help='bounded CPU evaluation processes')
    parser.add_argument('--baseline-snapshot',type=Path,default=HERE/'website_baselines.json')
    parser.add_argument('--server-comparison',type=Path,help='saved GET /comparison JSON; score these server results directly')
    parser.add_argument('--limits',type=Path,help='saved server evaluation.limits; student fuel caps remain stricter')
    parser.add_argument('--reports-only',action='store_true',help='rescore existing reports without executing WASM')
    parser.add_argument('--plot',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.server_comparison:
        source=json.loads(args.server_comparison.read_text())
        own=[row for row in source['points'] if row.get('own') and row.get('team_type')!='baseline']
        baseline=[row for row in source['points'] if row.get('team_type')=='baseline']
        result=score_portfolio(own,baseline,source['fuel_reference']/1.2)
        result.update(exact_website_comparison=True,evaluation_hashes=source['evaluation_hashes'])
    else:
        from students.model import PROFILES
        from verify_limits import LIMIT_VALUES
        limits=json.loads(args.limits.read_text()) if args.limits else dict(LIMIT_VALUES)
        modules={}
        if args.students:
            modules.update({name:(args.students/name/'native/student.wasm',spec['fuel']) for name,spec in PROFILES.items()})
        if args.teacher:modules['teacher']=(args.teacher,limits['step_fuel'])
        if not modules:parser.error('provide --students and/or --teacher, or --server-comparison')
        def evaluate(item):
            name,(module,cap)=item;directory=args.out/name;directory.mkdir(exist_ok=True)
            if not args.reports_only:
                if not module.is_file():raise FileNotFoundError(module)
                local=dict(limits,step_fuel=min(cap,limits['step_fuel']))
                config=directory/'evaluation_limits.json';config.write_text(json.dumps(local))
                command=[sys.executable,'-m','viobench','evaluate',str(module.resolve()),*(args.episodes or []),
                         '--limits',str(config.resolve()),'-o',str(directory.resolve())]
                with (directory/'evaluate.log').open('w') as stream:
                    completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                if completed.returncode and not (directory/'report.json').is_file():
                    raise RuntimeError(f'{name}: evaluator failed; see {directory}/evaluate.log')
            report=json.loads((directory/'report.json').read_text());summary=report['summary']
            if len(report['episodes'])!=summary['episodes']:raise ValueError(f'{name}: incomplete report')
            return dict(label=name,fuel_per_frame_max=summary['fuel_per_frame_max'],
                sim3_ate_rmse_mean=summary['sim3_ate_rmse_mean'],completed=summary['completed'],episodes=summary['episodes'],
                elapsed_s=report['elapsed_s'],evaluation_hashes=[row['hash'] for row in report['episodes']])
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:own=list(pool.map(evaluate,modules.items()))
        selected=own[0]['evaluation_hashes']
        if any(row['evaluation_hashes']!=selected for row in own):raise ValueError('Model reports use different episode selections')
        source=json.loads(args.baseline_snapshot.read_text())
        baseline=source['points']
        result=score_portfolio(own,baseline,limits['step_fuel'])
        result.update(exact_website_comparison=(source.get('evaluation_hashes')==selected and source.get('limits')==limits),
                      evaluation_hashes=selected,baseline_provenance=source.get('provenance'))
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    title='Exact website comparison' if result['exact_website_comparison'] else 'Approximate baseline-snapshot comparison; episode/settings match unverified'
    lines=[title,'','| Model | Completed | Fuel/frame | Mean ATE (m) | Frontier | Standalone gain (%) | Unique gain (%) |',
           '|---|---|---:|---:|---|---:|---:|']
    for row in result['points']:
        completed=f"{row['completed']}/{row['episodes']}" if 'completed' in row else 'unavailable'
        standalone=f"{row['standalone_gain_percent']:.6g}" if 'standalone_gain_percent' in row else 'unavailable'
        unique=f"{row['unique_gain_percent']:.6g}" if 'unique_gain_percent' in row else 'unavailable'
        lines.append(f"| {row['label']} | {completed} | {row['fuel_per_frame_max']:,.0f} | {row['sim3_ate_rmse_mean']:.6g} | {row['on_frontier']} | {standalone} | {unique} |")
    lines+=['',f"Combined gain: {result.get('claimed_percent')}%.",
            'Standalone gains overlap and must not be added. Unique gain is the area lost when removing that model.']
    (args.out/'comparison.md').write_text('\n'.join(lines)+'\n')
    if args.plot and not plot_comparison(result,baseline,args.out/'pareto.png'):
        print('Optional Pareto PNG skipped: install matplotlib to enable plotting.',file=sys.stderr)
    print('\n'.join(lines))


if __name__=='__main__':main()
