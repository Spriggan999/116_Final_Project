"""Write a human-readable LR comparison and fresh-training configurations."""
import argparse
import json
from pathlib import Path

from lr_sweep import write_json


def scientific(value):
    return f'{value:.0e}'.replace('e-0','e-').replace('e+0','e+')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sweep',type=Path)
    args=parser.parse_args();root=args.sweep;folder=root/'confirmation'
    recommendations=json.loads((folder/'recommendations.json').read_text())
    independent=json.loads((folder/'independent_check.json').read_text()) if (folder/'independent_check.json').exists() else {}
    trials=json.loads((folder/'all_trials.json').read_text())
    embedding_trials=json.loads((root/'teacher_embedding_trials.json').read_text())
    embedding=json.loads((root/'recommendations.json').read_text())['teacher_embedding']
    improvers=[name for name,row in independent.items()
               if row['state_loss_change_pct']<0 and row['ate_change_pct']<0]
    outcome=('Candidates improving both independent state loss and aligned ATE: '+(', '.join(improvers) or 'none')+'.'
             if independent else 'Independent comparison has not been completed.')
    lines=['# Fresh-checkpoint learning-rate study','',
        'These are best tested rates for a bounded fixed-data SL pilot, not proven global optima.',
        'All candidates use matched initial weights, sample order and update budgets within each condition.',
        'Final ranking averages normalized tail validation-state loss across 2/10/20-second horizons and two state seeds.',
        'The original LR is included in every confirmation condition. Teacher embedding training is a separate single-seed pilot.',
        f'Completed {len(trials)} state trials and {len(embedding_trials)} embedding trials; '
        f'{sum(row["status"]!="complete" for row in trials)} state trials failed.',
        '',
        'Production defaults are unchanged. The aggregate winners below are experimental candidates, not blanket recommendations.',
        outcome,
        'Use the independent results before adopting a candidate. Investigate the window-target limitation before',
        'a full curriculum rerun. The generated configs reproduce pilot candidates and require that judgment.',
        '', '| Model | Current LR | Best tested LR | Normalized validation-score reduction | Independent state-loss change |',
        '|---|---:|---:|---:|---:|']
    order=['1k','10k','100k','1m_ssm','1m','1m_hybrid','100m','100m_hybrid','teacher']
    for name in order:
        row=recommendations[name];change=independent.get(name,{}).get('state_loss_change_pct')
        held=f'{change:+.2f}%' if change is not None else 'not evaluated'
        lines.append(f'| {name} | {scientific(row["baseline_lr"])} | {scientific(row["lr"])} | {row["relative_score_reduction_pct"]:.2f}% | {held} |')
    lines+=['',f'Teacher embedding LR: {scientific(embedding["lr"])} (existing 3e-4).',
        '', 'The validation column is a normalized state-loss comparison, not a website ATE/area improvement.',
        'Negative independent state-loss changes mean improvement; positive means regression.',
        'Independent scenes 160–167 were not used to select rates. Those checkpoints were selected on tuning validation scenes,',
        'and are pilot-trained checkpoints from matched seed-29 20-second trials.',
        '', '## Existing W&B evidence', '',
        'The stopped student run completed 607 epochs at a constant 3e-4 for every student, without a curriculum promotion.',
        'Validation regressions motivate testing lower rates, but do not establish that LR caused those regressions.']
    teacher_history=root/'existing_teacher_wandb_analysis.json'
    if teacher_history.exists():
        previous=json.loads(teacher_history.read_text());history=previous['history']
        lines.append(f'The previous teacher SL run reduced its LR from {history[0]["student/lr"]:g} '
                     f'to {history[-1]["student/lr"]:g} over {len(history)} epochs. '
                     'The teacher baseline in this study is its starting LR, 1e-4; these bounded trials use constant rates.')
    if independent:
        lines+=['', '## Independent-scene metrics', '',
            'ATE values below are mean Sim(3)-aligned camera-position RMSE over eight complete 20-second flights,',
            'not website scores. State loss and aligned ATE can favor different rates.', '',
            '| Model | Baseline state loss | Selected state loss | Baseline ATE (m) | Selected ATE (m) | Selected checkpoint update |',
            '|---|---:|---:|---:|---:|---:|']
        for name in order:
            before=independent[name]['baseline'];after=independent[name]['selected']
            lines.append(f'| {name} | {before["state_loss"]:.3f} | {after["state_loss"]:.3f} | '
                         f'{before["mean_ate_m"]:.4f} | {after["mean_ate_m"]:.4f} | {after["best_step"]} |')
    lines+=['', '## Per-horizon results','', '| Model | 2 s LR | 10 s LR | 20 s LR |','|---|---:|---:|---:|']
    for name in order:
        row=recommendations[name]['best_lr_by_horizon']
        lines.append('| '+name+' | '+' | '.join(scientific(row[str(h)]) for h in [2,10,20])+' |')
    lines+=['','## Interpretation and limits','',
        '- Eight ProcTHOR training scenes, 16 training flights; eight separate tuning-validation scenes and eight additional comparison scenes.',
        '- Initial trials: 512 state updates, 256 teacher embedding updates. Second-seed trials: 1,024 state updates.',
        '- At 2 seconds, batches are 8 for students and 4 for teacher state; at 10/20 seconds, both use batch 1. Teacher embedding uses batch 8.',
        '- The old teacher used embedding batch 64 and configured state batch 8. LR conclusions are conditional on the pilot batch sizes.',
        '- Fresh student/teacher-state weights for every candidate. Teacher state candidates share one encoder trained fresh within this study.',
        '- No HSSD, curriculum advancement, augmentation changes, scheduler changes or muP conversion.',
        '- Candidate rates may behave differently over much longer training or maximum augmentation. Teacher PPO was not tuned.',
        '- A rate winning at a search boundary or differing little from baseline is not an established optimum.',
        '- Check `best_step`: zero means the candidate checkpoint never improved on its initialization.',
        '- Mid-flight window velocity/position target consistency is a separate limitation; see `window_target_audit.json`.',
        '', '## Files','',
        '- `confirmation/student_lrs.json`: per-student LR configuration for a fresh run.',
        '- `confirmation/fresh_teacher_lr_config.json`: teacher SL settings with the selected embedding/state rates and pilot-compatible batch sizes.',
        '- `confirmation/selected/`: pilot student and teacher weights.',
        '- `confirmation/all_trials.json`: full trial records and checkpoint locations.',
        '- `confirmation/independent_check.json`: separate baseline-versus-selected scene comparison.',
        '- Trial checkpoints and raw recordings remain on temporary storage; preserve them before cleanup.',
        '', '## W&B','']
    for label,path in [('Initial sweep',root/'wandb.json'),('Confirmation',folder/'wandb.json'),('Independent comparison',folder/'holdout_wandb.json')]:
        if path.exists():lines.append(f'- [{label}]({json.loads(path.read_text())["url"]})')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    base=json.loads(Path('curriculum.json').read_text());base.update(json.loads(Path('supervised_procthor.json').read_text()))
    base.update(LR_FRAME=embedding['lr'],LR_STUDENT=recommendations['teacher']['lr'],STUDENT_HIDDEN=896,
                USE_HYBRID_SSM_TRANSFORMER=True,USE_LEARNED_VISUAL_ENCODER=True,VISUAL_PREDICTION='embedding',
                EMBEDDING_DIM=48,EMBEDDING_HIDDEN=256,USE_TEACHER=False,BATCH=4,FRAME_PAIR_BATCH=8)
    write_json(folder/'fresh_teacher_lr_config.json',base)
    selected=folder/'selected/teacher'
    source=root/'selected/teacher/training_config.json'
    if source.exists():
        saved=json.loads(source.read_text());saved.update(LR_FRAME=embedding['lr'],LR_STUDENT=recommendations['teacher']['lr'])
        write_json(selected/'training_config.json',saved)
    write_json(folder/'comparison.json',dict(recommendations=recommendations,independent_check=independent,
                                             teacher_embedding=embedding,provisional=True,
                                             production_defaults_changed=False))
    if independent:
        try:import matplotlib
        except ImportError:
            print(root/'comparison.md')
            print('Report/configs saved; install matplotlib to generate the optional comparison chart.')
            return
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values=[independent[name]['state_loss_change_pct'] for name in order]
        fig,ax=plt.subplots(figsize=(10,5))
        ax.barh(order,values,color=['#bb4433' if v>0 else '#287d67' for v in values])
        ax.axvline(0,color='black',linewidth=.8);ax.invert_yaxis()
        ax.set_xlabel('State-loss change vs original LR (%) — negative is better')
        ax.set_title('Independent-scene check of pilot-selected learning rates')
        fig.tight_layout();fig.savefig(root/'independent_state_loss.png',dpi=160);plt.close(fig)
    print(root/'comparison.md')


if __name__=='__main__':main()
