import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gated_lr_sweep as sweep

PHYSICAL = dict(position_rmse_m=1., velocity_rmse_mps=1., orientation_rmse_deg=1., angular_velocity_rmse_radps=1.)


def fake_run(path, profile, checks, ate, physical=None, skipped=0., parent=None, start=0):
    """Write the files gated_engine produces that the sweep reads."""
    path.mkdir(parents=True)
    (path/'distillation.json').write_text(json.dumps(dict(warm_start=str(parent) if parent else None,
                                                          warm_start_stage=parent is not None)))
    (path/profile).mkdir()
    rows = []
    for check in range(start+1, checks+1):
        row = {'epoch': check, 'fresh_checks': check, 'procthor_curriculum_val/fresh_data_check': True,
               'procthor_val/unaligned_ate_rmse_m': ate, 'train_optimizer_update_skipped': skipped}
        row.update({'procthor_curriculum_val/'+k: v for k, v in (physical or PHYSICAL).items()})
        rows.append(json.dumps(row))
    (path/profile/'metrics.jsonl').write_text('\n'.join(rows)+'\n')


class SweepTests(unittest.TestCase):
    def test_source_is_nearest_log_horizon_with_trained_tiebreak_and_override(self):
        sources = [(.005, 80, Path('/a')), (5.12, 2375, Path('/b')), (10.24, 2530, Path('/c'))]
        self.assertEqual(sweep.choose_source(sources, .16)[0], Path('/b'))   # 32x either way
        self.assertEqual(sweep.choose_source(sources, 2.56)[0], Path('/b'))
        self.assertEqual(sweep.choose_source(sources, .01)[0], Path('/a'))
        self.assertEqual(sweep.choose_source(sources, 20.48)[0], Path('/c'))
        self.assertEqual(sweep.choose_source(sources, .16, {.16: '/a'})[0], Path('/a'))
        with self.assertRaises(ValueError):sweep.choose_source(sources, .16, {.16: '/missing'})

    def test_rank_guardrails_and_tiebreaks(self):
        def r(lr, ate, skip=0., pos=1.):
            return dict(lr=lr, summary=dict(ate=ate, skip_fraction=skip, numerically_valid=skip < .01,
                                            physical=dict(PHYSICAL, position_rmse_m=pos)))
        control = r(3e-4, 100.)['summary']
        ranked = sweep.rank([r(3e-4, 100.), r(1e-4, 80.), r(3e-5, 81.), r(1e-5, 50., pos=1.2)], control)
        # 1e-5 violates the 10% physical guardrail; 3e-5 ties 1e-4 within 2% and is lower.
        self.assertEqual([x['lr'] for x in ranked], [3e-5, 1e-4, 3e-4])
        ranked = sweep.rank([r(1e-4, 80.), r(3e-5, 81., skip=.02)], control)
        self.assertEqual([x['lr'] for x in ranked], [1e-4])

    def test_summary_follows_continuations_and_requires_budget(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fake_run(root/'first', 'baseline', 30, 200.)
            fake_run(root/'second', 'baseline', 100, 100., parent=root/'first', start=30)
            self.assertIsNone(sweep.summary(root/'first', 'baseline', 100))
            value = sweep.summary(root/'second', 'baseline', 100)
            self.assertEqual(value['checks'], 100)
            self.assertEqual(value['ate'], 100.)       # last ten checks only
            self.assertTrue(value['numerically_valid'])

    def test_refine_runs_staged_decision_to_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plan = dict(out=str(root/'runs'), validation_manifest='m.json', checkpoint_root=str(root),
                        horizons=[20.48], sources={'20.48': dict(run=str(root/'src'))}, tasks=[])
            ates = {1e-5: 90., 3e-5: 70., 1e-4: 60., 3e-4: 100.}
            for lr in sweep.RATES:plan['tasks'].append(sweep.task(plan, 'baseline', 20.48, lr, 30))
            def complete():
                for t in plan['tasks']:
                    run = Path(t['run'])
                    if run.exists():continue
                    parent = t['command'][t['command'].index('--warm-start')+1] if '--warm-start-stage' in t['command'] else None
                    fake_run(run, t['profile'], t['checks'], ates[t['lr']], parent=parent,
                             start=30 if parent else 0)
            complete()
            added = sweep.refine(plan)
            # Finalists are 1e-4 and 3e-5: continue baseline, start combined.
            self.assertEqual(sorted(added), sorted(['baseline-h20.48-lr0.0001-n100', 'combined-h20.48-lr0.0001-n30',
                                                    'baseline-h20.48-lr3e-05-n100', 'combined-h20.48-lr3e-05-n30']))
            for _ in range(4):
                complete();sweep.refine(plan)
            self.assertEqual(plan['decisions']['20.48'], dict(status='selected', lr=1e-4))
            self.assertEqual(sweep.schedule(plan), [dict(horizon_seconds=20.48, lr=1e-4)])
            continuation = next(t for t in plan['tasks'] if t['id'] == 'baseline-h20.48-lr0.0001-n100')
            self.assertIn('--warm-start-stage', continuation['command'])
            self.assertEqual(continuation['command'][continuation['command'].index('--resume-lr')+1], '0.0001')

    def test_diverging_candidates_are_ranked_out_without_stopping(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plan = dict(out=str(root/'runs'), validation_manifest='m.json', checkpoint_root=str(root),
                        horizons=[20.48], sources={'20.48': dict(run=str(root/'src'))}, tasks=[])
            ates = {1e-5: 90., 3e-5: 70., 1e-4: 60.}
            for lr in sweep.RATES:plan['tasks'].append(sweep.task(plan, 'baseline', 20.48, lr, 30))
            def complete():
                for t in plan['tasks']:
                    if Path(t['run']).exists() or t.get('failed'):continue
                    # 3e-4 diverges everywhere; combined 1e-4 diverges too.
                    if t['lr'] == 3e-4 or (t['profile'] == 'combined' and t['lr'] == 1e-4):
                        t['failed'] = 'diverged';continue
                    parent = t['command'][t['command'].index('--warm-start')+1] if '--warm-start-stage' in t['command'] else None
                    fake_run(Path(t['run']), t['profile'], t['checks'], ates[t['lr']], parent=parent,
                             start=30 if parent else 0)
            for _ in range(6):
                complete();sweep.refine(plan)
            self.assertEqual(plan['decisions']['20.48'], dict(status='selected', lr=3e-5))
            ids = {t['id'] for t in plan['tasks']}
            self.assertNotIn('combined-h20.48-lr0.0001-n100', ids)   # nothing to continue
            self.assertNotIn('baseline-h20.48-lr0.0003-n100', ids)
            self.assertEqual(sweep.pending(plan), [])

    def test_run_task_marks_persistent_failure(self):
        with tempfile.TemporaryDirectory() as root:
            t = dict(id='x', profile='baseline', checks=30, run=str(Path(root)/'x'),
                     command=[sys.executable, '-c', 'raise SystemExit(3)'])
            saves = []
            self.assertFalse(sweep.run_task(t, lambda: saves.append(1), attempts=2))
            self.assertIn('last exit 3', t['failed'])
            self.assertTrue(saves)
            self.assertTrue(Path(t['run']+'.log').exists())

    def test_interrupted_task_continues_into_new_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plan = dict(out=str(root/'runs'), validation_manifest='m.json', sources={'20.48': dict(run=str(root/'src'))})
            t = sweep.task(plan, 'baseline', 20.48, 1e-4, 30, architecture='persistent_visual_v2')
            fake_run(Path(t['run']), 'baseline', 12, 50.)
            for name in ('recovery_state.json', 'latest.pt'):(Path(t['run'])/name).write_text('{}')
            first = t['run']
            sweep.resume_task(t)
            c = t['command']
            self.assertEqual(t['run'], first+'-r1')
            self.assertEqual(c[c.index('--out')+1], first+'-r1')
            self.assertEqual(c[c.index('--warm-start')+1], first)
            self.assertIn('--warm-start-stage', c)
            self.assertNotIn('--init-weights', c)
            self.assertNotIn('--allow-architecture-transfer', c)
            self.assertEqual(c[c.index('--resume-lr')+1], '0.0001')
            self.assertEqual(c.count('--out'), 1)
            fake_run(Path(t['run']), 'baseline', 30, 40., parent=first, start=12)
            self.assertEqual(sweep.summary(t['run'], 'baseline', 30)['checks'], 30)
            # No completed epoch: restart the original arguments in another new directory.
            Path(t['run'], 'recovery_state.json').unlink(missing_ok=True)
            sweep.resume_task(t)
            self.assertEqual(t['run'], first+'-r1-r1')
            self.assertEqual(t['attempts'], [first, first+'-r1'])

    def test_schedule_merges_equal_neighbours(self):
        plan = dict(horizons=[.16, 2.56, 10.24], decisions={'0.16': dict(lr=3e-4), '2.56': dict(lr=3e-4),
                                                             '10.24': dict(lr=3e-5)})
        self.assertEqual(sweep.schedule(plan), [dict(horizon_seconds=.16, lr=3e-4), dict(horizon_seconds=10.24, lr=3e-5)])
        plan['decisions'].pop('10.24')
        with self.assertRaises(ValueError):sweep.schedule(plan)


if __name__ == '__main__':unittest.main()
