"""Promotion must follow the current held-out distribution, not long-flight ATE."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import legacy
import config
from curriculum import motion_scores
from gated_curriculum import GatedStudentCurriculum, VALIDATION_POLICY
from student_curriculum import StagedStudentCurriculum


def settings():
    return SimpleNamespace(**{key:value for key,value in vars(config).items() if key.isupper()})


class CurriculumValidationTests(unittest.TestCase):
    def update(self,curriculum,index,accuracy=None,loss=.2,ate=100.):
        names=('a','b')
        if accuracy is None:accuracy={key:0. for key in curriculum.current_accuracy_targets()}
        return curriculum.update(str(index),{curriculum.probe_axes[0]:dict.fromkeys(names,loss)},
                                 dict.fromkeys(names,ate),{name:dict(accuracy) for name in names})

    def test_competence_promotes_while_loss_improves_and_full_ate_fluctuates(self):
        curriculum=GatedStudentCurriculum(settings(),[2,3])
        for index in range(6):
            event=self.update(curriculum,index,loss=1./(index+1),ate=100. if index%2 else 1.)
            if index<5:self.assertFalse(event['promoted'])
        self.assertEqual(event['promoted'],['motion'])
        self.assertEqual(event['reasons']['motion'],{'a':'competence','b':'competence'})
        self.assertEqual(curriculum.stage_checks,0)
        self.assertEqual(event['validation_policy'],VALIDATION_POLICY)
        self.assertIsNone(curriculum.update('5',{},{}))

    def test_flat_loss_does_not_promote_poor_accuracy_before_cap(self):
        curriculum=GatedStudentCurriculum(settings(),[2,3],max_stage_checks=8)
        bad={key:target*2 for key,target in curriculum.current_accuracy_targets().items()}
        for index in range(7):
            event=self.update(curriculum,index,bad)
            self.assertFalse(event['promoted'])
            self.assertFalse(curriculum.paused)
        event=self.update(curriculum,7,bad)
        self.assertEqual(event['promoted'],['motion'])
        self.assertTrue(event['forced_promotion'])
        self.assertEqual(set(event['reasons']['motion'].values()),{'budget_exhausted'})

    def test_competence_requires_consecutive_checks_for_every_component_and_model(self):
        curriculum=GatedStudentCurriculum(settings(),[2,3])
        for index in range(5):self.update(curriculum,index)
        bad={key:0. for key in curriculum.current_accuracy_targets()};bad['orientation_rmse_deg']=100.
        self.assertFalse(self.update(curriculum,5,bad)['promoted'])
        for index in (6,7):self.assertFalse(self.update(curriculum,index)['promoted'])
        self.assertEqual(self.update(curriculum,8)['promoted'],['motion'])

    def test_invalid_measurement_does_not_spend_budget(self):
        curriculum=GatedStudentCurriculum(settings(),[2,3])
        with self.assertRaises(ValueError):self.update(curriculum,0,{'position_rmse_m':float('nan')})
        self.assertEqual(curriculum.stage_checks,0)
        self.assertTrue(curriculum.fresh('0'))

    def test_targets_scale_with_duration_and_difficulty(self):
        curriculum=GatedStudentCurriculum(settings(),[2,12001])
        short=curriculum.current_accuracy_targets()
        curriculum.window_stage=1
        long=curriculum.current_accuracy_targets()
        self.assertAlmostEqual(long['position_rmse_m'],1.22)
        curriculum.schedule.stages.update(motion=9,imu=9,visual=9)
        noisy=curriculum.current_accuracy_targets()
        for key in short:
            self.assertGreater(long[key],short[key])
            self.assertAlmostEqual(noisy[key],long[key]*3.5)

    def test_cap_advances_window_and_resets_budget(self):
        curriculum=GatedStudentCurriculum(settings(),[2,3],max_stage_checks=6)
        curriculum.schedule.stages['motion']=9
        bad={key:100. for key in curriculum.current_accuracy_targets()}
        for index in range(6):event=self.update(curriculum,index,bad)
        self.assertTrue(event['window_promoted'])
        self.assertEqual(curriculum.length,3)
        self.assertEqual(curriculum.schedule.stages['motion'],0)
        self.assertEqual(curriculum.stage_checks,0)

    def test_final_stage_never_advances_past_schedule_or_freezes(self):
        curriculum=GatedStudentCurriculum(settings(),[2],max_stage_checks=6)
        curriculum.schedule.stages.update(motion=9,imu=9,visual=9)
        for index in range(8):event=self.update(curriculum,index)
        self.assertEqual(curriculum.phase,'final')
        self.assertFalse(event['promoted'])
        self.assertFalse(event['forced_promotion'])
        self.assertEqual(curriculum.final_cycles,8)
        self.assertIsNone(curriculum.finish_epoch(100,{'a':{},'b':{}},{'a'}))

    def test_one_unqualified_peer_blocks_competence_without_freezing_others(self):
        curriculum=GatedStudentCurriculum(settings(),[2,3])
        good={key:0. for key in curriculum.current_accuracy_targets()}
        bad=dict(good);bad['velocity_rmse_mps']=10.
        for index in range(6):
            event=curriculum.update(str(index),{'motion':{'a':.2,'b':.2}},
                                    {'a':1.,'b':1.},{'a':good,'b':bad})
        self.assertEqual(event['readiness']['motion'],{'a':'competence','b':None})
        self.assertFalse(event['promoted'])
        self.assertFalse(curriculum.paused)

    def test_invalid_budget_and_target_configuration_is_rejected(self):
        for options in ({'max_stage_checks':5},{'accuracy_patience':7},{'targets':{'wrong':[1.,0.]}},
                        {'targets':{'position_rmse_m':[-1.,0.]}}):
            with self.assertRaises(ValueError):GatedStudentCurriculum(settings(),[2,3],**options)

    def test_hybrid_budget_event_logs_without_legacy_timeout_fields(self):
        from gated_tracking import GatedLogger
        class Run:
            def __init__(self):self.summary={};self.logs=[]
            def define_metric(self,*args,**kwargs):pass
            def log(self,record):self.logs.append(record)
        curriculum=GatedStudentCurriculum(settings(),[2,3],max_stage_checks=6)
        bad={key:100. for key in curriculum.current_accuracy_targets()}
        for index in range(6):event=self.update(curriculum,index,bad)
        event['student_probe_losses']={'motion':{'a':.2,'b':.2}}
        run=Run();logger=GatedLogger(run,['a','b'])
        logger.curriculum(6,curriculum,event)
        self.assertTrue(run.logs[-1]['curriculum/forced_promotion'])
        self.assertEqual(run.logs[-1]['curriculum/a/ready_reason'],'budget_exhausted')
        self.assertIn('curriculum/last_forced_promotion',run.summary)

    def test_noise_phase_still_filters_current_motion_and_fixed_horizon(self):
        entries=[]
        for scene in range(2):
            state=torch.zeros(401,18);state[:,3:12]=torch.eye(3).flatten()
            state[:,12]=torch.linspace(scene*3,scene*3+2,401)
            imu=torch.zeros(401,6);imu[:,2]=9.81
            entries.append(dict(state=state,imu=imu,cams=torch.zeros(81,3,16,16,dtype=torch.uint8),
                                cam_ticks=torch.arange(0,401,5),stride=5,dt=.005,scene_id=192+scene,
                                file=f'heldout-{192+scene}.npz'))
        curriculum=GatedStudentCurriculum(settings(),[9,17])
        curriculum.window_stage=1
        curriculum.schedule.stages.update(motion=9,imu=4,visual=0)
        store=SimpleNamespace(entries=entries)
        first=curriculum.probe_loader(store,'imu',1)
        second=curriculum.probe_loader(store,'imu',3)
        self.assertEqual(first.dataset.indices,second.dataset.indices)
        view=first.dataset.dataset
        self.assertEqual(view.base.length,17)
        self.assertEqual(view.stages,{'motion':9,'visual':0,'imu':4})
        self.assertEqual(view.mix,[1.,0.,0.])
        scores=motion_scores(view.base)
        cuts=torch.quantile(scores,torch.linspace(0,1,11,dtype=torch.float64)[1:-1])
        bins=torch.bucketize(scores,cuts,right=False)
        self.assertTrue(all(bins[index]==9 for index in first.dataset.indices))
        torch.testing.assert_close(first.dataset[0]['imu'],second.dataset[0]['imu'])

    def test_old_validation_history_is_reset_without_losing_curriculum_position(self):
        old=StagedStudentCurriculum(settings(),[2,3])
        old.window_stage=1;old.schedule.stages['motion']=4
        old.gate('fit','a').update(42.)
        old.paused={'a'};old.readiness={'motion':{'a':'plateau'}}
        current=GatedStudentCurriculum(settings(),[2,3])
        current.load_state_dict(old.state_dict(),preserve_progress=True,students=['a'])
        self.assertEqual(current.length,3)
        self.assertEqual(current.schedule.stages['motion'],4)
        self.assertEqual(current.student_gates,{})
        self.assertEqual(current.paused,set())
        self.assertTrue(current.validation_migration['plateau_counters_reset'])

    def test_new_policy_resume_preserves_budget_and_accuracy_history(self):
        previous=GatedStudentCurriculum(settings(),[2,3],max_stage_checks=12)
        for index in range(5):self.update(previous,index)
        current=GatedStudentCurriculum(settings(),[2,3])
        current.load_state_dict(copy.deepcopy(previous.state_dict()),preserve_progress=True,students=['a','b'])
        self.assertEqual(current.stage_checks,5)
        self.assertEqual(current.max_stage_checks,12)
        self.assertEqual(current.hybrid_history,previous.hybrid_history)
        self.assertIsNone(current.validation_migration)
        self.assertEqual(self.update(current,'new')['promoted'],['motion'])

    def test_old_stage_loss_policy_migrates(self):
        old=StagedStudentCurriculum(settings(),[2,3])
        state=old.state_dict();state['validation_policy']='current_horizon_motion_and_noise_state_loss_v1'
        current=GatedStudentCurriculum(settings(),[2,3])
        current.load_state_dict(state,preserve_progress=True,students=['a'])
        self.assertEqual(current.stage_checks,0)
        self.assertTrue(current.validation_migration['minimum_budget_restarts'])

    def test_physical_accuracy_ignores_initial_state_and_uses_vector_rmse(self):
        from gated_training import curriculum_squared_errors
        truth={key:torch.zeros(1,3,3) for key in ('pos','vel','omega')}
        truth['rot']=torch.eye(3).repeat(1,3,1,1)
        pred={key:value.clone() for key,value in truth.items()}
        pred['pos'][:,0]=10000.;pred['pos'][:,1:,0]=3.;pred['pos'][:,1:,1]=4.
        pred['rot'][:,1:]=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        errors=curriculum_squared_errors(pred,truth)
        self.assertAlmostEqual(errors['position_rmse_m'],25.)
        self.assertAlmostEqual(errors['orientation_rmse_deg'],8100.,places=2)
        self.assertEqual(errors['velocity_rmse_mps'],0.)


if __name__=='__main__':unittest.main()
