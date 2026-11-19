import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
import legacy
from gated_models import GatedModel, PROFILES
from gated_data import GatedStateWindows, causal_gravity
from training_policy import apply_learning_rate, physical_selection_score, load_model_weights
from gated_curriculum import GatedStudentCurriculum
from test_curriculum_validation import settings
from test_gated import inputs


class PolicyTests(unittest.TestCase):
    def test_moving_start_supervision_and_causal_gravity(self):
        count=21
        state=torch.zeros(count,18);state[:,3:12]=torch.eye(3).flatten()
        time=torch.arange(count)*.005
        state[:,0]=2*time;state[:,12]=2
        imu=torch.zeros(count,6);imu[:,2]=9.81;imu[1:,:2]=5
        entry=dict(state=state,imu=imu,dt=.005,stride=10,cam_ticks=torch.tensor([0,10,20]),
                   cams=torch.zeros(3,3,8,8,dtype=torch.uint8))
        store=SimpleNamespace(entries=[entry],corrected_supervision=True)
        sample=GatedStateWindows(store,21,False,True)[0]
        torch.testing.assert_close(sample['vel'][:,0],torch.full((21,),2.))
        torch.testing.assert_close(sample['pos'][:,0],2*time)
        torch.testing.assert_close(sample['gravity'],torch.tensor([0.,0.,-9.81]))
        store.corrected_supervision=False
        sample=GatedStateWindows(store,21,False,True)[0]
        self.assertEqual(sample['vel'].abs().sum(),0)

    def test_causal_gravity_skips_tick_zero_placeholder_and_ignores_future(self):
        # Real recordings store an all-zero IMU row at tick 0.
        count=41
        state=torch.zeros(count,18);state[:,3:12]=torch.eye(3).flatten()
        imu=torch.zeros(count,6);imu[1:,2]=9.81;imu[1:20,0]=1.
        entry=dict(state=state,imu=imu,dt=.005,stride=10,cam_ticks=torch.tensor([0,10,20,30,40]),
                   cams=torch.zeros(5,3,8,8,dtype=torch.uint8))
        store=SimpleNamespace(entries=[entry],corrected_supervision=True)
        windows=GatedStateWindows(store,21,False,True)
        first=windows.window((0,0),21)['gravity']
        self.assertGreater(first.norm(),9.8)
        torch.testing.assert_close(first,causal_gravity(imu,1))
        later=windows.window((0,20),21)['gravity']
        expected=-9.81*imu[11:21,:3].mean(0)/imu[11:21,:3].mean(0).norm()
        torch.testing.assert_close(later,expected)
        imu[21:]=torch.randn(count-21,6)
        torch.testing.assert_close(windows.window((0,20),21)['gravity'],later)

    def test_curriculum_augmentation_keeps_each_policys_gravity(self):
        # AugmentedDataset re-estimates gravity from (noisy) IMU; it must use the
        # window's own estimator, not a fixed future-looking one.
        import config as cfg
        from curriculum import AugmentedDataset
        count=81
        state=torch.zeros(count,18);state[:,3:12]=torch.eye(3).flatten()
        imu=torch.zeros(count,6);imu[1:,2]=9.81;imu[50:,0]=5.
        entry=dict(state=state,imu=imu,dt=.005,stride=10,cam_ticks=torch.arange(0,81,10),
                   cams=torch.zeros(9,3,8,8,dtype=torch.uint8),file='synthetic',scene_id=0)
        store=SimpleNamespace(entries=[entry],corrected_supervision=True)
        windows=GatedStateWindows(store,21,False,True)
        windows.items=[(0,40)]
        for imu_level in (0,1):
            schedule=SimpleNamespace(levels=[0.,1.],mix=[1.,0.,0.],all_easier=False,
                                     stages=dict(motion=0,imu=imu_level,visual=0))
            view=AugmentedDataset(windows,cfg,schedule,123)
            noisy=view.flight_imu(0)
            torch.testing.assert_close(view[0]['gravity'],causal_gravity(noisy,40))
        torch.testing.assert_close(windows[0]['gravity'],torch.tensor([0.,0.,-9.81]))
        store.corrected_supervision=False
        legacy_gravity=AugmentedDataset(windows,cfg,SimpleNamespace(levels=[0.,1.],mix=[1.,0.,0.],all_easier=False,
                                stages=dict(motion=0,imu=0,visual=0)),123)[0]['gravity']
        old=-imu[41:51,:3].mean(0);old=9.81*old/old.norm()
        torch.testing.assert_close(legacy_gravity,old)

    def test_resume_rate_survives_optimizer_restore_and_schedule_boundaries(self):
        model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        rows={'a':dict(optimizer=optimizer)};saved=copy.deepcopy(optimizer.state_dict())
        for _ in range(3):
            optimizer.load_state_dict(saved);apply_learning_rate(rows,1e-4)
            self.assertEqual(optimizer.param_groups[0]['lr'],1e-4)
        schedule=[dict(horizon_seconds=.16,lr=3e-4),dict(horizon_seconds=10.24,lr=3e-5)]
        apply_learning_rate(rows,schedule=schedule,horizon=10.24)
        self.assertEqual(optimizer.param_groups[0]['lr'],3e-5)
        apply_learning_rate(rows,1e-5,schedule,20.48)
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-5)

    def test_budget_stop_and_fixed_horizon(self):
        for fixed in (False,True):
            c=GatedStudentCurriculum(settings(),[21],budget_action='stop',fixed_horizon=fixed)
            poor={k:10*v for k,v in c.current_accuracy_targets().items()}
            for i in range(18):event=c.update(str(i),{'motion':{'a':1.}},{'a':1.},{'a':poor})
            self.assertEqual(c.schedule.stages['motion'],0)
            self.assertEqual(event['stop_requested'],not fixed)
            self.assertFalse(event['forced_promotion'])
            self.assertGreater(physical_selection_score(poor,c.current_accuracy_targets()),1)
            resumed=GatedStudentCurriculum(settings(),[21],budget_action='stop',fixed_horizon=fixed)
            resumed.load_state_dict(c.state_dict(),preserve_progress=True,students=['a'])
            self.assertEqual(resumed.stage_checks,18)

    def test_explicit_architecture_transfer(self):
        old=GatedModel(PROFILES['baseline'])
        new=GatedModel(dict(PROFILES['baseline'],architecture='persistent_visual_v2'))
        with self.assertRaises(RuntimeError):load_model_weights(new,old.state_dict())
        load_model_weights(new,old.state_dict(),True)
        self.assertEqual(new.velocity_update.weight.abs().sum(),0)
        torch.testing.assert_close(old.gru.weight_ih_l0,new.gru.weight_ih_l0)

    def test_v2_streaming_parity_persistence_and_backward(self):
        for profile in ('baseline','combined'):
            torch.manual_seed(17)
            model=GatedModel(dict(PROFILES[profile],architecture='persistent_visual_v2')).eval()
            torch.nn.init.normal_(model.velocity_update.weight,std=.01)
            torch.nn.init.normal_(model.orientation_update.weight,std=.001)
            if model.innovation is not None:
                # Nonzero so the innovation path (which reads `previous`) is exercised.
                torch.nn.init.normal_(model.innovation[-1].weight,std=.05)
            batch=inputs()
            with torch.no_grad():
                pred=model(**batch);state=model.stream_init();values=[]
                for tick in range(21):
                    slots=(batch['ticks'][0]==tick).nonzero().flatten()
                    image=batch['images'][:,slots[0]] if len(slots) else None
                    value,state=model.stream_step(image,batch['imu'][:,tick],.005,batch['gravity'],state)
                    values.append(value)
                for key in ('pos','vel','rot','omega','corrections'):
                    torch.testing.assert_close(pred[key],torch.stack([v[key] for v in values],1),atol=3e-6,rtol=3e-5)
                changed={k:v.clone() for k,v in batch.items()};changed['images'][:,2]+=.1
                future=model(**changed)
                torch.testing.assert_close(pred['pos'][:,:20],future['pos'][:,:20])
                if model.auxiliary is not None:
                    # The JEPA head predicts a capture's target without seeing it.
                    torch.testing.assert_close(pred['auxiliary'][:,2],future['auxiliary'][:,2])
                changed={k:v.clone() for k,v in batch.items()};changed['images'][:,1]+=.2
                altered=model(**changed)
                self.assertGreater((pred['vel'][:,9:20]-altered['vel'][:,9:20]).abs().max(),0)
            model.train()
            result=model(**batch)
            loss=sum(result[k].square().mean() for k in ('pos','vel','rot','omega'))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
