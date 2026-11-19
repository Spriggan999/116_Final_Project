"""Resource-pressure exits, byte-capped recording, and small validation manifests."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
import legacy
import config
import gated_collect
from gated_data import GatedTrainingData
from gated_recovery import save_weights
from resource_limits import Limits, ResourceGuard, ResourceLimitError, quota_limits, disk_check, MIB
from safe_storage import SafeRollingBatches, scratch_environment


class ResourceTests(unittest.TestCase):
    def test_quota_parser_handles_starred_usage_and_small_home_quota(self):
        quotas=quota_limits('edge:/rpool/cc 14404441 20971520 20971520 234975 0 0\n'
                            'edge:/rpool/shadow 4095000* 4194304 4194304 0 0 0\n')
        self.assertEqual(quotas['edge:/rpool/cc'],(20971520-14404441)*1024)
        with patch('resource_limits.filesystem',return_value=('nfs4','edge:/rpool/shadow/user')), \
             patch('resource_limits.shutil.disk_usage',return_value=SimpleNamespace(free=10**15)):
            with self.assertRaisesRegex(ResourceLimitError,'quota'):
                disk_check('.',Limits(),quotas=quotas)

    def test_low_disk_rejected_before_write(self):
        with patch('resource_limits.shutil.disk_usage',return_value=SimpleNamespace(free=100*MIB)):
            with self.assertRaisesRegex(ResourceLimitError,'disk space'):
                disk_check('.',Limits())

    def test_ram_backed_root_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('resource_limits.filesystem',return_value=('tmpfs','tmpfs')):
            with self.assertRaisesRegex(ResourceLimitError,'RAM-backed'):
                with ResourceGuard(Path(directory)/'run',Path(directory)/'scratch'):
                    self.fail('must not start')

    def test_owned_storage_and_rss_budgets_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';scratch=root/'scratch'
            run.mkdir();scratch.mkdir()
            (scratch/'payload').write_bytes(bytes(2*MIB))
            guard=ResourceGuard(run,scratch,replace(Limits(),max_scratch_mib=1))
            guard.quota_checked=__import__('time').monotonic()
            with patch('resource_limits.disk_check'), \
                 patch('resource_limits.available_memory',return_value=100*1024*MIB), \
                 patch('resource_limits.process_tree_rss',return_value=0):
                with self.assertRaisesRegex(ResourceLimitError,'Storage budget'):
                    guard.check()
            with patch('resource_limits.disk_check'), \
                 patch('resource_limits.available_memory',return_value=100*1024*MIB), \
                 patch('resource_limits.process_tree_rss',return_value=100*1024*MIB):
                with self.assertRaisesRegex(ResourceLimitError,'RSS'):
                    guard.check()

    def test_watchdog_cancels_only_registered_work_and_interrupts_main(self):
        with tempfile.TemporaryDirectory() as directory:
            guard=ResourceGuard(Path(directory)/'run',Path(directory)/'scratch')
            cancelled=[]
            guard.cancellers.append(lambda:cancelled.append(True))
            with patch.object(guard.done,'wait',return_value=False), \
                 patch.object(guard,'check',side_effect=ResourceLimitError('low RAM')), \
                 patch('resource_limits.os.kill') as kill:
                guard._watch()
            self.assertEqual(cancelled,[True])
            kill.assert_called_once_with(os.getpid(),signal.SIGINT)
            self.assertIsInstance(guard.failure,ResourceLimitError)

    def test_interrupted_checkpoint_keeps_previous_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            path.write_bytes(b'previous checkpoint')
            def failed(value,target):
                target.write_bytes(b'incomplete')
                raise OSError('disk full')
            with patch('gated_recovery.torch.save',side_effect=failed):
                with self.assertRaises(OSError):save_weights({},path)
            self.assertEqual(path.read_bytes(),b'previous checkpoint')
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_streamed_npz_obeys_cap_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory,patch('gated_collect.disk_check'):
            path=Path(directory)/'flight.npz'
            arrays=dict(camera=np.random.default_rng(17).integers(0,256,(4,16,16,3),dtype=np.uint8),
                        imu=np.zeros((41,6),np.float32))
            self.assertFalse(gated_collect.write_episode(path,arrays,200))
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.npz.tmp').exists())
            self.assertTrue(gated_collect.write_episode(path,arrays,100000))
            with np.load(path) as loaded:
                for name,value in arrays.items():np.testing.assert_array_equal(loaded[name],value)

    def test_collector_stores_twenty_hz_without_losing_imu_samples(self):
        class Env:
            total_instances=1
            frame_stride=4
            dt=.005
            def reset(self):self.tick=0
            def render(self):pass
            def step(self):self.tick+=1
            def terminated(self):return [False]
            def frames(self):return np.zeros((4,4,3),np.uint8)+self.tick
            def observe_privileged(self):
                state=np.zeros(18,np.float32);state[3:12]=np.eye(3).ravel();return state
            def observe_imu(self):return np.array([0,0,9.81,0,0,0,123,456],np.float32)
        with patch('gated_collect.available_memory',return_value=100*1024*MIB):
            row=gated_collect.collect_round(Env(),41)[0]
        np.testing.assert_array_equal(row['camera_ticks'],[0,10,20,30,40])
        self.assertEqual(len(row['state']),41)
        self.assertEqual(int(row['frame_stride']),10)
        self.assertEqual(row['imu'].shape,(41,6))

    def test_validation_manifest_references_existing_data_without_copying(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir();out=root/'out';out.mkdir()
            for scene in (0,192):
                state=np.zeros((41,18),np.float32);state[:,3:12]=np.eye(3).ravel()
                imu=np.zeros((41,6),np.float32);imu[:,2]=9.81
                np.savez(data/f'rollout_s{scene}_e0.npz',scene_id=scene,camera=np.zeros((5,16,16,3),np.uint8),
                         imu=imu,state=state,dt=.005,frame_stride=10,camera_ticks=np.arange(0,41,10))
            settings=SimpleNamespace(**{name:value for name,value in vars(config).items() if name.isupper()})
            settings.DEVICE='cpu';settings.H=settings.W=16;settings.WINDOW=2
            settings.STREAM_DATA=False;settings.WORKDIR=data;settings.BATCH=1
            with ExitStack() as stack,patch('os.link',side_effect=AssertionError('no validation copies')):
                GatedTrainingData(SimpleNamespace(out=out),settings,stack)
            manifest=json.loads((out/'procthor_data/manifest.json').read_text())
            self.assertEqual(len(manifest),1)
            self.assertEqual(Path(manifest[0]['file']),data/'rollout_s192_e0.npz')
            self.assertEqual(list((out/'procthor_data').glob('*.npz')),[])

    def test_rolling_cleanup_preserves_one_validation_set_and_no_home_copy(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir()
            settings=SimpleNamespace(**{name:value for name,value in vars(config).items() if name.isupper()})
            settings.DEVICE='cpu';settings.H=settings.W=16;settings.WINDOW=2;settings.BATCH=1
            settings.STREAM_DATA=True;settings.STREAM_SCENES=1;settings.STREAM_EPISODES_PER_SCENE=1
            settings.STREAM_VALIDATION_SCENES=1;settings.STREAM_MAX_MIB=64;settings.COLLECT_STEPS=41
            settings.NUM_SCENES=192;settings.SCENE_OFFSET=0
            guard=SimpleNamespace(root=root,limits=Limits(),cancellers=[],check=lambda **kw:None)
            def producer(target,spec,stop):
                state=np.zeros((41,18),np.float32);state[:,3:12]=np.eye(3).ravel()
                imu=np.zeros((41,6),np.float32);imu[:,2]=9.81
                np.savez(target/'rollout_s0_e0.npz',scene_id=spec['scene_offset'],camera=np.zeros((5,16,16,3),np.uint8),
                         imu=imu,state=state,dt=.005,frame_stride=10,camera_ticks=np.arange(0,41,10))
                (target/'manifest.json').write_text(json.dumps([dict(scene_index=spec['scene_offset'])]))
            def rolling(*args,**kwargs):
                return SafeRollingBatches(*args,**kwargs,collect=producer)
            args=SimpleNamespace(out=out,stream_dir=root/'recordings',collector=None)
            with patch('resource_limits.ACTIVE',guard),patch('safe_storage.filesystem',return_value=('ext4','disk')), \
                 patch('gated_data.SafeRollingBatches',side_effect=rolling),ExitStack() as stack:
                data=GatedTrainingData(args,settings,stack)
                owned=data.rolling.owned
                manifest=json.loads((out/'procthor_data/manifest.json').read_text())
                retained=Path(manifest[0]['file'])
                self.assertTrue(retained.exists())
                self.assertFalse(owned in retained.parents)
            self.assertFalse(owned.exists())
            self.assertTrue(retained.exists())
            self.assertEqual(list((out/'procthor_data').glob('*.npz')),[])

    def test_build_and_download_caches_are_watched_and_environment_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);before=os.environ.get('TMPDIR')
            with scratch_environment(root):
                for name in ('TMPDIR','HYPERLUX_CACHE_DIR','CUDA_CACHE_PATH','CONTA_CACHE','TRITON_CACHE_DIR'):
                    self.assertTrue(root in Path(os.environ[name]).parents)
                self.assertLessEqual(int(os.environ['CMAKE_BUILD_PARALLEL_LEVEL']),2)
                self.assertLessEqual(int(os.environ['HYPERLUX_BUILD_JOBS']),2)
            self.assertEqual(os.environ.get('TMPDIR'),before)


if __name__=='__main__':unittest.main()
