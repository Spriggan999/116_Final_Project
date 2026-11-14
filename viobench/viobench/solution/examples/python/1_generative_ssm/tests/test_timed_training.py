"""Timed launcher termination, persistent checkpoints, and end-of-run mirrors."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


RUNNER='''
import sys
from pathlib import Path
from types import SimpleNamespace
from run_until_11 import run
args=SimpleNamespace(local_root=Path(sys.argv[2]),checkpoint_root=Path(sys.argv[3]),
    max_data_mib=6,seconds=None if sys.argv[4]=='None' else float(sys.argv[4]),stop_grace=.5)
sys.exit(run(args,trainer=Path(sys.argv[1])))
'''

TRAINER='''
import argparse,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out');a,other=p.parse_known_args()
root=Path(a.out)
(root/'state_student.pt').write_bytes(b'completed checkpoint')
(root/'state_student.pt.tmp').write_bytes(b'incomplete checkpoint')
(root/'.post-sl-unpublished').mkdir()
(root/'.post-sl-unpublished/state_student.pt').write_bytes(b'unpublished')
(root/'environment.json').write_text(__import__('json').dumps(dict(
    cache=os.environ['CONTA_CACHE'],search=os.environ['CONTA_SEARCH_PATHS'],
    flags=other)))
'''


def command(tmp_path,body,seconds=20,env=None):
    trainer=tmp_path/'trainer.py';trainer.write_text(TRAINER+'\n'+body)
    local=tmp_path/'local';primary=tmp_path/'persistent'
    command=[sys.executable,'-c',RUNNER,str(trainer),str(local),str(primary),str(seconds)]
    environment=dict(os.environ,PYTHONPATH=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
    if env:environment.update(env)
    return command,environment,local,primary


def outputs(local,primary):
    runs=list(primary.iterdir())
    assert len(runs)==1
    run=runs[0];mirror=local/'checkpoint-backups'/run.name
    return run,mirror,json.loads((run/'run_status.json').read_text())


def test_default_has_no_deadline_when_clock_passes_eleven(tmp_path):
    args,env,local,primary=command(tmp_path,'time.sleep(.3)',seconds=None)
    # Simulate launching before 11 and monitoring after 11 without waiting.
    clock='''
import run_until_11 as launcher
from datetime import datetime
class BeforeEleven(datetime):
    @classmethod
    def now(cls,tz=None):return datetime(2026,11,13,10,59,59,tzinfo=launcher.PACIFIC)
launcher.datetime=BeforeEleven
launcher.time.time=lambda:datetime(2026,11,13,11,1,tzinfo=launcher.PACIFIC).timestamp()
'''
    args[2]=args[2].replace('from run_until_11 import run','from run_until_11 import run\n'+clock)
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    run,mirror,status=outputs(local,primary)
    assert status['stop_reason'] is None and not status['forced_stop']
    assert 'No automatic time limit' in result.stdout and 'Stop at:' not in result.stdout
    assert (mirror/'state_student.pt').read_bytes()==b'completed checkpoint'


@pytest.mark.parametrize('exit_code',[0,7])
@pytest.mark.parametrize('seconds',[None,20])
def test_normal_and_failed_training_both_make_a_final_mirror(tmp_path,exit_code,seconds):
    args,env,local,primary=command(tmp_path,f'raise SystemExit({exit_code})',seconds=seconds)
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==exit_code,result.stdout+result.stderr
    run,mirror,status=outputs(local,primary)
    assert (run/'state_student.pt').read_bytes()==(mirror/'state_student.pt').read_bytes()
    assert not (mirror/'state_student.pt.tmp').exists()
    assert not (mirror/'.post-sl-unpublished').exists()
    assert status['checkpoint_files']==['state_student.pt']
    assert status['child_returncode']==exit_code
    assert 'BACKUP COMPLETE' in result.stdout
    recorded=json.loads((run/'environment.json').read_text())
    assert recorded['cache']==str(local/'conta') and recorded['search']==''
    assert '--skip-export' in recorded['flags']
    config_path=Path(recorded['flags'][recorded['flags'].index('--config')+1])
    assert config_path==HERE/'curriculum.json'
    config=json.loads(config_path.read_text())
    assert (config['FRAME_PAIR_BATCH'],config['BATCH'])==(64,8)
    assert (config['EPOCHS_FRAME'],config['EPOCHS_STUDENT'])==(400,600)


GRACEFUL='''
def stop(signum,frame):
    (root/'interrupt_checkpoint.pt').write_bytes(b'saved during interruption')
    raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
(root/'ready').touch()
print('READY',flush=True)
while True:time.sleep(.1)
'''


def test_deadline_waits_for_graceful_save_before_mirroring(tmp_path):
    args,env,local,primary=command(tmp_path,GRACEFUL,seconds=2)
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    run,mirror,status=outputs(local,primary)
    assert status['stop_reason']=='deadline' and not status['forced_stop']
    assert (mirror/'interrupt_checkpoint.pt').read_bytes()==b'saved during interruption'


def test_termination_signal_also_stops_trainer_and_mirrors(tmp_path):
    args,env,local,primary=command(tmp_path,GRACEFUL,seconds=None)
    process=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+4
        while time.monotonic()<deadline and not list(primary.glob('*/ready')):
            assert process.poll() is None
            time.sleep(.02)
        assert list(primary.glob('*/ready'))
        process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==143,stdout+stderr
        run,mirror,status=outputs(local,primary)
        assert status['stop_reason']=='SIGTERM'
        assert (mirror/'interrupt_checkpoint.pt').is_file()
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_unresponsive_trainer_is_killed_before_final_mirror(tmp_path):
    body='signal.signal(signal.SIGINT,signal.SIG_IGN)\nwhile True:time.sleep(.1)'
    args,env,local,primary=command(tmp_path,body,seconds=2)
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    run,mirror,status=outputs(local,primary)
    assert status['forced_stop'] and status['child_returncode']==-signal.SIGKILL
    assert (mirror/'state_student.pt').is_file()


def test_failed_backup_reports_error_without_deleting_persistent_weights(tmp_path):
    binary=tmp_path/'bin';binary.mkdir()
    rsync=binary/'rsync';rsync.write_text('#!/bin/sh\nexit 23\n');rsync.chmod(0o700)
    args,env,local,primary=command(tmp_path,'',env=dict(PATH=str(binary)+os.pathsep+os.environ['PATH']))
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==1
    run,mirror,status=outputs(local,primary)
    assert 'BACKUP FAILED' in result.stderr
    assert (run/'state_student.pt').read_bytes()==b'completed checkpoint'


def test_early_failure_without_any_model_is_not_reported_as_success(tmp_path):
    args,env,local,primary=command(tmp_path,'')
    trainer=Path(args[3]);trainer.write_text('raise SystemExit(0)\n')
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==1
    run,mirror,status=outputs(local,primary)
    assert status['checkpoint_files']==[]
    assert 'No model checkpoint' in result.stderr and (mirror/'train.log').is_file()


@pytest.mark.parametrize('skip_embedding',[False,True])
def test_launcher_forwards_custom_config_and_continues_sl_without_overwriting_source(tmp_path,skip_embedding):
    source=tmp_path/'source';source.mkdir()
    (source/'frame_student.pt').write_bytes(b'preserved embedding checkpoint')
    config=tmp_path/'candidate.json';config.write_text(json.dumps(dict(EMBEDDING_DIM=96)))
    args,env,local,primary=command(tmp_path,'',seconds=None)
    injected=f'args.config=Path({str(config)!r})\nargs.from_checkpoints=Path({str(source)!r})\n'
    injected+=f'args.skip_embedding_training={skip_embedding!r}\n'
    args[2]=args[2].replace('sys.exit(run(args,',injected+'sys.exit(run(args,')
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    run,mirror,status=outputs(local,primary)
    flags=json.loads((run/'environment.json').read_text())['flags']
    assert flags[flags.index('--config')+1]==str(config)
    assert flags[flags.index('--from-checkpoints')+1]==str(source) and '--sl' in flags
    assert ('--skip-embedding-training' in flags)==skip_embedding
    assert (source/'frame_student.pt').read_bytes()==b'preserved embedding checkpoint'
    assert (mirror/'state_student.pt').is_file()


def test_skip_embedding_launcher_requires_source_before_creating_run(tmp_path):
    args,env,local,primary=command(tmp_path,'',seconds=None)
    args[2]=args[2].replace('sys.exit(run(args,','args.skip_embedding_training=True\nsys.exit(run(args,')
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'requires --from-checkpoints with frame_student.pt' in result.stderr
    assert not primary.exists() and not local.exists()
