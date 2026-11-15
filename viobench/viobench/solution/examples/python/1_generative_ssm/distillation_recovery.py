"""Atomic curriculum/counter recovery metadata, including older-run migration."""
from pathlib import Path
import json


def records(path):
    if not path.is_file():return []
    lines=path.read_text().splitlines();result=[]
    for index,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
    return result


def next_collection_index(batch):
    name=Path(str(batch)).name
    return int(name[6:])+1 if name.startswith('batch-') and name[6:].isdigit() else 0


def load_recovery(directory,profiles):
    path=directory/'recovery_state.json'
    if path.is_file():
        state=json.loads(path.read_text())
        if set(profiles)-set(state['students']):raise ValueError('Recovery metadata is missing selected students')
        return state
    # Older runs already saved per-student gates and epoch metrics. Recover
    # these instead of treating their current stage as a brand-new stage.
    config=json.loads((directory/'distillation.json').read_text())
    curriculum_path=directory/'curriculum_state.json'
    curriculum=json.loads(curriculum_path.read_text()) if curriculum_path.is_file() else None
    events=records(directory/'curriculum_metrics.jsonl')
    promotions=[row for row in events if row['promoted'] or row['window_promoted']]
    last_promotion=promotions[-1] if promotions else None
    students={}
    for name in profiles:
        rows=records(directory/name/'metrics.jsonl')
        if not rows:raise ValueError(f'Cannot recover {name}: no completed epoch metrics')
        latest=rows[-1]
        grace=0
        if last_promotion is not None:
            grace=config.get('curriculum_settings',{}).get('CURRICULUM_EARLY_STOP_GRACE',3)
            previous=last_promotion['recording_batch']
            for row in rows:
                if row['epoch']<=last_promotion['epoch']:continue
                if row['recording_batch']!=previous:grace=max(0,grace-1)
                previous=row['recording_batch']
        students[name]=dict(active=latest['active'],best_validation_mean_ate_m=latest['best_validation_mean_ate_m'],
            completed_epoch=latest['epoch'],patience=dict(bad_cycles=latest['non_improving_data_cycles'],
                last_batch=latest['recording_batch'],grace_remaining=grace))
    performance=records(directory/'performance.jsonl')
    epoch=performance[-1]['epoch'] if performance else max(value['completed_epoch'] for value in students.values())
    batch=performance[-1]['recording_batch'] if performance else next(iter(students.values()))['patience']['last_batch']
    return dict(version=1,completed_epochs=epoch,curriculum=curriculum,students=students,
                collection_next_index=next_collection_index(batch),legacy_migration=True)


def save_recovery(directory,epoch,curriculum,students,stage_checkpoints,rolling):
    state=dict(version=1,completed_epochs=epoch,curriculum=curriculum.state_dict() if curriculum else None,
        curriculum_stage_index=stage_checkpoints.index,
        collection_next_index=next_collection_index(rolling.current.name) if rolling else 0,
        students={name:dict(active=row['active'],best_validation_mean_ate_m=row['best'],
                           completed_epoch=row.get('completed_epoch',epoch),patience=dict(vars(row['patience']))) for name,row in students.items()})
    target=directory/'recovery_state.json';temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    temporary.replace(target)
    return state
