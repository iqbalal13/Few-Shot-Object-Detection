import os, sys, subprocess, hashlib, json, time, shutil
from pathlib import Path

G3_OUTPUT = WORKSPACE/'outputs'/'stage1_coco80'
G3_OUTPUT.mkdir(parents=True, exist_ok=True)
G3_SOURCE_MANIFEST = WORKSPACE/'source_manifest.json'
G3_SIGNATURE = G3_OUTPUT/'source_run_signature.json'
G3_STATUS = G3_OUTPUT/'stage1_timebudget_status.json'
G3_CONSOLE_LOG = G3_OUTPUT/'stage1_console.log'

# Optional short end-to-end launcher test. Turn on only if needed; the pilot is
# isolated from the final checkpoint and intentionally SKIPS COCO-Val.
RUN_G3_PILOT = True  # mandatory 90-second batch-8 GPU test
G3_PILOT_SECONDS = 90

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def make_g3_signature():
    if not G3_SOURCE_MANIFEST.is_file():
        raise RuntimeError('G1 manifest missing; rerun Step 12.')
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=str(ROOT),
                            capture_output=True,text=True,check=True).stdout.strip()
    return {'protocol':'LOCKED-v2.1',
            'source_manifest_sha256':sha256_file(G3_SOURCE_MANIFEST),
            'official_checkpoint_sha256':sha256_file(CHECKPOINT_PATH),
            'source_repo_commit':commit,'seed':SEED,
            'batch_size':BATCH_SIZE_STAGE1,'dataset_file':'coco',
            'source_classes':80,'total_unique_images':8000,
            'backbone':'resnet101','freeze_backbone_at_layer':STAGE1_FREEZE_BACKBONE_AT_LAYER,
            'episode_size':5,'num_feature_levels':1,
            'time_budget_seconds':STAGE1_TIME_BUDGET_SECONDS}

def make_subprocess_env():
    environment = os.environ.copy()
    # `sys.path.insert` from Step 9 lives only in Colab kernel! A new Python
    # interpreter does NOT inherit that list. Pass explicit import search roots.
    environment['PYTHONPATH'] = os.pathsep.join(filter(None,[
        str(FRAMEWORK.resolve()),str(ROOT.resolve()),environment.get('PYTHONPATH','')]))
    environment['PYTHONUNBUFFERED'] = '1'
    return environment

def run_logged(command, output_log, environment):
    """Stream subprocess output to screen AND persist every line for diagnosis."""
    output_log = Path(output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    print('Launching:', ' '.join(map(str,command)),flush=True)
    from collections import deque
    last_lines = deque(maxlen=70)
    with output_log.open('w',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=str(ROOT),env=environment,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line,end='',flush=True)
                last_lines.append(line.rstrip())
        except KeyboardInterrupt:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            raise
        code = process.wait()
    if code != 0:
        print('\n[G3 ERROR] subprocess exit code:',code,
              '\nCombined traceback log:',output_log,
              '\nLast output lines:\n'+'\n'.join(last_lines),flush=True)
        raise RuntimeError(f'G3 subprocess exited {code}. Read {output_log} and stage1_timebudget_status.json.')
    print('Saved combined training log:',output_log,flush=True)

def guard_prior_run(signature):
    if G3_SIGNATURE.exists():
        if json.loads(G3_SIGNATURE.read_text()) != signature:
            raise RuntimeError('G3 output belongs to a DIFFERENT dataset or checkpoint/config. '
                               'Choose a new WORKSPACE; do not overwrite provenance.')
    else:
        if (G3_OUTPUT/'checkpoint.pth').exists():
            raise RuntimeError('Untracked checkpoint.pth exists: never overwrite.')
        G3_SIGNATURE.write_text(json.dumps(signature,indent=2))

    if (G3_OUTPUT/'checkpoint.pth').exists():
        if not G3_STATUS.exists():
            raise RuntimeError('Checkpoint without G3 status; preserve and inspect, no overwrites.')
        status = json.loads(G3_STATUS.read_text())
        if status.get('status') != 'finished_with_coco_val':
            raise RuntimeError('Checkpoint exists but not a validated finished run. Protect it; '
                               'choose another output folder to retry.')
        return True

    # If an earlier crash was truly before the first optimizer step, archive
    # its console/status and retry. If progress or a partial snapshot exists,
    # stop rather than discarding evidence or unknowingly restarting training.
    if G3_STATUS.is_file():
        prior = json.loads(G3_STATUS.read_text())
        if int(prior.get('steps_completed',0)) > 0 or (G3_OUTPUT/'partial_latest.pth').exists():
            raise RuntimeError('Previous G3 reached optimizer steps / partial snapshot, but '
                               'checkpoint is incomplete. Preserve logs and partial model; '
                               'do not silently restart or treat as resumable.')
        archive = G3_OUTPUT/'failed_attempts'/f'prestep_failure_{int(time.time())}'
        archive.mkdir(parents=True,exist_ok=True)
        for name in ('stage1_timebudget_status.json','stage1_console.log','train_steps_metrics.jsonl'):
            old = G3_OUTPUT/name
            if old.exists(): shutil.move(str(old),str(archive/name))
        print('Archived previous pre-training failure at:',archive)
    elif G3_CONSOLE_LOG.exists():
        archive = G3_OUTPUT/'failed_attempts'/f'launcher_failure_{int(time.time())}'
        archive.mkdir(parents=True,exist_ok=True)
        shutil.move(str(G3_CONSOLE_LOG),str(archive/'stage1_console.log'))
    return False

if RUN_STAGE1_TRAIN:
    assert STAGE1_FREEZE_BACKBONE_AT_LAYER == 3, 'Locked freeze policy is 3'
    report=json.loads(G2_REPORT.read_text()) if G2_REPORT.is_file() else {}
    assert report.get('passed') is True, 'G2 full detector test must PASS.'
    assert report.get('freeze_backbone_at_layer') == 3, 'G2 freeze audit missing/mismatched; rerun G2.'
    assert report.get('backbone_freeze_report',{}).get('layer3',{}).get('trainable_params') == 0, 'G2 Layer3 not frozen.'
    assert report.get('source_manifest_sha256') == sha256_file(G3_SOURCE_MANIFEST), (
         'G2 smoke test ran on another source manifest! Re-run G2 for LOCKED v2.1.')
    assert (ROOT/'data/coco/annotations/instances_train2017.json').is_file(), 'G1 COCO-80 data not ready'

    environment = make_subprocess_env()
    # Cheap fail-fast preflight (never trains and never needs COCO GPU compute).
    import_check = subprocess.run([sys.executable,'-m','metadetr_pipeline.timebudget','--help'],
        cwd=str(ROOT),env=environment,capture_output=True,text=True)
    if import_check.returncode != 0:
        print(import_check.stdout,import_check.stderr)
        raise RuntimeError('G3 preflight failed: module not importable in child process. '
                           'Review PYTHONPATH and Step 9 overlay.')
    print('G3 subprocess import PREFLIGHT PASS. Overlay:',FRAMEWORK)

    signature=make_g3_signature()
    done=guard_prior_run(signature)
    if done:
        print('Found prior completed G3 with COCO-Val; no retraining.')
    else:
        if RUN_G3_PILOT:
            pilot_dir = WORKSPACE/'outputs'/'stage1_pilot_90s_no_val'
            pilot_cmd=[sys.executable,'-u','-m','metadetr_pipeline.timebudget',
                       '--repo',str(ROOT),'--checkpoint',str(CHECKPOINT_PATH),
                       '--output',str(pilot_dir),'--budget-seconds',str(G3_PILOT_SECONDS),
                       '--seed',str(SEED),'--batch-size',str(BATCH_SIZE_STAGE1),
                       '--workers','2','--freeze-backbone-at-layer',str(STAGE1_FREEZE_BACKBONE_AT_LAYER),
                       '--skip-evaluation']
            if pilot_dir.exists() and any(pilot_dir.iterdir()):
                raise RuntimeError('Pilot directory already has files: protect previous pilot, '
                                   'remove/move it manually if appropriate.')
            run_logged(pilot_cmd,pilot_dir/'stage1_pilot_console.log',environment)
            print('Pilot is NOT an evaluated G3 checkpoint. Final training now starts from official initializer.')

        cmd=[sys.executable,'-u','-m','metadetr_pipeline.timebudget',
             '--repo',str(ROOT),'--checkpoint',str(CHECKPOINT_PATH),
             '--output',str(G3_OUTPUT),
             '--budget-seconds',str(STAGE1_TIME_BUDGET_SECONDS),
             '--seed',str(SEED),'--batch-size',str(BATCH_SIZE_STAGE1),
             '--workers','2','--chunk-size','50',
             '--freeze-backbone-at-layer',str(STAGE1_FREEZE_BACKBONE_AT_LAYER),
             '--snapshot-interval-seconds','300']
        run_logged(cmd,G3_CONSOLE_LOG,environment)

    assert (G3_OUTPUT/'checkpoint.pth').is_file(), 'COCO-80 checkpoint not produced.'
    assert G3_STATUS.is_file(), 'Time-budget status JSON missing.'
    status=json.loads(G3_STATUS.read_text())
    print('G3 time-budget summary:',json.dumps(status,indent=2))
    assert status.get('status')=='finished_with_coco_val', 'G3 did not complete official COCO-Val.'
    assert status.get('steps_completed',0)>0,'No completed optimizer steps.'
    assert status.get('freeze_backbone_at_layer') == 3, 'Stage1 freeze status mismatch.'
    assert status.get('epoch_complete') is True and int(status.get('steps_completed', 0)) == EXPECTED_STAGE1_STEPS, (
        'FULL-EPOCH gate failed: expected all 1,000 optimizer steps for batch size 8. '
        'Inspect stage1_timebudget_status.json; preserve this run and do not silently '
        'approve a partial-epoch checkpoint.'
    )
    print(f'FULL EPOCH PASS: exactly {EXPECTED_STAGE1_STEPS}/{EXPECTED_STAGE1_STEPS} optimizer steps completed (8,000 images, batch size {BATCH_SIZE_STAGE1}).')
    print('COCO-Val results still require human review. CCTV remains gated by SOURCE_CHECKPOINT_APPROVED.')
else:
    print('G3 disabled. Source sampling/architecture unchanged.')
