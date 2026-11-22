"""Submit one PEAR experiment to four DRAC queues and retain the first runner."""

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".state"
MANIFEST = STATE / "race.json"
TARGETS = ("tamia", "fir", "rorqual", "trillium-gpu")
SESSION = "pear-cluv-race"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def choose_winner(candidates):
    # A forecast StartTime on a PENDING job is not evidence that it started.
    eligible = [(alias, item) for alias, item in candidates.items()
                if item.get("state") in {"RUNNING", "COMPLETING", "COMPLETED"}]
    if not eligible:
        return None
    return min(eligible, key=lambda pair: (pair[1].get("start") or "9999", pair[0]))[0]


REMOTE = r'''
import datetime, hashlib, json, os, pathlib, subprocess, sys
action=sys.argv[1]; data=json.loads(sys.argv[2]); cfg=data['config']
mp=pathlib.Path(cfg['monitor']); m=json.loads(mp.read_text()); env=os.environ.copy(); env['TZ']='UTC'
def save(path,value):
 temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def profile():
 root=pathlib.Path(cfg['root'])
 return {'source_revision':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
         'source_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(root),'diff','--binary'])).hexdigest(),
         'launcher_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['deploy/train.sbatch','scripts/start_ray_node.sh','scripts/gpu_process_variables.sh']}}
def observe(job):
 rows=subprocess.check_output(['squeue','--me','-h','-o','%i|%j|%T|%S|%k'],text=True,env=env).splitlines()
 active=[dict(zip(['job_id','name','state','start','comment'],row.split('|',4))) for row in rows if row.strip()]
 own=[r for r in active if r['name'] in ['pear-stage1','cluv-pear-stage1']]
 found=next((r for r in active if r['job_id']==job),None)
 if found: state,start=found['state'],found['start']
 else:
  raw=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State%40,Start'],text=True,env=env)
  hits=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==job]
  if len(hits)!=1:raise RuntimeError('No authoritative Slurm state for '+job)
  state,start=hits[0][1].split()[0],hits[0][2]
 result={'job_id':job,'state':state,'start':start,'monitor_job_id':str(m['current_job_id']),'active_jobs':own,'run':str(pathlib.Path(cfg['work'])/'runs'/('train-'+job))}
 if data.get('audit'):
  raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,env=env)
  result['resources']=dict(t.split('=',1) for t in raw.split() if '=' in t)
  result.update(profile())
 return result
if action=='profile':
 result=profile()
elif action=='snapshot':
 result=observe(str(data.get('job_id',m['current_job_id'])))
else:
 job=str(data['job_id']); expected=str(data.get('expected_current',job))
 if str(m['current_job_id'])!=expected:raise RuntimeError('Current job changed; refuse stale mutation')
 if action=='register':
  receipt=pathlib.Path(cfg['work'])/'logs'/('cluv-race-'+data['race_id']+'-'+job+'-submit.json')
  if receipt.exists():
   assert str(json.loads(receipt.read_text())['job_id'])==job
  else:save(receipt,data['receipt'])
  m.update(current_job_id=job,formal_job_id=job,current_run=str(pathlib.Path(cfg['work'])/'runs'/('train-'+job)),phase='formal',formal_submit_receipt=str(receipt))
 elif action=='mark':
  m['race']=data['race']
  if data['race']['state'] in ['cancel_requested','lost']:m['phase']='race_cancel_pending' if data['race']['state']=='cancel_requested' else 'race_lost'
  elif data['race']['state']=='winner':m['phase']='formal'
 elif action in ['cancel','release']:
  if action=='cancel':
   assert m.get('race',{}).get('race_id')==data['race_id'] and m['race']['state']=='cancel_requested'
  status=observe(job)
  if action=='cancel' and status['state'] in ['PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING']:
   subprocess.run(['scancel',job],check=True)
  elif action=='release' and status['state']=='PENDING':
   raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,env=env)
   fields=dict(t.split('=',1) for t in raw.split() if '=' in t)
   if fields.get('Reason')=='JobHeldUser':subprocess.run(['scontrol','release',job],check=True)
   elif fields.get('Reason')=='JobHeldAdmin':raise RuntimeError('Candidate is held by an administrator')
  print(json.dumps(observe(job)));sys.exit(0)
 else:raise RuntimeError('Unknown remote action')
 m['last_checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 m.setdefault('history',[]).append({'at':m['last_checked_at'],'event':'cluv_race_'+action,'job_id':job,'race_id':data['race_id'],'race':m.get('race')})
 save(mp,m);result={'job_id':job,'updated':True}
print(json.dumps(result))
'''


def rpc(alias, action, **data):
    settings = json.loads((ROOT / "clusters.json").read_text())
    if alias not in TARGETS:
        raise ValueError("Cluster outside the four PEAR targets")
    data["config"] = settings[alias]
    command = "python3 -c " + shlex.quote(REMOTE) + " " + action + " " + shlex.quote(json.dumps(data))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias, command],
                            text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(alias + ": " + result.stderr.strip())
    return json.loads(result.stdout)


def check_resources(alias, item):
    f = item["resources"]
    cpus = 96 if alias == "trillium-gpu" else 64 if alias == "rorqual" else 48
    memory = "0" if alias == "trillium-gpu" else "450G"
    tres = dict(part.split("=", 1) for part in f["ReqTRES"].split(","))
    if not (f["NumNodes"] in {"8", "8-8"} and f["TimeLimit"] == "02:00:00"
            and f["CPUs/Task"] == str(cpus) and f["MinMemoryNode"] == memory
            and any(tres.get(key) == "32" for key in ("gres/gpu", "gres/gpu:h100"))):
        raise RuntimeError(alias + ": resource request does not match the authorized 32-GPU experiment")


def persist(state):
    state["updated_at"] = utc()
    atomic(MANIFEST, state)
    atomic(STATE / "races" / (state["race_id"] + ".json"), state)


def mark(state, alias, outcome):
    item = state["candidates"][alias]
    value = {"race_id": state["race_id"], "state": outcome, "controller_host": state["controller_host"],
             "controller_manifest": str(MANIFEST), "winner": state.get("winner"), "updated_at": utc()}
    return rpc(alias, "mark", job_id=item["job_id"], race_id=state["race_id"], race=value)


async def submit_new(state, snapshots):
    # Reuse Cluv's real sync, submission building, sbatch and job-cache APIs.
    # The explicit map excludes Mila itself and the Trillium CPU scheduler.
    from cluv.cache import save_job
    from cluv.cli.login import get_remote_without_2fa_prompt
    from cluv.cli.submit import get_submissions, submit_job
    from cluv.cli.sync import sync_common_part, sync_per_cluster_part

    remotes = dict(zip(TARGETS, await asyncio.gather(*(get_remote_without_2fa_prompt(a) for a in TARGETS))))
    if any(r is None for r in remotes.values()):
        raise RuntimeError("All four authenticated connections are required; use ./login --check")
    expected = json.loads((ROOT / "launcher_manifest.json").read_text())["files"]
    profiles = await asyncio.gather(*(asyncio.to_thread(rpc, a, "profile") for a in TARGETS))
    for alias, profile in zip(TARGETS, profiles):
        if profile["launcher_hashes"] != expected:
            raise RuntimeError(alias + ": deploy the shared launcher before allocating GPUs")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit the controller project before submission")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    state["phase"] = "syncing"; persist(state)
    await sync_common_part(remotes, sync_datasets=False)
    await asyncio.gather(*(sync_per_cluster_part(r, sync_datasets=False, sync_results=False) for r in remotes.values()))
    state["phase"] = "submitting"; persist(state)

    async def one(alias):
        options = await get_submissions(alias, remotes[alias], job_script=None,
                    sbatch_args=["--hold", "--comment=pear-race-" + state["race_id"]], program_args=[],
                    chunking=None, git_commit=head, vram=None)
        if len(options) != 1:
            raise RuntimeError("Exactly one account/configuration is required per cluster")
        log = STATE / "races" / (state["race_id"] + "-" + alias + ".submission.log")
        log.touch()
        job = await submit_job(options[0], log)
        save_job(job)
        item = {"job_id": str(job.job_id), "state": "PENDING", "run": "", "submission_log": str(log)}
        state["candidates"][alias] = item; persist(state)
        audited = await asyncio.to_thread(rpc, alias, "snapshot", job_id=str(job.job_id), audit=True)
        check_resources(alias, audited)
        if audited["launcher_hashes"] != expected:
            raise RuntimeError(alias + ": launcher changed during submission; keep the candidate held")
        cfg = json.loads((ROOT / "clusters.json").read_text())[alias]
        receipt = {"job_id": str(job.job_id), "race_id": state["race_id"], "submitted_at": utc(),
                   "job_script": str(Path(cfg["root"]).parent / "pear_cluv_control/scripts/job.sh"),
                   "sbatch_args": dict(job.sbatch_args), "program_args": list(job.program_args),
                   "native_sbatch_command": job.sbatch_command, "controller_revision": head,
                   "source_revision": audited["source_revision"], "source_diff_sha256": audited["source_diff_sha256"],
                   "resources": audited["resources"]}
        await asyncio.to_thread(rpc, alias, "register", job_id=str(job.job_id),
                    expected_current=snapshots[alias]["monitor_job_id"], race_id=state["race_id"], receipt=receipt)
        state["candidates"][alias].update(audited); persist(state)
    results = await asyncio.gather(*(one(a) for a in TARGETS), return_exceptions=True)
    errors = [str(r) for r in results if isinstance(r, BaseException)]
    if errors:
        state["phase"] = "submission_error"; state["errors"] = errors; persist(state)
        raise RuntimeError("Submission incomplete; keep known jobs held and inspect receipts before retrying: " + "; ".join(errors))
    # All four jobs exist, are recorded and audited before any can start.
    for alias in TARGETS:
        await asyncio.to_thread(mark, state, alias, "contender")
    releases = await asyncio.gather(*(asyncio.to_thread(rpc, a, "release", job_id=state["candidates"][a]["job_id"],
                                            race_id=state["race_id"]) for a in TARGETS), return_exceptions=True)
    state["release_pending"] = [a for a,r in zip(TARGETS,releases) if isinstance(r,BaseException)]
    state["release_errors"] = {a:str(r) for a,r in zip(TARGETS,releases) if isinstance(r,BaseException)}
    persist(state)


def watch(state):
    last_summary = None
    while True:
        if not state.get("winner"):
            for alias in list(state.get("release_pending", [])):
                try:
                    rpc(alias, "release", job_id=state["candidates"][alias]["job_id"], race_id=state["race_id"])
                    state["release_pending"].remove(alias)
                    state["release_errors"].pop(alias, None)
                except Exception as error:
                    state["release_errors"][alias] = str(error)
        snapshots = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            queries = {pool.submit(rpc, a, "snapshot", job_id=x["job_id"]): a
                       for a, x in state["candidates"].items()}
            for query in as_completed(queries):
                alias = queries[query]; item = state["candidates"][alias]
                try:
                    fresh = query.result()
                    if fresh["monitor_job_id"] != item["job_id"]:
                        raise RuntimeError("Authoritative current job changed; inspect before resuming this race")
                    snapshots[alias] = fresh
                    item.update(fresh); item.pop("query_error", None)
                except Exception as error:
                    item["query_error"] = str(error)
        if not state.get("winner"):
            winner = choose_winner(snapshots)
            if winner:
                state["winner"] = winner; state["winner_selected_at"] = utc()
                state["phase"] = "cancelling_losers"; persist(state)
        if state.get("winner"):
            if not state.get("winner_marked"):
                try:
                    mark(state, state["winner"], "winner"); state["winner_marked"] = True
                except Exception as error:
                    state["winner_mark_error"] = str(error)
            losers = [a for a in state["candidates"] if a != state["winner"]]
            for alias in losers:
                if alias not in snapshots:
                    continue
                item = snapshots[alias]
                try:
                    if item["state"] in ACTIVE:
                        # Record intentional cancellation before touching this exact job ID.
                        mark(state, alias, "cancel_requested")
                        rpc(alias, "cancel", job_id=item["job_id"], race_id=state["race_id"])
                    elif item["state"] in TERMINAL:
                        mark(state, alias, "lost")
                except Exception as error:
                    state["candidates"][alias]["query_error"] = str(error)
            if all(a in snapshots and snapshots[a]["state"] in TERMINAL for a in losers):
                state["phase"] = "winner_selected"; state["race_finished_at"] = utc(); persist(state)
                print(json.dumps({"winner": state["winner"], "job_id": state["candidates"][state["winner"]]["job_id"]}), flush=True)
                return
        elif len(snapshots) == 4 and all(x["state"] in TERMINAL for x in snapshots.values()):
            state["phase"] = "all_failed"; persist(state)
            raise RuntimeError("All four candidates ended without a running winner")
        summary = {a: (x.get("state"), x.get("query_error")) for a, x in state["candidates"].items()}
        if summary != last_summary:
            print(utc(), json.dumps(summary), flush=True); last_summary = summary
        persist(state)
        time.sleep(10)


def worker():
    STATE.mkdir(mode=0o700, exist_ok=True)
    with (STATE / "race-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
        if previous and previous.get("phase") in {"submitting", "submission_error", "syncing"}:
            raise RuntimeError("Interrupted submission: reconcile queue and receipts before starting another race")
        snapshots = {a: rpc(a, "snapshot") for a in TARGETS}
        for alias, item in snapshots.items():
            extras = [x for x in item["active_jobs"] if x["job_id"] != item["monitor_job_id"]]
            if extras:
                raise RuntimeError(alias + ": another PEAR training job exists: " + str(extras))
        if previous and any(item["state"] in ACTIVE for item in snapshots.values()) and previous.get("candidates"):
            if all(previous["candidates"].get(a, {}).get("job_id") == snapshots[a]["job_id"] for a in TARGETS):
                previous["controller_pid"] = os.getpid(); persist(previous); watch(previous); return
        race_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        state = {"race_id": race_id, "created_at": utc(), "controller_pid": os.getpid(),
                 "controller_host": os.uname().nodename, "phase": "preflight", "candidates": {}, "winner": None}
        if any(item["state"] in ACTIVE for item in snapshots.values()):
            # Adopt the current race candidates and preserve their queue positions.
            for alias, item in snapshots.items():
                if item["state"] in ACTIVE:
                    audited = rpc(alias, "snapshot", audit=True); check_resources(alias, audited)
                    item.update(audited)
            state["adopted_existing_jobs"] = True; state["candidates"] = snapshots; persist(state)
            for alias in TARGETS: mark(state, alias, "contender")
        else:
            state["adopted_existing_jobs"] = False; persist(state)
            asyncio.run(submit_new(state, snapshots))
        state["phase"] = "queueing"; persist(state); watch(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    os.chdir(ROOT); STATE.mkdir(mode=0o700, exist_ok=True)
    if options.status:
        print(MANIFEST.read_text() if MANIFEST.exists() else "No PEAR queue race has been started.")
        return
    if options.worker:
        try: worker()
        except Exception as error:
            if MANIFEST.exists():
                state = json.loads(MANIFEST.read_text()); state["controller_error"] = str(error); persist(state)
            raise
        return
    with (STATE / "race-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True).returncode == 0:
            print("Race controller already active. Use ./submit --status or tmux attach -t " + SESSION)
            return
        # Refuse partial submissions; retries must reconcile the saved IDs first.
        if MANIFEST.exists():
            state = json.loads(MANIFEST.read_text())
            if state.get("phase") in {"submitting", "submission_error", "syncing"}:
                raise SystemExit("Inspect .state/race.json and reconcile held jobs before retrying.")
        command = shlex.join([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"])
        command += " >> " + shlex.quote(str(STATE / "race.log")) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "-c", str(ROOT), command], check=True)
        print("Started four-cluster PEAR queue race in tmux " + SESSION + ". Use ./submit --status.")


if __name__ == "__main__":
    main()
