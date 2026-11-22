"""Register current PEAR job IDs in Cluv's cache without submitting any jobs."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from cluv.cache import Job, load_jobs, save_job
from cluv.remote import control_socket_is_running


ROOT = Path(__file__).resolve().parents[1]


def fetch(alias, settings):
    code = """
import json, os, pathlib, shlex, subprocess, datetime
settings = SETTINGS
monitor = json.loads(pathlib.Path(settings['monitor']).read_text())
job = str(monitor['current_job_id'])
is_build = str(monitor.get('build_job_id')) == job and monitor.get('build_submit_receipt')
receipt_path = monitor['build_submit_receipt'] if is_build else monitor['formal_submit_receipt']
receipt = json.loads(pathlib.Path(receipt_path).read_text())
command = receipt.get('new_submit_command', receipt.get('command'))
if receipt.get('native_sbatch_command'):
    job_script = pathlib.Path(receipt['job_script'])
    flags = receipt['sbatch_args']
    program_args = receipt['program_args']
    command_text = receipt['native_sbatch_command']
else:
    if not command:
        raise RuntimeError('No submission command in the authoritative receipt')
    scripts = [i for i, arg in enumerate(command) if arg.endswith('.sbatch')]
    if len(scripts) != 1:
        raise RuntimeError('Expected one Slurm launcher in the authoritative receipt')
    script_index = scripts[0]
    job_script = pathlib.Path(command[script_index])
    if not job_script.is_absolute():
        job_script = pathlib.Path(settings['root']) / job_script
    flags = {}
    for value in command[1:script_index]:
        if value.startswith('--'):
            key, sep, item = value[2:].partition('=')
            flags[key] = item if sep else True
    program_args = command[script_index + 1:]
    command_text = shlex.join(command)
env = os.environ.copy(); env['TZ'] = 'UTC'
query = subprocess.run(['scontrol', 'show', 'job', job, '-o'], text=True,
                       capture_output=True, env=env)
if query.returncode == 0:
    fields = dict(item.split('=', 1) for item in query.stdout.split() if '=' in item)
    submitted, state = fields['SubmitTime'], fields['JobState']
else:
    # Slurm removes finished jobs from scontrol; accounting retains their real state.
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
                                   '--format=JobIDRaw,Submit,State'], text=True, env=env)
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    matches = [row for row in rows if row[0] == job]
    if len(matches) != 1:
        raise RuntimeError('Job missing from both Slurm controller and accounting: ' + job)
    _, submitted, state = matches[0][:3]
print(json.dumps({
    'logical_cluster': ALIAS, 'current_job_id': int(job),
    'submitted_at': submitted + '+00:00',
    'git_commit': receipt['source_revision'],
    'job_script': str(job_script),
    'sbatch_args': flags, 'program_args': program_args,
    'sbatch_command': command_text, 'state': state,
    'current_run': monitor['current_run'],
    'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'import_origin': 'Read from authoritative PEAR monitor and real Slurm job; no submission'
}))
""".replace("SETTINGS", repr(settings)).replace("ALIAS", repr(alias))
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias,
               "python3 -c " + shlex.quote(code)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "SSH query failed")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, help="Initial verified Slurm metadata manifest")
    options = parser.parse_args()
    os.chdir(ROOT)
    settings = json.loads((ROOT / "clusters.json").read_text())
    records = []
    if options.seed:
        records = json.loads(options.seed.read_text())
    else:
        for alias, item in settings.items():
            if not asyncio.run(control_socket_is_running(alias)):
                print(alias + ": disconnected; use ./login " + alias, file=sys.stderr)
                continue
            try:
                records.append(fetch(alias, item))
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                print(alias + ": " + str(error), file=sys.stderr)

    known = {(job.cluster, job.job_id) for job in load_jobs()}
    state_dir = ROOT / ".state"
    state_dir.mkdir(mode=0o700, exist_ok=True)
    manifest = state_dir / "imported-jobs.jsonl"
    for record in records:
        alias = record["logical_cluster"]
        if alias not in settings:
            raise RuntimeError("Unknown cluster in import: " + alias)
        job_id = int(record["current_job_id"])
        if job_id <= 0:
            raise RuntimeError("Invalid job ID")
        launcher = Path(record["job_script"])
        formal = Path(settings[alias]["root"]) / "deploy/train.sbatch"
        control = Path(settings[alias]["root"]).parent / "pear_cluv_control/scripts/job.sh"
        build_dir = Path(settings[alias]["work"]) / "logs"
        if launcher not in (formal, control) and not (launcher.parent == build_dir and launcher.name.startswith("source-cache-validation-") and launcher.suffix == ".sbatch"):
            raise RuntimeError("Unexpected source launcher")
        identity = alias, job_id
        if identity not in known:
            save_job(Job(
                cluster=alias, remote=None, job_script=Path(record["job_script"]),
                sbatch_args=record["sbatch_args"], program_args=record["program_args"],
                sbatch_command=record["sbatch_command"], n_chunks=None,
                git_commit=record["git_commit"], job_id=job_id,
                submitted_at=datetime.fromisoformat(record["submitted_at"]),
            ))
            known.add(identity)
            record["registered_at"] = datetime.now(timezone.utc).isoformat()
            with manifest.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            manifest.chmod(0o600)
            print(alias + ": registered existing job " + str(job_id))
        else:
            print(alias + ": existing job " + str(job_id) + " already registered")


if __name__ == "__main__":
    main()
