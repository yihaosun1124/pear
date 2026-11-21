#!/usr/bin/env bash
#SBATCH --job-name=pear-stage1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --mem=450G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/pear_work/logs/%x-%j.out
#SBATCH --error=/scratch/%u/pear_work/logs/%x-%j.err

# Shared orchestration for Stage 1 and the one-step smoke test. See deploy/README.md.
# Single-quoted commands expand variables in the selected compute node's shell.
# shellcheck disable=SC2016
set -euo pipefail
umask 077
# DRAC's sbatch wrapper may use --export=NONE. Subsequent srun steps need this environment.
export SLURM_EXPORT_ENV=ALL
MODE=train
CONFIG_ONLY=0
ROOT=${PEAR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ "${1:-}" = --smoke ]; then
  MODE=smoke
  shift
fi
if [ "$MODE" = smoke ]; then
  export STEPS=1 EPOCHS=1 BATCH=4 N=2 MAX_PROMPT=4096 MAX_RESPONSE=4096
  export TRAINER_NNODES=1 ROLLOUT_NNODES=1
  export SESSIONS_PER_REPLICA=8 SPECULATIVE_TOKENS=0 LIBERO_SERVERS_PER_NODE=4 OBJECT_STORE_GB=8
  export LOGGER='["console","file"]'
else
  # Keep official training hyperparameters; fit the host prefix cache to the deployment budget.
  export STEPS=1000 EPOCHS=100 BATCH=128 N=8 MAX_PROMPT=4096 MAX_RESPONSE=160000
  export TRAINER_NNODES=4 ROLLOUT_NNODES=12
  export SESSIONS_PER_REPLICA=64 SPECULATIVE_TOKENS=3 LIBERO_SERVERS_PER_NODE=32 OBJECT_STORE_GB=64
  export LOGGER='["console","file","tensorboard"]'
fi
export LMCACHE_GB=64
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config-only) CONFIG_ONLY=1 ;;
    --help)
      echo 'Submit from the repo root: sbatch --account=... --partition=... deploy/train.sbatch [NAME=value ...]'
      echo 'Names: PEAR_ROOT WORK_DIR RUN MODEL EPISODES BATCH N STEPS EPOCHS MAX_PROMPT MAX_RESPONSE'
      echo '       TRAINER_NNODES ROLLOUT_NNODES SESSIONS_PER_REPLICA SPECULATIVE_TOKENS LIBERO_SERVERS_PER_NODE LOGGER'
      echo '       LMCACHE_GB (host prefix cache per TP4 rollout replica; default 64 GiB)'
      echo 'Use --config-only to compose the config without an allocation; -- passes remaining Hydra overrides.'
      exit 0
      ;;
    PEAR_ROOT=*) ROOT=${1#*=} ;;
    WORK_DIR=* | RUN=* | MODEL=* | EPISODES=* | BATCH=* | N=* | STEPS=* | EPOCHS=* | MAX_PROMPT=* | MAX_RESPONSE=* | \
      TRAINER_NNODES=* | ROLLOUT_NNODES=* | SESSIONS_PER_REPLICA=* | SPECULATIVE_TOKENS=* | \
      LIBERO_SERVERS_PER_NODE=* | LOGGER=* | LMCACHE_GB=*) export "${1?}" ;;
    --)
      shift
      break
      ;;
    *) echo "Unknown argument: $1 (Hydra overrides go after --)" >&2; exit 2 ;;
  esac
  shift
done
cd "$ROOT"
ROOT=$PWD
export WORK_DIR=${WORK_DIR:-/scratch/$USER/pear_work}
export GPU=h100
for name in STEPS EPOCHS BATCH N MAX_PROMPT MAX_RESPONSE SESSIONS_PER_REPLICA LIBERO_SERVERS_PER_NODE \
  TRAINER_NNODES ROLLOUT_NNODES LMCACHE_GB; do
  [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
[[ $SPECULATIVE_TOKENS =~ ^[0-9]+$ ]] || { echo 'SPECULATIVE_TOKENS must be nonnegative' >&2; exit 2; }
((N >= 2 && MAX_PROMPT + MAX_RESPONSE <= 262144)) || { echo 'Invalid GRPO N or context budget' >&2; exit 2; }
if [ "$MODE" = smoke ] && [ "$STEPS" != 1 ]; then
  echo 'The smoke test requires STEPS=1' >&2
  exit 2
fi
TOTAL_NODES=$((TRAINER_NNODES + ROLLOUT_NNODES))
if [ "$MODE" = smoke ]; then
  if [[ "$TRAINER_NNODES" != 1 || "$ROLLOUT_NNODES" != 1 ]]; then
    echo 'Smoke needs a 1+1 node split' >&2
    exit 2
  fi
else
  ((TRAINER_NNODES * 4 % 8 == 0)) || { echo 'Official H100 SP8 needs trainer GPUs divisible by 8' >&2; exit 2; }
fi
RECIPE=$ROOT/examples/models/qwen3_8_27b_pear/train/run_stage1.sh
test -f "$RECIPE"
test -x "$WORK_DIR/envs/verl/bin/python"

if [ "$CONFIG_ONLY" = 1 ]; then
  export RUN=${RUN:-$WORK_DIR/cache/config-check-$MODE}
  export LOCAL_DIR=$RUN/local
else
  : "${SLURM_JOB_ID:?Submit with sbatch, or use --config-only}"
  [ "${SLURM_JOB_NUM_NODES:?}" = "$TOTAL_NODES" ] || {
    echo "Need $TRAINER_NNODES trainer + $ROLLOUT_NNODES rollout nodes; allocation has $SLURM_JOB_NUM_NODES" >&2
    exit 2
  }
  export RUN=${RUN:-$WORK_DIR/runs/$MODE-$SLURM_JOB_ID}
  if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && [ "$MODE" = smoke ]; then
    export RUN=$RUN-r$SLURM_RESTART_COUNT
  fi
  # Some DRAC GPU nodes mount /tmp as tmpfs: JIT and Ray files would consume the job's RAM budget.
  LOCAL_PARENT=${SLURM_TMPDIR:-}
  if [ -n "$LOCAL_PARENT" ] && [ "$(stat -f -c %T "$LOCAL_PARENT")" = tmpfs ]; then
    LOCAL_PARENT=
  fi
  if [ -z "$LOCAL_PARENT" ]; then
    if [ -d /localscratch ]; then
      LOCAL_PARENT=/localscratch/$USER
    else
      LOCAL_PARENT=/tmp/$USER
      [ "$(stat -f -c %T /tmp)" != tmpfs ] || {
        echo 'Need a disk-backed SLURM_TMPDIR or /localscratch for node-local caches' >&2
        exit 2
      }
    fi
  fi
  export LOCAL_DIR=$LOCAL_PARENT/pear-$SLURM_JOB_ID
fi
if [ "$MODE" = smoke ]; then
  # A fresh smoke run must not pass by reusing a previous checkpoint.
  test ! -e "$RUN/ckpt" || { echo "Smoke RUN already has checkpoints: $RUN" >&2; exit 2; }
fi
mkdir -p "$RUN" "$LOCAL_DIR"
# Initialize the batch shell too: token generation and cleanup run outside the
# srun workers and must not inherit a submitting CPU job's node-local caches.
source "$ROOT/scripts/gpu_process_variables.sh" "$WORK_DIR/envs/verl"
export CCACHE_DIR=$WORK_DIR/cache/ccache XDG_CACHE_HOME=$LOCAL_DIR/xdg-cache
export RAY_TMPDIR=$LOCAL_DIR/ray HF_HOME=$WORK_DIR/cache/hf HF_HUB_OFFLINE=1
export NCCL_LAUNCH_ORDER_IMPLICIT=1 NCCL_MNNVL_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

EXTRA_ARGS=(
  trainer.n_gpus_per_node=4
  actor_rollout_ref.rollout.n_gpus_per_node=4
  trainer.max_actor_ckpt_to_keep=1
)
if [ "$MODE" = smoke ]; then
  cat > "$RUN/episodes.json" <<'JSON'
{"env_class":"pear.envs.libero.env.LiberoEnv","data_source":"suite",
 "env":{"suite":["libero_spatial"],"task_id":"0","trial_id":"0-3"}}
JSON
  export EPISODES=$RUN/episodes.json
  # Keep the exact token cap of the verified one-step run.
  EXTRA_ARGS+=(
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=4
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.custom.lmcache.enable=false
    actor_rollout_ref.rollout.custom.lmcache.l1_size_gb=0
    actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.enforce_eager=true
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
    algorithm.filter_groups.enable=false trainer.resume_mode=disable
  )
else
  EXTRA_ARGS+=(actor_rollout_ref.rollout.custom.lmcache.l1_size_gb="$LMCACHE_GB")
fi
if [ "$CONFIG_ONLY" = 1 ]; then
  exec bash "$RECIPE" "${EXTRA_ARGS[@]}" "$@" --cfg job --resolve
fi

exec > >(tee -a "$RUN/allocation-$SLURM_JOB_ID.log") 2>&1
date -u
echo "mode=$MODE job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST commit=$(git rev-parse HEAD)"
git diff --no-ext-diff --binary > "$RUN/source-$SLURM_JOB_ID.patch"
mkdir -p "$RUN/launch-$SLURM_JOB_ID"
cp deploy/{smoke_test.sbatch,train.sbatch} "$RUN/launch-$SLURM_JOB_ID/"

# A successful CPU validation may seed bytecode for this exact environment and checkout.
# The archive stays within its Slurm cluster; native libraries are only read to warm the local page cache.
if [ -f "$WORK_DIR/cache/python-bytecode-ready.json" ]; then
  srun --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c '/usr/bin/python3 -B -S - "$@" << "PY_NODE_CACHE"
import hashlib, json, os, pathlib, subprocess, sys, tarfile, time
root, work, local = map(pathlib.Path, sys.argv[1:])
meta = json.loads((work / "cache/python-bytecode-ready.json").read_text())
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], universal_newlines=True).strip()
env = (work / "envs/verl").resolve()
cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
if not cluster or meta.get("cluster") != cluster or meta.get("git_head") != head or meta.get("python_env") != str(env):
    print(json.dumps({"node": os.uname().nodename, "cache_seed": "skipped", "reason": "cluster, environment or checkout differs"}), flush=True)
    sys.exit(0)
archive = pathlib.Path(meta["archive"]).resolve()
if os.path.commonpath([str(archive), str((work / "cache").resolve())]) != str((work / "cache").resolve()):
    raise RuntimeError("Bytecode archive must be in this work directory cache")
started = time.monotonic()
digest = hashlib.sha256()
with archive.open("rb") as stream:
    for data in iter(lambda: stream.read(8 * 1024**2), b""):
        digest.update(data)
if digest.hexdigest() != meta["sha256"]:
    raise RuntimeError("Bytecode archive checksum mismatch")
target = local / "python-bytecode"
target.mkdir(parents=True, exist_ok=True)
with tarfile.open(str(archive), "r:gz") as bundle:
    members = bundle.getmembers()
    for member in members:
        name = pathlib.PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()) or (member.isfile() and not member.name.endswith(".pyc")):
            raise RuntimeError("Invalid bytecode archive member")
    bundle.extractall(str(target), members=members)
native_bytes = 0
native_count = 0
buffer = bytearray(8 * 1024**2)
for name in meta.get("native_libraries", []):
    if not os.path.isabs(name):
        raise RuntimeError("Native warm-up path must be absolute")
    library = pathlib.Path(os.path.normpath(name))
    if os.path.commonpath([str(library), str(env)]) != str(env):
        raise RuntimeError("Native warm-up path is outside this environment")
    try:
        stream = library.open("rb")
    except FileNotFoundError:
        continue
    with stream:
        while True:
            count = stream.readinto(buffer)
            if not count:
                break
            native_bytes += count
    native_count += 1
print(json.dumps({"node": os.uname().nodename, "cache_seed": "loaded", "archive_sha256": digest.hexdigest(), "native_libraries": native_count, "native_bytes": native_bytes, "seconds": time.monotonic() - started}), flush=True)
PY_NODE_CACHE' pear-node-cache "$ROOT" "$WORK_DIR" "$LOCAL_DIR" \
    > "$RUN/python-cache-stage-$SLURM_JOB_ID.log" 2>&1
fi
if [ -f "$WORK_DIR/cache/transformers-source-ready.json" ]; then
  srun --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
    /usr/bin/python3 -B -S pear/utils/sitecustomize/pear_source_cache.py stage "$ROOT" "$WORK_DIR" "$LOCAL_DIR" \
    > "$RUN/python-source-stage-$SLURM_JOB_ID.log" 2>&1
fi
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD=${NODES[0]}
WORKER=${NODES[TRAINER_NNODES]}
export RAY_ADDRESS=$HEAD:6379 PORT=6379
export NUM_CPUS=$((SLURM_CPUS_PER_TASK - 8))
((NUM_CPUS >= LIBERO_SERVERS_PER_NODE + 4)) || { echo 'Too few CPU slots' >&2; exit 2; }
export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$RUN/ray-token
"$WORK_DIR/envs/verl/bin/python" -c \
  'import secrets,sys; open(sys.argv[1],"w").write(secrets.token_hex(32))' "$RAY_AUTH_TOKEN_PATH"

cleanup() {
  code=$?
  trap - EXIT
  if [ -n "${TRAIN_PREFLIGHT_PID:-}" ]; then
    kill "$TRAIN_PREFLIGHT_PID" 2>/dev/null || true
    wait "$TRAIN_PREFLIGHT_PID" 2>/dev/null || true
  fi
  srun --overlap --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'if [ -d "$LOCAL_DIR/ray/session_latest/logs" ]; then
      tar -czf "$RUN/ray-logs-$SLURM_JOB_ID-$(hostname).tar.gz" -C "$LOCAL_DIR/ray/session_latest" logs; fi' \
    > "$RUN/archive-ray-$SLURM_JOB_ID.log" 2>&1 || true
  srun --overlap --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
    "$WORK_DIR/envs/verl/bin/ray" stop --force > "$RUN/ray-stop-$SLURM_JOB_ID.log" 2>&1 || true
  for pid in "${HEAD_PID:-}" "${NODE_PIDS[@]}"; do
    if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
  done
  srun --overlap --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'if [[ "$LOCAL_DIR" == /tmp/"$USER"/pear-"$SLURM_JOB_ID" ]]; then rm -rf -- "$LOCAL_DIR"; fi' \
    > "$RUN/cleanup-local-$SLURM_JOB_ID.log" 2>&1 || true
  printf '{"slurm_job_id":"%s","exit_code":%s}\n' "$SLURM_JOB_ID" "$code" > "$RUN/job-result-$SLURM_JOB_ID.json"
  exit "$code"
}
NODE_PIDS=()
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun --overlap --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
  bash -c 'hostname; nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader; ip -br address' \
  > "$RUN/hardware-$SLURM_JOB_ID.log"

if [ "$MODE" = smoke ]; then
  # Run the regressions on their respective nodes before paying for the model load.
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c 'mkdir -p "$LOCAL_DIR"; source scripts/gpu_process_variables.sh "$WORK_DIR/envs/verl"
      test=tests/models/test_qwen3_5_training_kernels_on_gpu.py
      python -u -m pytest "$test::test_the_hopper_delta_rule_backward_reuses_the_recompute_s_tensors"' \
    > "$RUN/training-preflight.log" 2>&1 &
  TRAIN_PREFLIGHT_PID=$!
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$WORKER" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c 'mkdir -p "$LOCAL_DIR"; source scripts/gpu_process_variables.sh "$WORK_DIR/envs/verl"; python -u -' \
    > "$RUN/flashinfer-preflight.log" 2>&1 <<'PY'
import json
import os
import pathlib
import sys
import flashinfer
import torch
from torch.nn import functional as F

torch.manual_seed(42)
x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
out = flashinfer.silu_and_mul(x)
reference = (F.silu(x[:, :512].float()) * x[:, 512:].float()).to(x.dtype)
torch.cuda.synchronize()
torch.testing.assert_close(out, reference, rtol=8e-3, atol=8e-3)
assert torch.isfinite(out).all()
print(json.dumps({"flashinfer_silu_and_mul": "passed", "device": torch.cuda.get_device_name(),
                  "max_abs_error": (out.float() - reference.float()).abs().max().item()}))
# Keep the imported CUDA/Python stack for the communication test. A fresh
# interpreter otherwise repeats thousands of remote source-file stat calls.
sys.stdout.flush()
sys.stderr.flush()
with (pathlib.Path(os.environ["RUN"]) / "allreduce-preflight.log").open("w") as stream:
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
import runpy

test = "tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py"
# Use the real module entry point so multiprocessing.spawn does not try to
# re-execute the stdin script as a nonexistent <stdin> file in each rank.
sys.argv = ["pytest", test + "::test_the_fused_all_reduces_on_four_ranks[False]"]
runpy.run_module("pytest", run_name="__main__", alter_sys=True)
PY
  wait "$TRAIN_PREFLIGHT_PID"
  TRAIN_PREFLIGHT_PID=
fi
srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash scripts/start_ray_node.sh trainer > "$RUN/ray-head-$SLURM_JOB_ID.log" 2>&1 &
HEAD_PID=$!
HEAD_READY=0
for ((attempt = 0; attempt < 90; attempt++)); do
  if "$WORK_DIR/envs/verl/bin/python" -c \
    'import socket,os; h,p=os.environ["RAY_ADDRESS"].split(":"); socket.create_connection((h,int(p)),2).close()' \
    >/dev/null 2>&1; then HEAD_READY=1; break; fi
  kill -0 "$HEAD_PID"
  sleep 2
done
[ "$HEAD_READY" = 1 ] || { echo 'Ray head startup timed out' >&2; exit 1; }
for ((index = 1; index < TOTAL_NODES; index++)); do
  node=${NODES[index]}
  role=rollout
  if ((index < TRAINER_NNODES)); then role=trainer; fi
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash scripts/start_ray_node.sh "$role" "$RAY_ADDRESS" > "$RUN/ray-$role-$node-$SLURM_JOB_ID.log" 2>&1 &
  NODE_PIDS+=("$!")
done
"$WORK_DIR/envs/verl/bin/python" - > "$RUN/cluster-$SLURM_JOB_ID.json" <<'PY'
import json
import os
import time
import ray

ray.init(address=os.environ["RAY_ADDRESS"])
deadline = time.monotonic() + 240
trainer_nodes = int(os.environ["TRAINER_NNODES"])
rollout_nodes = int(os.environ["ROLLOUT_NNODES"])
expected = {"GPU": 4 * (trainer_nodes + rollout_nodes), "trainer": 4 * trainer_nodes, "rollout": 4 * rollout_nodes}
while time.monotonic() < deadline:
    resources = ray.cluster_resources()
    nodes = [node for node in ray.nodes() if node["Alive"]]
    if len(nodes) == trainer_nodes + rollout_nodes and all(resources.get(key, 0) == n for key, n in expected.items()):
        print(json.dumps({"resources": resources, "nodes": nodes}, default=str, indent=2))
        ray.shutdown()
        break
    time.sleep(2)
else:
    raise RuntimeError(f"Ray cluster was not ready: expected {expected}, got {ray.cluster_resources()}")
PY
bash "$RECIPE" "${EXTRA_ARGS[@]}" "$@"
if [ "$MODE" = smoke ]; then
  "$WORK_DIR/envs/verl/bin/python" - "$RUN" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
checkpoint = run / "ckpt" / "global_step_1"
metrics = []
for path in run.glob("metrics_*.jsonl"):
    for line in path.read_text().splitlines():
        try:
            metrics.append(json.loads(line))
        except json.JSONDecodeError:
            pass
assert checkpoint.is_dir(), f"Missing step 1 checkpoint: {checkpoint}"
checkpoint_files = {}
for kind in ("model", "optim", "extra_state"):
    for rank in range(4):
        name = f"{kind}_world_size_4_rank_{rank}.pt"
        matches = list(checkpoint.rglob(name))
        assert len(matches) == 1, f"Missing or ambiguous checkpoint shard: {name}"
        size = matches[0].stat().st_size
        assert size > 0, f"Empty checkpoint shard: {matches[0]}"
        checkpoint_files[str(matches[0].relative_to(checkpoint))] = size
step_metrics = [r for r in metrics if r.get("step") == 1 and "actor/grad_norm" in r.get("data", {})]
assert step_metrics, "No step 1 actor update metrics written"
assert math.isfinite(float(step_metrics[-1]["data"]["actor/grad_norm"])), "Non-finite gradients skipped the update"
result = {"completed_rl_step": 1, "checkpoint": str(checkpoint),
          "checkpoint_files": checkpoint_files, "metrics": step_metrics[-1]}
(run / "verified-step.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
print("PEAR_RL_STEP_VERIFIED " + json.dumps(result, default=str))
PY
fi
