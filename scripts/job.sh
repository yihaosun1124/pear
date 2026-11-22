#!/usr/bin/env bash
set -euo pipefail
# Cluv supplies the resources, paths and environment from pyproject.toml.
: "${PEAR_ROOT:?Missing PEAR_ROOT}" "${WORK_DIR:?Missing WORK_DIR}"
case ${CLUV_CLUSTER:-} in tamia|fir|rorqual|trillium-gpu|nibi) ;; *) exit 1 ;; esac
trainer=${TRAINER_NNODES:-2}
rollout=${ROLLOUT_NNODES:-6}
[[ ${SLURM_JOB_NUM_NODES:?} == $((trainer + rollout)) ]] || { echo 'Unexpected node count' >&2; exit 1; }
args=("PEAR_ROOT=$PEAR_ROOT" "WORK_DIR=$WORK_DIR" "RUN="
      "TRAINER_NNODES=$trainer" "ROLLOUT_NNODES=$rollout" "GPUS_PER_NODE=${GPUS_PER_NODE:-4}"
      "LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-32}" LMCACHE_GB=64)
if [[ ${ALLOW_TMPFS_LOCAL:-0} == 1 ]]; then args+=(ALLOW_TMPFS_LOCAL=1); fi
args+=(-- trainer.max_actor_ckpt_to_keep=1)
if [[ -n ${NUMA_HEADROOM_GB:-} ]]; then
  args+=("+actor_rollout_ref.rollout.custom.lmcache.numa_headroom_gb=$NUMA_HEADROOM_GB")
fi
cd -- "$PEAR_ROOT"
exec bash deploy/train.sbatch "${args[@]}" "$@"
