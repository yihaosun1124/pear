"""Call the installed PEAR launcher inside a Cluv allocation."""

import json
import hashlib
import os
from pathlib import Path
import sys


def main():
    config = json.loads((Path(__file__).resolve().parents[1] / "clusters.json").read_text())
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    matches = [(name, item) for name, item in config.items() if item["slurm_cluster"] == cluster]
    if len(matches) != 1:
        raise SystemExit("Choose tamia, fir, rorqual or trillium-gpu; Mila is the controller.")
    _, settings = matches[0]
    if os.environ.get("SLURM_JOB_NUM_NODES") != "8":
        raise SystemExit("PEAR requires eight nodes: two actor and six rollout nodes.")
    root = Path(settings["root"])
    manifest = json.loads((Path(__file__).resolve().parents[1] / "launcher_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise SystemExit("Shared PEAR launcher differs from the validated manifest: " + name)
    launcher = root / "deploy/train.sbatch"
    if not launcher.is_file():
        raise SystemExit("Installed PEAR launcher is missing: " + str(launcher))
    command = [
        "bash", str(launcher), "PEAR_ROOT=" + str(root),
        "WORK_DIR=" + settings["work"], "RUN=", "TRAINER_NNODES=2",
        "ROLLOUT_NNODES=6", "LMCACHE_GB=64",
    ]
    if settings.get("allow_tmpfs_local"):
        command.append("ALLOW_TMPFS_LOCAL=1")
    command.extend(["--", "trainer.max_actor_ckpt_to_keep=1"])
    if "numa_headroom_gib" in settings:
        command.append("+actor_rollout_ref.rollout.custom.lmcache.numa_headroom_gb="
                       + str(settings["numa_headroom_gib"]))
    command.extend(sys.argv[1:])
    os.chdir(root)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
