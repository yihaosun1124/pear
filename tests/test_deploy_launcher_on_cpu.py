"""Exercise the shared launcher's real argument/default handling without GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "deploy/train.sbatch"


class SharedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scratch = self.root / "scratch/s/testuser"
        work = self.scratch / "pear_work"
        executable = work / "envs/verl/bin/python"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o755)
        runtime = self.root / "scripts/gpu_process_variables.sh"
        runtime.parent.mkdir()
        runtime.write_text('pear_set_ray_tmpdir() { export RAY_TMPDIR="$LOCAL_DIR/r"; }\n')
        recipe = self.root / "examples/models/qwen3_8_27b_pear/train/run_stage1.sh"
        recipe.parent.mkdir(parents=True)
        probe = "import json,os,sys; keys={'trainer':'TRAINER_NNODES','rollout':'ROLLOUT_NNODES','batch':'BATCH','n':'N','steps':'STEPS','response':'MAX_RESPONSE','tmpfs':'ALLOW_TMPFS_LOCAL','gpus':'GPUS_PER_NODE'}; data={k:int(os.environ[v]) for k,v in keys.items()}; data.update(work=os.environ['WORK_DIR'],args=sys.argv[1:]); print(json.dumps(data))"
        import shlex
        recipe.write_text('#!/bin/bash\nexec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(probe) + ' "$@"\n')

    def run_launcher(self, *args, smoke=False):
        env = {**os.environ, "SCRATCH": str(self.scratch), "USER": "testuser"}
        env.pop("WORK_DIR", None)
        env.pop("RUN", None)
        env.pop("ALLOW_TMPFS_LOCAL", None)
        mode = ["--smoke"] if smoke else []
        return subprocess.run(["bash", str(LAUNCHER), *mode, "--config-only", "PEAR_ROOT=" + str(self.root), *args],
                              text=True, capture_output=True, env=env)

    def test_defaults_are_32_gpus_and_official_training_parameters(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual((output["trainer"], output["rollout"]), (2, 6))
        self.assertEqual((output["batch"], output["n"], output["steps"], output["response"]), (128, 8, 1000, 160000))

    def test_default_work_directory_uses_the_site_scratch_path(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["work"], str(self.scratch / "pear_work"))

    def test_explicit_tmpfs_profile_is_a_deployment_option(self):
        result = self.run_launcher("ALLOW_TMPFS_LOCAL=1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["tmpfs"], 1)

    def test_invalid_tmpfs_profile_is_rejected(self):
        result = self.run_launcher("ALLOW_TMPFS_LOCAL=2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ALLOW_TMPFS_LOCAL must be 0 or 1", result.stderr)

    def test_eight_gpu_nodes_keep_the_formal_eight_plus_twenty_four_split(self):
        result = self.run_launcher("GPUS_PER_NODE=8", "TRAINER_NNODES=1", "ROLLOUT_NNODES=3")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual((output["trainer"] * output["gpus"], output["rollout"] * output["gpus"]), (8, 24))
        self.assertIn("trainer.n_gpus_per_node=8", output["args"])
        self.assertIn("actor_rollout_ref.rollout.n_gpus_per_node=8", output["args"])
        self.assertEqual((output["batch"], output["n"], output["steps"], output["response"]), (128, 8, 1000, 160000))

    def test_eight_plus_eight_smoke_uses_an_eight_rank_actor(self):
        result = self.run_launcher("GPUS_PER_NODE=8", smoke=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual((output["trainer"], output["rollout"], output["gpus"], output["steps"]), (1, 1, 8, 1))
        self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=8", output["args"])
        self.assertIn("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=8", output["args"])

    def test_four_gpu_node_cannot_supply_an_official_eight_gpu_actor_alone(self):
        result = self.run_launcher("GPUS_PER_NODE=4", "TRAINER_NNODES=1", "ROLLOUT_NNODES=3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SP8", result.stderr)


if __name__ == "__main__":
    unittest.main()
