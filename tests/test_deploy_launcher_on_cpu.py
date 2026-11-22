"""Exercise the shared launcher's real argument/default handling without GPUs."""

import json
import os
from pathlib import Path
import subprocess
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
        recipe.write_text('''#!/bin/bash
printf '{"trainer":%s,"rollout":%s,"batch":%s,"n":%s,"steps":%s,"response":%s,"tmpfs":%s,"work":"%s"}\n' \\
 "$TRAINER_NNODES" "$ROLLOUT_NNODES" "$BATCH" "$N" "$STEPS" "$MAX_RESPONSE" "$ALLOW_TMPFS_LOCAL" "$WORK_DIR"
''')

    def run_launcher(self, *args):
        env = {**os.environ, "SCRATCH": str(self.scratch), "USER": "testuser"}
        env.pop("WORK_DIR", None)
        env.pop("RUN", None)
        env.pop("ALLOW_TMPFS_LOCAL", None)
        return subprocess.run(["bash", str(LAUNCHER), "--config-only", "PEAR_ROOT=" + str(self.root), *args],
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


if __name__ == "__main__":
    unittest.main()
