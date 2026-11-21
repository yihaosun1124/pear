import os
import subprocess
import tempfile
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "scripts/gpu_process_variables.sh"
CACHES = {
    "VLLM_CACHE_ROOT": "vllm",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang/tmp",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "CUDA_CACHE_PATH": "cuda",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "fa4_cute",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer_verl",
}


class GpuProcessVariablesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local = self.root / "new-job"
        self.local.mkdir()
        self.environment = self.root / "verl"
        (self.environment / "bin").mkdir(parents=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # Only the filesystem query and the interpreter's sysconfig query need
        # stand-ins. The real Bash helper computes and exports every cache path.
        programs = {
            self.bin / "stat": '#!/bin/sh\nprintf "%s\\n" "$PEAR_TEST_FILESYSTEM"\n',
            self.environment / "bin/python": '#!/bin/sh\nprintf "%s\\n" "$PEAR_TEST_PURELIB"\n',
        }
        for path, source in programs.items():
            path.write_text(source)
            path.chmod(0o755)

    def source(self, slurm=True, filesystem="xfs"):
        env = dict(os.environ)
        env.pop("SLURM_JOB_ID", None)
        env.pop("SLURM_JOBID", None)
        env.update({name: str(self.root / "old-cpu-job" / name) for name in CACHES})
        env.update(
            LOCAL_DIR=str(self.local),
            WORK_DIR=str(self.root / "work"),
            PATH=str(self.bin) + os.pathsep + env.get("PATH", ""),
            PEAR_TEST_FILESYSTEM=filesystem,
            PEAR_TEST_PURELIB=str(self.environment / "lib/python3.12/site-packages"),
            PYTHONPYCACHEPREFIX=str(self.root / "old-cpu-job/python-bytecode"),
            CUDA_VISIBLE_DEVICES="0,1,2,3",
            ROCR_VISIBLE_DEVICES="old-amd-selector",
            HIP_VISIBLE_DEVICES="old-amd-selector",
            HF_HOME=str(self.root / "shared-downloads"),
            CCACHE_DIR=str(self.root / "shared-ccache"),
        )
        if slurm:
            env["SLURM_JOB_ID"] = "998877"
        names = [*CACHES, "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE",
                 "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "HF_HOME", "CCACHE_DIR"]
        script = ('set -euo pipefail\nsource "$1" "$2"\n'
                  'for name in ' + " ".join(names) + '; do\n'
                  '  printf "%s=%s\\n" "$name" "${!name-}"\n'
                  'done\n')
        result = subprocess.run(
            ["bash", "-c", script, "cache-regression", str(HELPER), str(self.environment)],
            env=env, text=True, capture_output=True, check=True, timeout=30,
        )
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_slurm_job_rebases_inherited_cpu_caches_and_preserves_gpu_assignment(self):
        values = self.source()
        for name, suffix in CACHES.items():
            self.assertEqual(values[name], str(self.local / suffix), name)
        self.assertEqual(values["PYTHONPYCACHEPREFIX"], str(self.local / "python-bytecode"))
        self.assertEqual(values["PYTHONDONTWRITEBYTECODE"], "")
        self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(values["ROCR_VISIBLE_DEVICES"], "")
        self.assertEqual(values["HIP_VISIBLE_DEVICES"], "")
        self.assertEqual(values["HF_HOME"], str(self.root / "shared-downloads"))
        self.assertEqual(values["CCACHE_DIR"], str(self.root / "shared-ccache"))

    def test_non_slurm_call_preserves_explicit_cache_overrides(self):
        values = self.source(slurm=False)
        for name in CACHES:
            self.assertEqual(values[name], str(self.root / "old-cpu-job" / name), name)

    def test_unsupported_filesystem_drops_the_parent_python_cache(self):
        values = self.source(filesystem="tmpfs")
        self.assertEqual(values["PYTHONPYCACHEPREFIX"], "")
        self.assertEqual(values["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
