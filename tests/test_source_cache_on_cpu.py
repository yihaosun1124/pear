import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STARTUP = Path(__file__).resolve().parents[1] / "pear/utils/sitecustomize"
spec = importlib.util.spec_from_file_location("source_cache_under_test", STARTUP / "pear_source_cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class SourceCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.local = self.root / "local"
        self.environment = self.work / "envs/verl"
        self.site = self.environment / "lib/python3.12/site-packages"
        self.source = self.site / "transformers"
        self.source.mkdir(parents=True)
        (self.source / "__init__.py").write_text('VERSION = "same-source"\n')
        (self.source / "data.txt").write_bytes(b"a resource, not only Python files\n")
        (self.source / "__pycache__").mkdir()
        (self.source / "__pycache__/ignored.pyc").write_bytes(b"not source")
        info = self.site / "transformers-1.0.dist-info"
        info.mkdir()
        (info / "RECORD").write_text("transformers/__init__.py,,\ntransformers/data.txt,,\n")
        self.addCleanup(patch.stopall)
        self.real_prefix = str(Path(sys.prefix).resolve())
        patch.dict(os.environ, SLURM_CLUSTER_NAME="test-cluster", LOCAL_DIR=str(self.local)).start()
        patch.object(sys, "prefix", str(self.environment)).start()
        patch("sysconfig.get_paths", return_value={"purelib": str(self.site)}).start()
        patch("subprocess.check_output", return_value="same-checkout\n").start()

    def build(self):
        meta = cache.build(self.root, self.work)
        ready = self.work / "cache/transformers-source-ready.json"
        ready.write_text(json.dumps(meta))
        return meta

    def test_snapshot_preserves_source_and_resources_and_activates_once(self):
        meta = self.build()
        result = cache.stage(self.root, self.work, self.local)
        self.assertEqual(result["source_cache"], "loaded")
        self.assertEqual(result["files"], 2)
        target = self.local / "python-packages"
        for relative, entry in meta["files"].items():
            self.assertEqual(hashlib.sha256((target / relative).read_bytes()).hexdigest(), entry["sha256"])
            self.assertEqual((target / relative).read_bytes(), (self.site / relative).read_bytes())
        with patch.object(sys, "path", list(sys.path)):
            self.assertTrue(cache.activate())
            self.assertTrue(cache.activate())
            self.assertEqual(sys.path.count(str(target)), 1)
            self.assertEqual(sys.path[0], str(target))

    def test_missing_incomplete_foreign_environment_tag_and_cluster_do_not_activate(self):
        self.assertFalse(cache.activate())
        self.build()
        cache.stage(self.root, self.work, self.local)
        marker = self.local / "python-packages/.ready.json"
        original = json.loads(marker.read_text())
        for key in ["python_env", "python_cache_tag", "cluster"]:
            marker.write_text(json.dumps(dict(original, **{key: "different"})))
            self.assertFalse(cache.activate(), key)
        marker.write_text(json.dumps(original))
        (marker.parent / "transformers/__init__.py").unlink()
        self.assertFalse(cache.activate())

    def test_symlinked_local_tree_does_not_activate(self):
        self.build()
        cache.stage(self.root, self.work, self.local)
        target = self.local / "python-packages"
        target.rename(self.local / "other")
        target.symlink_to(self.local / "other", target_is_directory=True)
        self.assertFalse(cache.activate())

    def test_foreign_cluster_skips_staging(self):
        self.build()
        with patch.dict(os.environ, SLURM_CLUSTER_NAME="another-cluster"):
            self.assertEqual(cache.stage(self.root, self.work, self.local)["source_cache"], "skipped")
        self.assertFalse((self.local / "python-packages").exists())

    def test_checksum_or_changed_installed_record_rejects_before_publishing(self):
        meta = self.build()
        Path(meta["record"]).write_text("updated distribution")
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.stage(self.root, self.work, self.local)
        self.assertFalse((self.local / "python-packages").exists())
        meta = self.build()
        Path(meta["archive"]).write_bytes(b"damaged archive")
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.stage(self.root, self.work, self.local)

    def test_traversal_and_links_reject_before_extraction(self):
        for kind in ["traversal", "symlink"]:
            meta = self.build()
            archive = Path(meta["archive"])
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("transformers/../../escaped" if kind == "traversal" else "transformers/link")
                if kind == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/tmp/unsafe-source-link"
                    bundle.addfile(member)
                else:
                    member.size = 1
                    bundle.addfile(member, io.BytesIO(b"x"))
            meta["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            (self.work / "cache/transformers-source-ready.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                cache.stage(self.root, self.work, self.local)
            self.assertFalse((self.local / "python-packages").exists())
            self.assertFalse((self.root / "escaped").exists())

    def test_real_startup_imports_identical_local_package_but_keeps_distribution_metadata(self):
        # A fresh interpreter exercises sitecustomize before Transformers imports.
        self.build()
        cache.stage(self.root, self.work, self.local)
        marker = self.local / "python-packages/.ready.json"
        meta = json.loads(marker.read_text())
        # sys.prefix was patched only in the parent; use the child's real prefix.
        meta["python_env"] = self.real_prefix
        marker.write_text(json.dumps(meta))
        env = dict(os.environ, PYTHONPATH=str(STARTUP) + os.pathsep + str(self.site),
                   PEAR_FAST_PACKAGE_METADATA="0")
        code = ("import transformers,importlib.metadata as m; "
                "assert transformers.VERSION == 'same-source'; "
                "assert '/python-packages/transformers/' in transformers.__file__; "
                "assert m.distribution('transformers')._path.name == 'transformers-1.0.dist-info'")
        subprocess.run([sys.executable, "-S", "-c",
                        "import site; site.main(); " + code], env=env, check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
