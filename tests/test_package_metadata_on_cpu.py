import importlib.metadata as metadata
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STARTUP = Path(__file__).resolve().parents[1] / "pear/utils/sitecustomize"
spec = importlib.util.spec_from_file_location("metadata_under_test", STARTUP / "pear_package_metadata.py")
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)


@unittest.skipUnless(sys.version_info[:2] == (3, 12), "Optimization targets the pinned Python 3.12 stdlib")
class PackageMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def distribution(self, name, entries, present=(), declared=None):
        info = self.root / f"{name}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
        (info / "RECORD").write_text("".join(f"{entry},,\n" for entry in entries))
        if declared is not None:
            (info / "top_level.txt").write_text(declared)
        for entry in present:
            path = self.root / entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        return metadata.PathDistribution(info)

    def compare(self, distributions):
        with patch.object(metadata, "distributions", return_value=distributions):
            expected = fast.ORIGINAL_PACKAGES_DISTRIBUTIONS()
            actual = fast.packages_distributions()
        self.assertEqual(actual, expected)
        return actual

    def test_missing_first_file_and_missing_entire_package(self):
        dist = self.distribution(
            "example",
            ["package/missing.h", "package/module.py", "gone/missing.py", "single.py", "LICENSE"],
            ["package/module.py", "single.py", "LICENSE"],
        )
        self.assertEqual(self.compare([dist]), {"package": ["example"], "single": ["example"]})

    def test_declared_names_and_shared_namespace_keep_distribution_order(self):
        first = self.distribution("first", [], declared="shared declared shared\n")
        second = self.distribution("second", ["shared/module.py"], ["shared/module.py"])
        result = self.compare([first, second])
        self.assertEqual(result["shared"], ["first", "first", "second"])

    def test_header_inventory_does_not_cause_one_stat_per_header(self):
        entries = ["package/__init__.py"] + [f"package/include/header_{i}.h" for i in range(10000)]
        dist = self.distribution("large", entries, ["package/__init__.py"])
        original_exists = Path.exists
        calls = []

        def exists(path):
            calls.append(path)
            return original_exists(path)

        with patch.object(metadata, "distributions", return_value=[dist]), patch.object(Path, "exists", exists):
            expected = fast.ORIGINAL_PACKAGES_DISTRIBUTIONS()
            original_count = len(calls)
            calls.clear()
            actual = fast.packages_distributions()
            fast_count = len(calls)
        self.assertEqual(actual, expected)
        self.assertGreaterEqual(original_count, 10001)
        self.assertLessEqual(fast_count, 2)

    def test_legacy_metadata_uses_the_standard_fallback(self):
        dist = self.distribution("legacy", [], ["legacy.py"])
        (dist._path / "RECORD").unlink()
        (dist._path / "SOURCES.txt").write_text("legacy.py\n")
        self.compare([dist])

    def test_bad_record_size_is_not_hidden(self):
        dist = self.distribution("broken", ["module.py"], ["module.py"])
        (dist._path / "RECORD").write_text("module.py,,invalid-size\n")
        with patch.object(metadata, "distributions", return_value=[dist]):
            with self.assertRaises(ValueError):
                fast.ORIGINAL_PACKAGES_DISTRIBUTIONS()
            with self.assertRaises(ValueError):
                fast.packages_distributions()

    def test_startup_can_enable_and_disable_the_optimization(self):
        for enabled in ("0", "1"):
            env = dict(os.environ, PYTHONPATH=str(STARTUP), PEAR_FAST_PACKAGE_METADATA=enabled)
            code = (
                "import importlib.metadata as m, pear_package_metadata as f; "
                f"assert (m.packages_distributions is f.packages_distributions) == {enabled == '1'}"
            )
            subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
