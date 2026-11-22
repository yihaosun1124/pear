import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("race", ROOT / "scripts/submit_race.py")
race = importlib.util.module_from_spec(spec)
spec.loader.exec_module(race)


class QueueRaceTests(unittest.TestCase):
    def test_forecast_start_on_pending_job_cannot_win(self):
        self.assertIsNone(race.choose_winner({"fir": {"state": "PENDING", "start": "2026-01-01T00:00:00"}}))

    def test_failed_and_cancelled_jobs_cannot_win(self):
        self.assertIsNone(race.choose_winner({"fir": {"state": "FAILED"}, "tamia": {"state": "CANCELLED"}}))

    def test_two_runners_are_ordered_by_real_start(self):
        self.assertEqual(race.choose_winner({"fir": {"state": "RUNNING", "start": "2026-01-01T02:00:00"},
                                           "tamia": {"state": "RUNNING", "start": "2026-01-01T01:00:00"}}), "tamia")

    def test_only_four_drac_targets_are_eligible(self):
        self.assertEqual(set(race.TARGETS), set(json.loads((ROOT / "clusters.json").read_text())))
        self.assertNotIn("mila", race.TARGETS)
        self.assertNotIn("trillium", race.TARGETS)

    def test_resource_audit_accepts_typed_gpu_and_node_range(self):
        item = {"resources": {"NumNodes": "8-8", "TimeLimit": "02:00:00", "CPUs/Task": "48",
                              "MinMemoryNode": "450G", "ReqTRES": "cpu=384,node=8,gres/gpu:h100=32"}}
        race.check_resources("fir", item)
        item["resources"]["ReqTRES"] = "gres/gpu=320"
        with self.assertRaises(RuntimeError): race.check_resources("fir", item)

    def test_winner_is_never_cancelled_and_losers_are_marked_first(self):
        state = {"race_id": "test", "controller_host": "test", "winner": None,
                 "candidates": {alias: {"job_id": str(i), "state": "PENDING"}
                                for i, alias in enumerate(race.TARGETS, 1)}}
        statuses = {alias: dict(x, monitor_job_id=x["job_id"], start="2026-01-01T00:00:00")
                    for alias, x in state["candidates"].items()}
        statuses["fir"]["state"] = "RUNNING"
        events = []
        def rpc(alias, action, **data):
            if action == "snapshot": return copy.deepcopy(statuses[alias])
            self.assertEqual(action, "cancel")
            self.assertNotEqual(alias, "fir")
            self.assertEqual(data["job_id"], statuses[alias]["job_id"])
            self.assertIn((alias, "cancel_requested"), events)
            events.append((alias, "cancel")); statuses[alias]["state"] = "CANCELLED"
        with patch.object(race, "rpc", side_effect=rpc), patch.object(race, "persist"), \
             patch.object(race, "mark", side_effect=lambda s,a,o: events.append((a,o))), \
             patch.object(race.time, "sleep"):
            race.watch(state)
        self.assertEqual(state["winner"], "fir")
        self.assertEqual({a for a,o in events if o == "cancel"}, set(race.TARGETS) - {"fir"})
        self.assertEqual(state["phase"], "winner_selected")

    def test_embedded_remote_program_has_valid_python_syntax(self):
        compile(race.REMOTE, "remote", "exec")

    def test_registration_creates_slurm_stdout_directory_before_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "runs").mkdir(); (work / "logs").mkdir()
            monitor = work / "runs/monitor.json"
            monitor.write_text(json.dumps({"current_job_id": "1"}))
            payload = {"config": {"monitor": str(monitor), "work": str(work)},
                       "job_id": "2", "expected_current": "1", "race_id": "test",
                       "receipt": {"job_id": "2", "sbatch_args": {
                           "output": str(work / "cluv_runs/fir_%j/slurm-%j.out")}}}
            with patch.object(sys, "argv", ["-c", "register", json.dumps(payload)]), patch("builtins.print"):
                exec(compile(race.REMOTE, "remote", "exec"), {})
            self.assertTrue((work / "cluv_runs/fir_2").is_dir())
            self.assertEqual(json.loads(monitor.read_text())["current_job_id"], "2")
            receipt = json.loads((work / "logs/cluv-race-test-2-submit.json").read_text())
            self.assertEqual(receipt["resolved_slurm_output"], str((work / "cluv_runs/fir_2/slurm-2.out").resolve()))


if __name__ == "__main__":
    unittest.main()
