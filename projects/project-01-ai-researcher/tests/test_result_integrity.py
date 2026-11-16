"""Offline regression checks for the normal CPU flow and result integrity."""

import ast
import copy
import json
import logging
import queue
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from research_agent import cpu_experiment as experiment
from research_agent.cpu_prompt_eval import calculate_metrics
from research_agent.result_integrity import (
    canonical_json, parse_json, read_json, render_report, sha256, verify_claims, write_json,
)


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(node for node in ast.walk(tree) if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.result = experiment.run_cpu_experiment(offline=True, output_root=cls.root)
        cls.directory = Path(cls.result["result_directory"])
        cls.manifest = read_json(cls.directory / "manifest.json")
        cls.metrics = read_json(cls.directory / "evaluation" / "metrics.json")
        cls.claim = {
            "run_id": cls.manifest["run_id"],
            "manifest_sha256": sha256(canonical_json(cls.manifest).encode("utf-8")),
            "scope": cls.manifest["scope"],
            "detector": "keyword_baseline", "metric": "f1", "value": 0.6,
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_offline_flow_and_saved_verification(self):
        record = read_json(self.directory / "run.json")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["claims_verified"], 1)
        self.assertEqual(record["token_info"]["prompt_token_count"], 0)
        self.assertEqual(experiment.verify_saved_run(self.directory)["evaluation"]["status"], "verified")
        self.assertIn("人工规范化对照", self.result["answer"])
        self.assertIn("0.600", self.result["answer"])
        self.assertFalse(list(self.directory.glob(".replay-*")))

    def test_correct_claims_are_accepted(self):
        for metric, value in (("f1", 0.6), ("precision", 0.75), ("recall", 0.5)):
            with self.subTest(metric=metric):
                claim = {**self.claim, "metric": metric, "value": value}
                result = verify_claims({"claims": [claim]}, self.manifest, self.metrics)
                self.assertEqual(len(result["verified"]), 1)

    def test_wrong_or_unsupported_claims_are_pending(self):
        mutations = [
            {"value": 1.0}, {"value": 0.75}, {"value": 0.7826086956521738},
            {"run_id": "other-run"}, {"manifest_sha256": "0" * 64},
            {"scope": "all_real_papers"}, {"metric": "statistical_significance"},
            {"detector": "unknown"}, {"value": float("nan")}, {"value": True},
            {"value": "0.6"}, {"value": 10 ** 500}, {"metric": []}, {"detector": {}},
            {"comment": "F1 为 1.000，所有真实论文均有效"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result = verify_claims(
                    {"claims": [{**self.claim, **mutation}]}, self.manifest, self.metrics,
                )
                self.assertEqual(result["verified"], [])
                self.assertEqual(len(result["review_required"]), 1)

    def test_malformed_payloads_and_json(self):
        for payload in (None, [], {}, {"claims": "text"}, {"claims": [self.claim] * 21}, {"claims": [], "analysis": "text"}):
            with self.subTest(payload=payload):
                self.assertEqual(verify_claims(payload, self.manifest, self.metrics)["verified"], [])
        for text in ('{"value": NaN}', '{"value": 1, "value": 2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json(text)
        self.assertEqual(experiment._check_claims("F1 为 1.000", self.manifest, self.metrics)["verified"], [])

    def test_report_excludes_unverified_model_text(self):
        payload = {"claims": [{**self.claim, "value": 1.0, "analysis": "FABRICATED_CONCLUSION"}]}
        claims = verify_claims(payload, self.manifest, self.metrics)
        report = render_report(self.metrics, self.manifest, {"claims": claims})
        self.assertNotIn("FABRICATED_CONCLUSION", report)
        self.assertIn("待核验陈述 1 项", report)
        self.assertIn("0.600", report)

    def test_input_tampering_is_rejected(self):
        for filename in ("dataset.json", "plan.json", "evaluate.py"):
            with self.subTest(filename=filename):
                path = self.directory / "inputs" / filename
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ValueError, "输入校验失败"):
                        experiment.verify_saved_run(self.directory)
                finally:
                    path.write_bytes(original)

    def test_modified_evaluator_cannot_be_executed_by_saved_audit(self):
        evaluator = self.directory / "inputs" / "evaluate.py"
        manifest_path = self.directory / "manifest.json"
        run_path = self.directory / "run.json"
        originals = {path: path.read_bytes() for path in (evaluator, manifest_path, run_path)}
        try:
            evaluator.write_text("raise RuntimeError('untrusted code executed')", encoding="utf-8")
            manifest = copy.deepcopy(self.manifest)
            manifest["files"]["evaluate.py"] = sha256(evaluator.read_bytes())
            write_json(manifest_path, manifest)
            record = read_json(run_path)
            record["manifest_sha256"] = sha256(canonical_json(manifest).encode("utf-8"))
            write_json(run_path, record)
            with self.assertRaisesRegex(ValueError, "评估器版本不匹配"):
                experiment.verify_saved_run(self.directory)
        finally:
            for path, contents in originals.items():
                path.write_bytes(contents)

    def test_modified_metrics_are_rejected(self):
        path = self.directory / "evaluation" / "metrics.json"
        original = path.read_bytes()
        try:
            metrics = read_json(path)
            metrics["detectors"]["keyword_baseline"]["f1"] = 1.0
            write_json(path, metrics)
            with self.assertRaisesRegex(ValueError, "指标.*不一致"):
                experiment.verify_saved_run(self.directory)
        finally:
            path.write_bytes(original)

    def test_missing_duplicate_and_coherently_changed_predictions(self):
        prediction_path = self.directory / "evaluation" / "predictions.jsonl"
        metric_path = self.directory / "evaluation" / "metrics.json"
        original_predictions = prediction_path.read_bytes()
        original_metrics = metric_path.read_bytes()
        predictions = [json.loads(line) for line in original_predictions.decode("utf-8").splitlines()]
        for mutation in ("missing", "duplicate", "coherent"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(predictions)
                try:
                    if mutation == "missing":
                        changed.pop()
                    elif mutation == "duplicate":
                        changed.append(copy.deepcopy(changed[0]))
                    else:
                        for row in changed:
                            row["prediction"] = row["label"]
                        metrics = read_json(metric_path)
                        for detector in metrics["detectors"]:
                            values = calculate_metrics([row for row in changed if row["detector"] == detector])
                            values["elapsed_seconds"] = 0.0
                            metrics["detectors"][detector] = values
                        write_json(metric_path, metrics)
                    prediction_path.write_text(
                        "\n".join(canonical_json(row) for row in changed) + "\n", encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "逐条预测.*不一致"):
                        experiment.verify_saved_run(self.directory)
                finally:
                    prediction_path.write_bytes(original_predictions)
                    metric_path.write_bytes(original_metrics)

    def test_legacy_results_are_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "缺少 manifest.json"):
                experiment.verify_saved_run(temporary)

    def test_web_and_main_entry_points(self):
        cpu_runner = Mock(return_value=self.result)
        namespace = {}
        function = load_function(PROJECT / "main_ai_researcher.py", "main_ai_researcher", namespace)
        with patch.object(experiment, "run_cpu_experiment", cpu_runner):
            self.assertEqual(function("question", "reference", experiment.CPU_MODE), self.result)
        cpu_runner.assert_called_once_with("question", "reference")
        namespace = {
            "Tuple": tuple, "logging": logging,
            "load_dotenv": Mock(), "find_dotenv": Mock(),
            "validate_input": lambda value: bool(value),
            "MODULE_DESCRIPTIONS": {experiment.CPU_MODE: experiment.CPU_DESCRIPTION},
            "main_ai_researcher": Mock(return_value=self.result),
        }
        web = load_function(PROJECT / "web_ai_researcher.py", "run_ai_researcher", namespace)
        self.assertEqual(web("question", "", experiment.CPU_MODE)[0], self.result["answer"])
        namespace["main_ai_researcher"].return_value = "existing text result"
        self.assertEqual(web("question", "", experiment.CPU_MODE)[0], "existing text result")

    def test_docker_mounts_protect_inputs_and_host_record(self):
        inputs = (self.directory / "inputs").resolve()
        outputs = (self.directory / "command-test").resolve()
        with patch.object(experiment.subprocess, "run", return_value=types.SimpleNamespace(stdout="cpu_evaluation_ok")) as run:
            experiment._run_evaluation(inputs, outputs, "test-run", "docker")
        arguments = run.call_args.args[0]
        self.assertIn(f"type=bind,source={inputs},target=/inputs,readonly", arguments)
        self.assertIn(f"type=bind,source={outputs},target=/outputs", arguments)
        self.assertNotIn(f"type=bind,source={self.directory},target=/workplace", arguments)
        self.assertIn("--read-only", arguments)
        self.assertIn("none", arguments)
        self.assertEqual(run.call_args.kwargs["timeout"], 45)

    def test_docker_timeout_cleans_up_the_owned_container(self):
        failure = subprocess.TimeoutExpired("docker", 45)
        with patch.object(experiment.subprocess, "run", side_effect=[failure, Mock()]) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                experiment._run_evaluation(
                    self.directory / "inputs", self.directory / "timeout-test", "timeout-run", "docker",
                )
        self.assertEqual(run.call_args.args[0], ["docker", "rm", "-f", "course_cpu_timeout-run"])

    def test_web_generator_displays_report_and_preserves_chat_state(self):
        class ImmediateThread:
            def __init__(self, target):
                self.target = target

            def start(self):
                self.target()

            def is_alive(self):
                return False

        for status, indicator in (("✅ Successfully completed", "status-success"), ("❌ Error", "status-error")):
            with self.subTest(status=status):
                namespace = {
                    "threading": types.SimpleNamespace(Thread=ImmediateThread), "queue": queue,
                    "run_ai_researcher": Mock(return_value=(self.result["answer"], "tokens", status)),
                    "get_latest_logs": lambda limit, state, logs, index: (state, index),
                    "LOG_QUEUE": queue.Queue(),
                }
                function = load_function(PROJECT / "web_ai_researcher.py", "process_with_live_logs", namespace)
                state = []
                messages = list(function("question", "", experiment.CPU_MODE, state, 0))
                self.assertIs(messages[-1][0], state)
                self.assertIn(indicator, messages[-1][1])
                self.assertEqual(messages[-1][2][-1][1], self.result["answer"])

    def test_online_orchestration_rejects_false_model_claims(self):
        plan = read_json(self.directory / "inputs" / "plan.json")

        def model_response(instructions, content):
            data = json.loads(content)
            text = canonical_json(plan)
            if "identity" in data:
                text = canonical_json({"claims": [{
                    **data["identity"], "detector": "keyword_baseline", "metric": "f1", "value": 1.0,
                }]})
            return text, {"prompt_token_count": 2, "completion_token_count": 3}

        real_evaluation = experiment._run_evaluation

        def local_evaluation(inputs, outputs, run_id, backend):
            real_evaluation(inputs, outputs, run_id, "subprocess")

        with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=Mock())}):
            with patch.object(experiment, "_ask_model", side_effect=model_response):
                with patch.object(experiment, "_run_evaluation", side_effect=local_evaluation):
                    result = experiment.run_cpu_experiment(output_root=self.root)
        directory = Path(result["result_directory"])
        record = read_json(directory / "run.json")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["claims_review_required"], 1)
        self.assertEqual(record["claims_verified"], 0)
        self.assertEqual(record["token_info"]["prompt_token_count"], 4)
        self.assertNotIn("F1为 1.000", result["answer"])
        self.assertIn("0.600", result["answer"])

    def test_verification_failure_stops_report_publication(self):
        real_evaluation = experiment._run_evaluation

        def corrupted_evaluation(inputs, outputs, run_id, backend):
            real_evaluation(inputs, outputs, run_id, "subprocess")
            if outputs.name == "evaluation":
                metrics = read_json(outputs / "metrics.json")
                metrics["detectors"]["keyword_baseline"]["f1"] = 1.0
                write_json(outputs / "metrics.json", metrics)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(experiment, "_run_evaluation", side_effect=corrupted_evaluation):
                with self.assertRaises(ValueError):
                    experiment.run_cpu_experiment(offline=True, output_root=root)
            directory = next(root.iterdir())
            record = read_json(directory / "run.json")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["stage"], "verify_evaluation")
            self.assertFalse((directory / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
