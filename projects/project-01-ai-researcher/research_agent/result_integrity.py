"""Deterministic verification for the CPU experiment's artifacts and claims."""

import hashlib
import json
import math
from pathlib import Path


SCOPE = "fixed_demo_test"
DETECTORS = {
    "keyword_baseline": "关键词基线",
    "model_rules": "规范化与模型规则",
}
METRICS = {
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1",
    "false_positive_rate": "误报率",
    "paired_success_rate": "成对成功率",
}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 包含重复字段")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("JSON 包含非有限数值")


def parse_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def _read_evaluation(directory):
    metrics = read_json(directory / "metrics.json")
    predictions = [
        parse_json(line)
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for values in metrics["detectors"].values():
        elapsed = values.pop("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("评估耗时无效")
    return metrics, predictions


def verify_outputs(evaluation_directory, replay_directory):
    metrics, predictions = _read_evaluation(evaluation_directory)
    replay_metrics, replay_predictions = _read_evaluation(replay_directory)
    if canonical_json(predictions) != canonical_json(replay_predictions):
        raise ValueError("逐条预测与可信复算不一致")
    if canonical_json(metrics) != canonical_json(replay_metrics):
        raise ValueError("评估指标与可信复算不一致")
    return replay_metrics, {
        "status": "verified",
        "metrics_sha256": sha256(canonical_json(replay_metrics).encode("utf-8")),
        "predictions_sha256": sha256(canonical_json(replay_predictions).encode("utf-8")),
    }


def freeze_inputs(directory, run_id, records, plan, evaluator, *, rule_source="model"):
    """Keep trusted inputs outside the evaluation container's writable mount."""
    inputs = directory / "inputs"
    inputs.mkdir()
    write_json(inputs / "dataset.json", records)
    write_json(inputs / "plan.json", plan)
    (inputs / "evaluate.py").write_bytes(evaluator.read_text(encoding="utf-8").encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": SCOPE,
        "rule_source": rule_source,
        "files": {
            name: sha256((inputs / name).read_bytes())
            for name in ("dataset.json", "plan.json", "evaluate.py")
        },
        "development_samples": sum(row["split"] == "development" for row in records),
        "test_samples": sum(row["split"] == "test" for row in records),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_inputs(directory, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("scope") != SCOPE:
        raise ValueError("评估记录的版本或适用范围无效")
    files = manifest.get("files", {})
    for name in ("dataset.json", "plan.json", "evaluate.py"):
        if sha256((directory / "inputs" / name).read_bytes()) != files.get(name):
            raise ValueError(f"评估输入校验失败 {name}")


def verify_claims(payload, manifest, metrics):
    """Certify structured numerical claims with the full source identity."""
    if (not isinstance(payload, dict) or set(payload) != {"claims"}
            or not isinstance(payload["claims"], list) or len(payload["claims"]) > 20):
        return {"verified": [], "review_required": [{"index": None, "reason": "invalid_claims_schema"}]}
    identity = sha256(canonical_json(manifest).encode("utf-8"))
    required = {"run_id", "manifest_sha256", "scope", "detector", "metric", "value"}
    result = {"verified": [], "review_required": []}
    for index, claim in enumerate(payload["claims"]):
        reason = None
        if not isinstance(claim, dict) or set(claim) != required:
            reason = "invalid_claim_schema"
        elif (claim["run_id"] != manifest["run_id"]
              or claim["manifest_sha256"] != identity or claim["scope"] != manifest["scope"]):
            reason = "source_or_scope_mismatch"
        elif (not isinstance(claim["detector"], str) or not isinstance(claim["metric"], str)
              or claim["detector"] not in DETECTORS or claim["metric"] not in METRICS):
            reason = "unsupported_detector_or_metric"
        elif type(claim["value"]) not in (int, float) or not 0 <= claim["value"] <= 1:
            reason = "invalid_numeric_value"
        elif not math.isclose(claim["value"], metrics["detectors"][claim["detector"]][claim["metric"]],
                              rel_tol=0, abs_tol=1e-12):
            reason = "claim_value_mismatch"
        if reason:
            result["review_required"].append({"index": index, "reason": reason})
        else:
            result["verified"].append(dict(claim))
    return result


def render_report(metrics, manifest, verification):
    """Render verified facts using fixed labels and host-verified values."""
    titles = dict(DETECTORS)
    rule_description = "规则由模型依据开发样本设计。"
    if manifest.get("rule_source") == "manual_control":
        titles["model_rules"] = "人工规范化对照"
        rule_description = "规则为人工构造的规范化关键词对照。"
    lines = [
        "# CPU 提示注入检测实验", "",
        "## 实验设置", "",
        f"- 开发集 {manifest['development_samples']} 条，测试集 {manifest['test_samples']} 条。",
        "- 数据来自手工构造的固定演示集。",
        f"- {rule_description}",
        "- 规则方法采用 NFKC 规范化、移除格式控制字符、折叠空白和正则匹配。",
        "- 成对成功率要求一对中的正常样本和注入样本均被正确分类。",
        "- 评估输入通过摘要核验，逐条预测和指标通过独立进程复算。",
        "", "## 已验证指标", "",
        "| 方法 | 精确率 | 召回率 | F1 | 误报率 | 成对成功率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for detector, title in titles.items():
        values = metrics["detectors"][detector]
        cells = [f"{values[metric]:.3f}" for metric in METRICS]
        lines.append("| " + " | ".join([title, *cells]) + " |")
    baseline = metrics["detectors"]["keyword_baseline"]["f1"]
    proposed = metrics["detectors"]["model_rules"]["f1"]
    lines.extend([
        "", "## 复算结论", "",
        f"规则方法在本演示集上的 F1 与关键词基线之差为 {proposed - baseline:+.3f}。",
        "", "## 数值陈述核验", "",
    ])
    for claim in verification["claims"]["verified"]:
        title = titles[claim["detector"]]
        metric = METRICS[claim["metric"]]
        value = metrics["detectors"][claim["detector"]][claim["metric"]]
        lines.append(f"- {title}的{metric}为 {value:.3f}，适用范围为本轮固定演示测试集。")
    pending = len(verification["claims"]["review_required"])
    lines.extend([
        f"- 待核验陈述 {pending} 项，原始回复保存在 model_claims.txt。",
        "", "## 实验来源", "",
        f"- 运行编号为 {manifest['run_id']}。",
        f"- 来源记录摘要为 {sha256(canonical_json(manifest).encode('utf-8'))}。",
        "- manifest.json 保存输入摘要，verification.json 保存复算结果与陈述核验记录。",
        "- inputs/plan.json 保存待检验的研究假设及规则。",
        "", "## 实验范围", "",
        "结果描述固定演示集上的文本分类效果。真实资料上的泛化、统计显著性和完整代理的攻击防御效果仍待独立实验验证。",
        "",
    ])
    return "\n".join(lines)
