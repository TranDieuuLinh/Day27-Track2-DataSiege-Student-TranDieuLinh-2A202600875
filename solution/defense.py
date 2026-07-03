"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict

# Baselines are mean ± 3σ on clean traffic; a few signals sit on the boundary.
# These multipliers nudge thresholds without hardcoding per-run answers.
FEATURE_SIGMA_MULT = 1.2
EMBEDDING_CENTROID_MULT = 0.9
CORPUS_AGE_MULT = 0.85
MEAN_AMOUNT_MAX_MULT = 0.98


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _tool_error(result):
    return "error" in result


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if _tool_error(profile):
        return Verdict(alert=False, pillar="checks", reason=profile["error"])

    b = ctx.baseline
    reasons = []
    if profile["row_count"] < b["row_count_min"] or profile["row_count"] > b["row_count_max"]:
        reasons.append("row_count")
    null_rate = profile["null_rate"].get("customer_id", 0)
    if null_rate > b["null_rate_max"]:
        reasons.append("null_rate")
    mean_max = b["mean_amount_max"] * MEAN_AMOUNT_MAX_MULT
    if profile["mean_amount"] < b["mean_amount_min"] or profile["mean_amount"] > mean_max:
        reasons.append("mean_amount")
    if profile["staleness_min"] > b["staleness_min_max"]:
        reasons.append("staleness")

    return Verdict(alert=bool(reasons), pillar="checks", reason=", ".join(reasons))


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if _tool_error(diff):
        return Verdict(alert=False, pillar="contracts", reason=diff["error"])

    b = ctx.baseline
    reasons = list(diff["violations"])
    if diff["freshness_delay_min"] > b["freshness_delay_max_min"]:
        reasons.append("freshness_delay")

    return Verdict(alert=bool(reasons), pillar="contracts", reason=", ".join(reasons))


def _learn_lineage_baseline(graph, ctx):
    """Capture expected upstream/downstream from the first healthy-looking run."""
    b = ctx.baseline
    if ctx.state.get("lineage_expected") is not None:
        return
    if graph["duration_ms"] > b["lineage_duration_ms_max"]:
        return
    if graph["actual_downstream_count"] < 1:
        return
    ctx.state["lineage_expected"] = {
        "upstream": list(graph["actual_upstream"]),
        "downstream_count": graph["actual_downstream_count"],
    }


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _tool_error(graph):
        return Verdict(alert=False, pillar="lineage", reason=graph["error"])

    b = ctx.baseline
    reasons = []
    if graph["duration_ms"] > b["lineage_duration_ms_max"]:
        reasons.append("runtime_anomaly")

    _learn_lineage_baseline(graph, ctx)
    expected = ctx.state.get("lineage_expected")
    if expected is not None:
        if graph["actual_upstream"] != expected["upstream"]:
            reasons.append("missing_upstream")
        if graph["actual_downstream_count"] != expected["downstream_count"]:
            reasons.append("orphan_output")

    return Verdict(alert=bool(reasons), pillar="lineage", reason=", ".join(reasons))


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _tool_error(drift):
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    sigma_max = ctx.baseline["feature_mean_shift_sigma_max"] * FEATURE_SIGMA_MULT
    if drift["mean_shift_sigma"] > sigma_max:
        return Verdict(alert=True, pillar="ai_infra", reason="feature_skew")
    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _tool_error(drift):
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    b = ctx.baseline
    reasons = []
    if drift["centroid_shift"] > b["embedding_centroid_shift_max"] * EMBEDDING_CENTROID_MULT:
        reasons.append("embedding_drift")
    if drift["avg_doc_age_days"] > b["corpus_avg_doc_age_days_max"] * CORPUS_AGE_MULT:
        reasons.append("corpus_staleness")

    return Verdict(alert=bool(reasons), pillar="ai_infra", reason=", ".join(reasons))
