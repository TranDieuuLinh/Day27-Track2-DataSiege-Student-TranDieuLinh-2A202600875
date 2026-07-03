# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

The subtle **ai_infra** faults were the hardest. On the public stream, a few
embedding-drift and corpus-staleness instances sat just inside the published
3σ baselines (centroid shift ~0.04 vs. a 0.0435 cap; doc age ~48 days vs. a
~50-day cap). **Feature skew** had the opposite problem: clean materializations
sometimes landed barely above the sigma threshold, so a naive baseline check
false-alarmed while still missing the near-boundary embedding faults.

**Distribution shifts** on `data_batch` were the next challenge — mean amount
can drift upward without crossing `mean_amount_max`, especially when row count
and null rate still look normal. Contract and lineage faults were more
straightforward: schema/type violations are explicit in `contract_diff`, and
missing upstream edges or zero downstream outputs are structural signals once
a healthy lineage topology is learned from early clean runs.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

I kept the strategy **one metered tool call per event** (cost ~180 on practice,
~240 on public vs. a 220 budget). That is slightly over budget on the longer
public stream but still cheaper than doubling up tools (e.g. `lineage_graph_slice`
at depth 2) on every event. The TPR gain from extra calls did not look worth the
overage penalty in simulation.

If I had another pass, I would spend budget more selectively rather than
uniformly: keep full checks on contract and lineage events (cheap, high signal),
and for **ai_infra** consider a two-stage rule — alert on obvious centroid/age
breaches first, then only re-check borderline embedding batches when
`budget_remaining()` allows. I would also track a rolling window of recent
feature-shift values in `ctx.state` to separate “this batch is high but normal
for this view” from true training-serving skew, instead of relying on a single
static sigma multiplier.
