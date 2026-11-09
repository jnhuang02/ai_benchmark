#!/usr/bin/env python3
"""Write the Stage 3A thesis plan and approved claim inventory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
FINAL = ROOT / "results" / "phase2-final-f51e57a"
CORRECTIONS = ROOT / "results" / "phase2-5-corrections-e97bb93"
ANALYSIS = ROOT / "results" / "phase2-5-analysis-4faac06"
SOURCE_MAP = THESIS / "phase2_5_number_source_map.csv"
NA_REGISTER = THESIS / "phase2_5_na_register.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=THESIS / "stage3a_plan_and_claims_report.md",
    )
    return parser.parse_args()


def f(value: float) -> str:
    return f"{float(value):.6f}"


def lookup_pair(
    frame: pd.DataFrame, protocol: str, left: str, right: str
) -> pd.Series:
    return frame.loc[
        frame["protocol"].eq(protocol)
        & frame["left"].eq(left)
        & frame["right"].eq(right)
    ].iloc[0]


def map_ids(
    source_map: pd.DataFrame,
    context: str,
    columns: tuple[str, ...] | None = None,
) -> str:
    local = source_map.loc[
        source_map["context"].str.contains(context, case=False, regex=False)
    ]
    if columns is not None:
        local = local.loc[local["source_column"].isin(columns)]
    if local.empty:
        raise AssertionError(f"No source-map rows for context: {context}")
    return ", ".join(local.sort_values("map_id")["map_id"].tolist())


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(FINAL / "primary_decision.csv")
    secondary = pd.read_csv(FINAL / "secondary_holm.csv")
    cv = pd.read_csv(FINAL / "cv5_main_table.csv").set_index("method")
    loso = pd.read_csv(FINAL / "loso_main_table.csv").set_index("method")
    sensitivity = pd.read_csv(
        FINAL / "sensitivity_main_table.csv"
    ).set_index("method")
    tails = pd.read_csv(ANALYSIS / "tail_summary.csv")
    support = pd.read_csv(CORRECTIONS / "loafo_family_support.csv")
    comparisons = pd.read_csv(
        CORRECTIONS / "loafo_group_median_comparisons.csv"
    )
    headline_counts = pd.read_csv(
        CORRECTIONS / "headline_claim_counts.csv"
    ).set_index("quantity")
    source_map = pd.read_csv(SOURCE_MAP, keep_default_na=False)
    na_register = pd.read_csv(NA_REGISTER, keep_default_na=False)

    cv_primary = lookup_pair(
        primary, "cv5_partitioned", "best_gnn", "mlp"
    )
    loso_primary = lookup_pair(primary, "loso", "best_gnn", "mlp")
    sensitivity_primary = lookup_pair(
        primary, "sensitivity", "best_gnn", "mlp"
    )
    gs_rf_cv = lookup_pair(
        secondary, "cv5_partitioned", "graphsage", "random_forest"
    )
    gs_rf_loso = lookup_pair(
        secondary, "loso", "graphsage", "random_forest"
    )
    shuffle_cv = lookup_pair(
        secondary, "cv5_partitioned", "graphsage", "graphsage_shuffled"
    )
    shuffle_loso = lookup_pair(
        secondary, "loso", "graphsage", "graphsage_shuffled"
    )
    gcn_cv = lookup_pair(
        secondary, "cv5_partitioned", "gcn", "graphsage"
    )
    gcn_loso = lookup_pair(
        secondary, "loso", "gcn", "graphsage"
    )
    gat_cv = lookup_pair(
        secondary, "cv5_partitioned", "gat", "graphsage"
    )
    gat_loso = lookup_pair(
        secondary, "loso", "gat", "graphsage"
    )
    headline_better = headline_counts.loc[
        "best_gnn_point_better_than_mlp"
    ]
    mi355 = comparisons.loc[
        comparisons["family_id"].eq("amd_cdna4_mi355x")
    ]
    mi355_gm = mi355.loc[mi355["method"].eq("group_median")].iloc[0][
        "group_median_error"
    ]
    trained = comparisons.loc[~comparisons["method"].eq("group_median")]
    better_count = (
        trained.groupby("family_id", as_index=False)
        .agg(
            trained_models=("method", "nunique"),
            trained_better_than_gm=(
                "model_better_than_group_median",
                "sum",
            ),
        )
    )
    rtsp = better_count.loc[
        better_count["family_id"].eq("nvidia_blackwell_rtx_pro")
    ].iloc[0]
    rtsp_gm_better = int(rtsp["trained_models"] - rtsp["trained_better_than_gm"])
    cv_rf = tails.loc[
        tails["protocol"].eq("cv5_partitioned")
        & tails["method"].eq("random_forest")
    ].iloc[0]
    cv_mlp = tails.loc[
        tails["protocol"].eq("cv5_partitioned")
        & tails["method"].eq("mlp")
    ].iloc[0]
    loso_rf = tails.loc[
        tails["protocol"].eq("loso")
        & tails["method"].eq("random_forest")
    ].iloc[0]
    loso_mlp = tails.loc[
        tails["protocol"].eq("loso")
        & tails["method"].eq("mlp")
    ].iloc[0]
    interval_columns = (
        "mean_difference",
        "difference_95ci_low",
        "difference_95ci_high",
    )
    c1_ids = map_ids(
        source_map,
        "cv5_partitioned: best_gnn minus mlp",
        interval_columns,
    )
    c2_ids = map_ids(
        source_map,
        "loso: best_gnn minus mlp",
        interval_columns,
    )
    c3_ids = map_ids(
        source_map,
        "sensitivity: best_gnn minus mlp",
        interval_columns,
    )
    c4_ids = ", ".join(
        [
            map_ids(
                source_map,
                "cv5_partitioned: graphsage minus random_forest",
                ("mean_difference", "holm_adjusted_p_value"),
            ),
            map_ids(
                source_map,
                "loso: graphsage minus random_forest",
                ("mean_difference", "holm_adjusted_p_value"),
            ),
        ]
    )
    c5_ids = ", ".join(
        [
            map_ids(
                source_map,
                "cv5_partitioned: graphsage minus graphsage_shuffled",
                ("mean_difference", "holm_adjusted_p_value"),
            ),
            map_ids(
                source_map,
                "loso: graphsage minus graphsage_shuffled",
                ("mean_difference", "holm_adjusted_p_value"),
            ),
        ]
    )
    c6_ids = ", ".join(
        [
            map_ids(
                source_map,
                "cv5_partitioned: gcn minus graphsage",
                ("mean_difference",),
            ),
            map_ids(
                source_map,
                "loso: gcn minus graphsage",
                ("mean_difference",),
            ),
        ]
    )
    c7_ids = ", ".join(
        [
            map_ids(
                source_map,
                "cv5_partitioned: gat minus graphsage",
                ("mean_difference", "holm_adjusted_p_value"),
            ),
            map_ids(
                source_map,
                "loso: gat minus graphsage",
                ("mean_difference", "holm_adjusted_p_value"),
            ),
        ]
    )
    c9_ids = map_ids(source_map, "LOAFO non-mixed families")
    c10_ids = ", ".join(
        [
            map_ids(
                source_map,
                "LOAFO headline claim count best_gnn_point_better_than_mlp",
            ),
            map_ids(
                source_map,
                "LOAFO headline claim count headline_families",
            ),
            map_ids(
                source_map,
                "LOAFO family support intel_xe2_battlemage_arc_pro_b70",
                ("supported_systems",),
            ),
            map_ids(
                source_map,
                (
                    "LOAFO headline families where group median beats "
                    "every trained model"
                ),
            ),
        ]
    )
    c11_ids = ", ".join(
        [
            map_ids(
                source_map,
                "cv5_partitioned random_forest tail p95",
            ),
            map_ids(
                source_map,
                "cv5_partitioned mlp tail p95",
            ),
            map_ids(source_map, "loso random_forest tail median"),
            map_ids(source_map, "loso mlp tail p95"),
        ]
    )
    c12_ids = ", ".join(
        [
            map_ids(source_map, "CV5 top normalized residual, row 0, abs_z"),
            map_ids(source_map, "CV5 top normalized residual, row 1, abs_z"),
        ]
    )
    claim_ids = set(
        re.findall(
            r"N\d{4}",
            ", ".join(
                [
                    c1_ids,
                    c2_ids,
                    c3_ids,
                    c4_ids,
                    c5_ids,
                    c6_ids,
                    c7_ids,
                    c9_ids,
                    c10_ids,
                    c11_ids,
                    c12_ids,
                ]
            ),
        )
    )
    do_not_cite_ids = set(
        na_register.loc[
            na_register["na_decision"].str.startswith("DO_NOT_CITE"),
            "map_id",
        ]
    )
    cited_do_not_cite = sorted(claim_ids & do_not_cite_ids)
    if cited_do_not_cite:
        raise AssertionError(
            f"Thesis claims cite do-not-cite rows: {cited_do_not_cite}"
        )
    na_counts = (
        na_register["na_decision"]
        .value_counts()
        .rename_axis("decision")
        .reset_index(name="count")
    )

    lines = [
        "# Stage 3A Plan and Claims Report",
        "",
        "Status: planning and claim approval only. No thesis LaTeX prose, "
        "abstract, table, figure, bibliography, or PDF was edited. Existing "
        "uncommitted thesis files were preserved.",
        "",
        "## 1. What was done",
        "",
        "- Audited the accepted Phase 2.5 evidence against the proposed Phase "
        "3 scope.",
        "- Defined the thesis title, framing, research questions, terminology, "
        "contributions, claim boundaries, chapter restructuring, and "
        "table/figure placement.",
        "- Prepared exact LOAFO and tail-summary text requirements for Stage 3B.",
        "- Kept the original Table 6.4, Phase 0 diagnostics, and historical "
        "run fragments outside the result claims.",
        "- Stopped before thesis editing.",
        "",
        "## 2. Proposed thesis plan",
        "",
        "### 2.1 Proposed title",
        "",
        "**Predicting MLPerf Inference Performance for Unseen Systems: A "
        "Comparison of Graph and Tabular Models**",
        "",
        "The title uses \"inference workloads\" throughout. Claims are not "
        "restricted to LLMs.",
        "",
        "### 2.2 Full abstract",
        "",
        "Predicting the performance of machine-learning systems before they "
        "are benchmarked is difficult because published results combine "
        "hardware, software, workload, and submitter-specific implementation "
        "choices. This dissertation evaluates whether system and workload "
        "metadata can predict out-of-sample MLPerf Inference v6.1 performance "
        "and whether explicit graph structure improves that prediction. The "
        "analysis uses 476 published Closed-division measurements from 105 "
        "submitter-aware systems in the Datacenter and Edge suites.",
        "",
        "The evaluation follows a pre-specified protocol frozen by commit hash "
        "before the Phase 2 model runs. The protocol defines grouped five-fold "
        "cross-validation over five partition seeds, leave-one-system-out "
        "validation, supported workload groups, system-clustered uncertainty, "
        "an equivalence margin, and no model tuning. The model ladder contains "
        "group-median, Ridge, random-forest, gradient-boosting, hierarchical "
        "mixed, MLP, GraphSAGE, GCN, and GAT models. All models use the same "
        "supported rows and the same log-transformed accelerator-count "
        "features. Test systems enter only the inference graph after model "
        "weights are fixed.",
        "",
        f"Random forest has the lowest repeated-CV primary error "
        f"(`{f(cv.loc['random_forest', 'primary_error_mean'])}`) and the lowest "
        f"LOSO primary error (`{f(loso.loc['random_forest', 'primary_error'])}`). "
        f"The corresponding MLP errors are "
        f"`{f(cv.loc['mlp', 'primary_error_mean'])}` and "
        f"`{f(loso.loc['mlp', 'primary_error'])}`. The sole primary hypothesis "
        f"compares the validation-selected best GNN with MLP. Its repeated-CV "
        f"difference is `+{f(cv_primary.mean_difference)}`, 95% CI "
        f"`[{f(cv_primary.difference_95ci_low)}, "
        f"{f(cv_primary.difference_95ci_high)}]`; its LOSO difference is "
        f"`+{f(loso_primary.mean_difference)}`, 95% CI "
        f"`[{f(loso_primary.difference_95ci_low)}, "
        f"{f(loso_primary.difference_95ci_high)}]`. Both intervals exclude "
        f"zero but cross the frozen `+0.05` boundary, so the combined "
        f"co-primary label is `inconclusive`.",
        "",
        f"Secondary comparisons show GraphSAGE worse than random forest under "
        f"both protocols and GAT worse than GraphSAGE under both protocols "
        f"after Holm correction. Real-edge GraphSAGE minus shuffled-edge "
        f"GraphSAGE is `+{f(shuffle_cv.mean_difference)}` under repeated CV, "
        f"labeled `{shuffle_cv.status}`, and "
        f"`{f(shuffle_loso.mean_difference)}` under LOSO, labeled "
        f"`{shuffle_loso.status}`. GCN minus GraphSAGE is "
        f"`{f(gcn_cv.mean_difference)}` under repeated CV, labeled "
        f"`{gcn_cv.status}`, and `+{f(gcn_loso.mean_difference)}` under LOSO, "
        f"labeled `{gcn_loso.status}`.",
        "",
        "The leave-one-accelerator-family-out stress test is descriptive "
        "because the frozen protocol contains no rule for pooling family "
        "errors. Across families, trained models do not reliably beat the "
        "group-median baseline. Best GNN has a lower primary-error point "
        "estimate than MLP in only 2 of 6 headline families; one of those two "
        "families, Intel Arc Pro B70, has only 3 supported systems and no "
        "reportable system-clustered interval. On MI355X, the group median has "
        "lower error than every trained model. Tail analysis "
        "also shows that model rankings depend on the statistic considered: "
        "random forest has the better median and larger share within ±25%, "
        "while MLP has the lower 90th and 95th percentiles.",
        "",
        "The thesis concludes that explicit heterogeneous graph structure did "
        "not provide a reliable predictive advantage under this evaluation. "
        "The strongest tabular model, not the graph model, supplies the best "
        "overall predictions. The study contributes a leakage-audited, "
        "system-held-out benchmark, a complete model ladder, family-level "
        "extrapolation evidence, and an auditable record of protocol changes "
        "and limitations.",
        "",
        "### 2.3 Framing",
        "",
        "The thesis follows a pre-specified protocol frozen by commit hash "
        "before the Phase 2 model runs. Its primary empirical result is "
        "negative for graph structure under that protocol: the best GNN is "
        "worse than MLP by about 0.10-0.12 log units under the two co-primary "
        "protocols, while the intervals cannot exclude differences as small "
        "as approximately 0.02-0.05 log units because they cross the frozen "
        "`+0.05` equivalence boundary.",
        "",
        "### 2.4 Full conclusion",
        "",
        "This dissertation asked whether published system and workload "
        "metadata can predict MLPerf Inference performance for unseen systems "
        "and whether explicit graph structure adds value over tabular and MLP "
        "models. The evaluation was governed by a pre-specified protocol "
        "frozen by commit hash before the Phase 2 model runs. The primary "
        "result is negative for graph structure: the best GNN has higher "
        "primary error than MLP under both co-primary protocols. Under "
        f"repeated CV the difference is `+{f(cv_primary.mean_difference)}` "
        f"with 95% CI `[{f(cv_primary.difference_95ci_low)}, "
        f"{f(cv_primary.difference_95ci_high)}]`; under LOSO it is "
        f"`+{f(loso_primary.mean_difference)}` with 95% CI "
        f"`[{f(loso_primary.difference_95ci_low)}, "
        f"{f(loso_primary.difference_95ci_high)}]`. Both are labeled "
        "`inconclusive` because their intervals exclude zero but cross the "
        "frozen practical-margin boundary. The data therefore do not support "
        "a claim that graph structure improves held-out-system prediction.",
        "",
        "The model ladder also shows that a standard tabular ensemble is the "
        f"strongest overall learner: random forest reaches `{f(cv.loc['random_forest', 'primary_error_mean'])}` "
        f"under repeated CV and `{f(loso.loc['random_forest', 'primary_error'])}` "
        f"under LOSO. GraphSAGE is worse than random forest after Holm "
        "correction under both protocols, and GAT is worse than GraphSAGE "
        "under both protocols. The shuffled-edge comparison does not provide "
        "a uniform edge-value conclusion: repeated CV is labeled "
        f"`{shuffle_cv.status}` while LOSO is labeled "
        f"`{shuffle_loso.status}`. These protocol-specific results are part "
        "of the conclusion rather than post hoc qualifications.",
        "",
        "The leave-one-family-out analysis exposes important scope limits. "
        "No pooled LOAFO estimate is valid because the frozen protocol has no "
        "pooling rule. The family-level results show that trained models do "
        "not reliably beat the group-median baseline. Best GNN beats MLP on "
        "the point estimate in only 2 of 6 headline families, and one of "
        "those families, Intel Arc Pro B70, has only three supported systems "
        "and therefore no reportable system-clustered interval. Large tail "
        "errors and interactive-throughput outliers further constrain claims "
        "about operational accuracy.",
        "",
        "The contribution is therefore methodological and empirical: a "
        "leakage-audited protocol, a transparent comparison of graph and "
        "tabular models, and a negative finding reported without tuning or "
        "result-dependent revision. Future work should test richer system "
        "metadata, treatment of interactive-throughput extremes, and graph "
        "constructions that are evaluated without exposing held-out systems.",
        "",
        "### 2.5 Research questions",
        "",
        "1. How accurately can published system and workload metadata predict "
        "performance on previously unseen MLPerf Inference systems?",
        "2. Does explicit graph structure add value over tabular and MLP "
        "baselines under the pre-specified protocol frozen by commit hash "
        "before the Phase 2 model runs?",
        "3. Where do the models fail, including workload-family extrapolation, "
        "distribution tails, partition variability, and interactive-throughput "
        "outliers?",
        "",
        "### 2.6 Definitions used once and consistently",
        "",
        "- **Submission:** a publisher's account of one system and its "
        "measurements in a suite.",
        "- **System:** a submitter-aware hardware/software configuration "
        "identified by suite, submitter, system name, and system type.",
        "- **Workload:** the named MLPerf model or benchmark task.",
        "- **Workload group:** the scenario, metric, workload, and suite "
        "combination used for target normalization and ranking.",
        "- **Result cell:** one published scalar performance measurement.",
        "- **Measurement:** synonymous with a result cell when referring to "
        "the numeric response.",
        "",
        "### 2.7 Contributions",
        "",
        "- A leakage-audited, submitter-aware evaluation protocol for 476 "
        "Closed-division measurements from 105 systems.",
        "- A comparison of group median, Ridge, random forest, gradient "
        "boosting, hierarchical mixed, MLP, GraphSAGE, GCN, and GAT models on "
        "shared held-out systems under a pre-specified protocol frozen by "
        "commit hash before the Phase 2 model runs.",
        "- A co-primary repeated-CV and LOSO analysis with system-clustered "
        "uncertainty and a practical equivalence margin.",
        "- A family-level LOAFO stress test that exposes heterogeneity and "
        "failure to reliably beat a simple group-median baseline.",
        "- An audit and reproducibility record covering leakage, protocol "
        "amendments, fold stability, sparse-group exclusions, and saved "
        "number-to-source mappings.",
        "",
        "## 3. Claim inventory",
        "",
        "| Claim ID | Approved claim | Source-map rows | Permitted wording |",
        "| --- | --- | --- | --- |",
        f"| C1 | Best GNN minus MLP under CV5 is `+{f(cv_primary.mean_difference)}`, 95% CI `[{f(cv_primary.difference_95ci_low)}, {f(cv_primary.difference_95ci_high)}]`, frozen label `{cv_primary.status}`. | {c1_ids} | State point estimate, interval, and label together. |",
        f"| C2 | Best GNN minus MLP under LOSO is `+{f(loso_primary.mean_difference)}`, 95% CI `[{f(loso_primary.difference_95ci_low)}, {f(loso_primary.difference_95ci_high)}]`, frozen label `{loso_primary.status}`. | {c2_ids} | State point estimate, interval, and label together; combined co-primary label is `inconclusive`. |",
        f"| C3 | Sensitivity best GNN minus MLP is `+{f(sensitivity_primary.mean_difference)}`, 95% CI `[{f(sensitivity_primary.difference_95ci_low)}, {f(sensitivity_primary.difference_95ci_high)}]`, label `{sensitivity_primary.status}`, but sensitivity is not co-primary. | {c3_ids} | Label as `worse`; do not use it to replace the co-primary decision. |",
        f"| C4 | GraphSAGE minus random forest is `+{f(gs_rf_cv.mean_difference)}` under CV5 and `+{f(gs_rf_loso.mean_difference)}` under LOSO; both are `worse` after Holm. | {c4_ids} | Report both protocols and Holm status. |",
        f"| C5 | Real-edge GraphSAGE minus shuffled-edge GraphSAGE is `+{f(shuffle_cv.mean_difference)}` under CV5, label `{shuffle_cv.status}` (Holm p `{f(shuffle_cv.holm_adjusted_p_value)}`), and `{f(shuffle_loso.mean_difference)}` under LOSO, label `{shuffle_loso.status}` (Holm p `{f(shuffle_loso.holm_adjusted_p_value)}`). | {c5_ids} | Do not say graph edges add value; state the protocol-specific labels. |",
        f"| C6 | GCN minus GraphSAGE is `{f(gcn_cv.mean_difference)}` under CV5, label `{gcn_cv.status}`, and `+{f(gcn_loso.mean_difference)}` under LOSO, label `{gcn_loso.status}`. | {c6_ids} | Report both protocol-specific conclusions. |",
        f"| C7 | GAT minus GraphSAGE is `+{f(gat_cv.mean_difference)}` under CV5 and `+{f(gat_loso.mean_difference)}` under LOSO; both are `worse` after Holm. | {c7_ids} | GAT failed without claiming numerical collapse; use the saved training diagnostics. |",
        f"| C8 | LOAFO comprises 13 descriptive family-level stress tests because Protocol v2.1 has no pooling rule. | {c9_ids} | Never present a pooled LOAFO estimate. |",
        f"| C9 | Trained models did not reliably beat the group-median baseline; best GNN beat MLP on the point estimate in only `{int(headline_better['value'])}` of 6 headline families. One of those families, Intel Arc Pro B70, has only 3 supported systems and no reportable system-clustered interval. | {c10_ids} | Use a descriptive count and family examples; do not infer precision from the B70 point estimate. |",
        f"| C10 | Tail errors remain large and model rankings depend on the chosen tail statistic. | {c11_ids} | Keep the tail table in Chapter 6 main text. |",
        f"| C11 | Two interactive-throughput rows dominate normalized RMSE. | {c12_ids} | Treat this as a limitation of normalized RMSE, not a ranking signal. |",
        "",
        "Provenance guardrail, not a result claim: the exact historical Table "
        "6.4 value is not cited as a thesis result and is discussed only in "
        "the provenance appendix.",
        "",
        "Forbidden primary language: \"no difference,\" \"competitive,\" or a "
        "claim that graph structure is equivalent to MLP.",
        "",
        "## 4. Required Stage 3B content",
        "",
        "### 4.1 LOAFO paragraph",
        "",
        "The Stage 3B LOAFO paragraph must state:",
        "",
        f"> Across held-out accelerator families, trained models did not "
        f"reliably beat the group-median baseline. Best GNN had a lower "
        f"primary-error point estimate than MLP in only "
        f"{int(headline_better['value'])} of 6 headline families; one of "
        f"those two families, Intel Arc Pro B70, has only 3 supported "
        f"systems and no reportable system-clustered interval. On MI355X, "
        f"the group median (`{f(mi355_gm)}`) had a lower error than every "
        f"trained model. On RTX PRO, the group median beat "
        f"{rtsp_gm_better} of "
        f"{int(rtsp['trained_models'])} trained models. These are descriptive "
        f"family-level comparisons; no pooled LOAFO estimate is made, and "
        f"intervals are restricted to families with at least 8 supported "
        f"systems.",
        "",
        "The paragraph must not imply that the LOAFO failures establish a "
        "causal mechanism.",
        "",
        "### 4.2 Tail table",
        "",
        "The RF/MLP/GraphSAGE tail table belongs in the Chapter 6 main text, "
        "not an appendix. It must label the CV5 maximum as the mean of the "
        "five per-partition maxima, define the ±25% and ±50% windows, and note "
        f"that RF has the lower median and larger ±25% share "
        f"(`{float(cv_rf.within_25pct) * 100:.6f}%` versus "
        f"`{float(cv_mlp.within_25pct) * 100:.6f}%` under CV5), while MLP has the "
        f"lower 95th percentile (`{f(cv_mlp.p95)}` versus "
        f"`{f(cv_rf.p95)}`). Under LOSO the same ordering holds: RF median "
        f"`{f(loso_rf['median'])}`, MLP 95th percentile "
        f"`{f(loso_mlp.p95)}` versus RF `{f(loso_rf.p95)}`.",
        "",
        "### 4.3 Chapter and section plan",
        "",
        "1. **Chapter 1: Introduction.** Research questions, one-time "
        "definitions, contributions, and a concise statement of the negative "
        "primary result.",
        "2. **Chapter 2: Related work.** System-performance prediction, "
        "tabular learning, heterogeneous graphs, MLPerf, and evidence gaps.",
        "3. **Chapter 3: Data and prediction task.** v6.1 Closed Datacenter "
        "and Edge scope, result-cell construction, workload groups, system "
        "identity, and data audit.",
        "4. **Chapter 4: Metadata and tabular models.** Features, target "
        "normalization, supported-row rule, deterministic baselines, and MLP.",
        "5. **Chapter 5: Heterogeneous graph models.** Implemented system, "
        "accelerator, processor, workload, and result nodes; containment, "
        "execution, and measurement relations; GraphSAGE, GCN, and GAT.",
        "6. **Chapter 6: Evaluation protocol and results.** Co-primary repeated "
        "CV and LOSO, the primary decision, the model ladder, secondary Holm "
        "comparisons, edge/shuffle and relation-removal ablations, GAT "
        "diagnostic, tail table in main text, descriptive LOAFO in three "
        "non-pooled groups, sensitivity, partition variability, and outlier "
        "limitation. Every table and figure is regenerated from final bundles "
        "with no stale footers.",
        "7. **Chapter 7: Discussion.** Why graph structure did not outperform "
        "tabular models, family heterogeneity, group-median baseline failure, "
        "tail behavior, and the limits of normalized RMSE.",
        "8. **Chapter 8: Conclusion.** Restate the exact co-primary result, "
        "shuffled-edge labels, GAT result, LOAFO caveat, and future work.",
        "",
        "**Appendix moves and deletions**",
        "",
        "| Existing material | Destination |",
        "| --- | --- |",
        "| Docker/LoadGen setup | Appendix A, apparatus |",
        "| Current Table 5.1 | Appendix A, apparatus |",
        "| QSL/QDL architecture | Appendix A, apparatus |",
        "| Open-division test lists | Delete; outside v6.1 Closed scope |",
        "| Versions 5/6 claims, historical runs, rented-GPU claims | Delete from main text |",
        "| DAG/topological analysis | Short future-work subsection in Chapter 7 |",
        "| Pooled MAPE | Appendix C, with singleton-heavy limitation |",
        "| Leakage/audit trail, mixed-run provenance, original 0.71 Spearman | Appendix B, provenance |",
        "| Full number-to-source map and test output | Appendix B/C and reproducibility package |",
        "",
        "**Methods disclosure required in Chapter 6**",
        "",
        "- Phases 0-1 exposed the research team to the same dataset before "
        "the final protocol was frozen.",
        "- The main protocol was pre-specified and frozen by commit hash "
        "before the Phase 2 model runs.",
        "- The v2.2 LOAFO addendum records that models, seeds `7`, `11`, and "
        "`16`, omission of the mixed model, and family-level paired inference "
        "were specified in orchestration rather than in the hash-pinned file.",
        "- The compute erratum corrects repeated CV to 25 outer folds and "
        "sensitivity to 5 outer folds.",
        "- The shrinkage secondary `S3A-SHRINK-001` was not executed because "
        "the pinned definition does not identify a unique posterior.",
        "- LOAFO has no pooled estimate; headline, small/integrated, and "
        "CPU-only results are reported separately and descriptively.",
        "- No model was tuned, LOSO uses three neural seeds while the other "
        "protocols use ten, and equivalence decisions use the frozen ±0.05 "
        "log-unit margin.",
        "",
        "### 4.4 NA-register and draft-text audit",
        "",
        "| NA decision | Count |",
        "| --- | ---: |",
    ]
    for row in na_counts.itertuples(index=False):
        lines.append(f"| `{row.decision}` | {row.count} |")
    lines.extend(
        [
            "",
            "The claim inventory cites no source-map row marked "
            "`DO_NOT_CITE` or `DO_NOT_CITE_NULL_CELL`. A source-wide search "
            "for conversational fragments and unfinished draft markers found "
            "none in the current reports. The word `placeholder` appears only in "
            "the intentional instruction to mark dedication, acknowledgments, "
            "and vita placeholders. Audited thesis sections—title, abstract, "
            "conclusion, Chapters 1-8, and appendices—cite no number marked "
            "`DO_NOT_CITE` or `DO_NOT_CITE_NULL_CELL`. Outside verbatim frozen "
            "protocol blocks, active reports use the commit-hash-frozen "
            "protocol wording requested for Stage 3B.",
            "",
            "### 4.5 Placeholders and final checks",
            "",
            "Dedication, acknowledgments, and vita remain visibly marked as "
            "placeholders until author text is supplied. Every result number "
            "must map to a saved file and column. Before Phase 3 completion, "
            "cross-references and bibliography must compile, Section 5.1.1 "
            "software versions must match the lockfile, and a fresh clone "
            "must reproduce the main table through documented commands.",
            "",
            "## 5. Open questions",
            "",
            "- Does the orchestrator approve the proposed title and claim "
            "inventory before Stage 3B edits LaTeX?",
            "- Should the title name `v6.1` explicitly, or keep the version "
            "in the abstract and data chapter?",
            "- Does the author want the optional semantic exploratory block "
            "supplied separately? It remains outside the main claims and was "
            "not started.",
            "",
            "## 6. Acceptance criteria",
            "",
            "| Item | Met? | Evidence |",
            "| --- | --- | --- |",
            "| Pre-specified frozen-protocol wording and new title proposed | Yes | Section 2 |",
            "| Full abstract pasted | Yes | Section 2.2 |",
            "| Full conclusion pasted | Yes | Section 2.4 |",
            "| Chapter outline with appendix moves and deletions | Yes | Section 4.3 |",
            "| Claim inventory maps every claim to source-map rows | Yes | Section 3 |",
            "| Methods disclose Phases 0-1 exposure, v2.2 LOAFO addendum, compute erratum, shrinkage deviation, and no LOAFO pooling | Yes | Section 4.3 |",
            "| Supported-system counts and B70 caveat included | Yes | C9 and Section 4.1 |",
            "| NA counts and do-not-cite citation audit included | Yes | Section 4.4 |",
            "| Draft-text audit completed | Yes | Section 4.4 |",
            "| Research questions, definitions, and contributions planned | Yes | Sections 2.5-2.7 |",
            "| Negative primary result stated with estimate, interval, and label | Yes | C1-C2 |",
            "| Shuffled-edge and GAT results assigned exact language | Yes | C5 and C7 |",
            "| LOAFO descriptive paragraph supplied | Yes | Section 4.1 |",
            "| Tail table placed in Chapter 6 main text | Yes | Section 4.2 |",
            "| Original Table 6.4 excluded as a result | Yes | provenance guardrail |",
            "| Chapter restructuring mapped | Yes | Section 4.3 |",
            "| Thesis LaTeX left untouched pending approval | Yes | No LaTeX edit performed |",
            "| Stopped after Stage 3A | Yes | No Stage 3B work started |",
            "",
            "**STOP.** Stage 3B must not begin without orchestrator approval "
            "of this plan and claim inventory.",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
