# B2: supported-row attrition

Label: **exploratory, post hoc, not part of the frozen protocol**.

A held-out row is evaluated only when its target group has at least three training rows and nonzero training scale in that outer fold. The analysis enumerates which rows and systems lose one or all held-out opportunities under repeated CV5 and LOSO.

No model is fitted. Suite, submitter, accelerator family, workload, scenario, and model characteristics come from the canonical 476-row MLPerf v6.1 Closed-division dataset.
