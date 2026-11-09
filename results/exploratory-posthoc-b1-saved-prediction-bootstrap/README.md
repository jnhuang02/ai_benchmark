# B1: saved-prediction GNN-minus-MLP bootstrap

Label: **exploratory, post hoc, not part of the frozen protocol**.

This analysis does not refit any model. It replays paired system-clustered bootstrap draws already computed from the final saved CV5 and LOSO prediction bundles, and verifies the observed point estimates directly from those predictions.

GraphSAGE and GCN are fixed architecture candidates; this comparison is not a replacement for the frozen best-GNN-versus-MLP co-primary hypothesis.
