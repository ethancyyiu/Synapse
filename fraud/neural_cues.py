"""Observable purchase-history cues for the separate synthetic fly experiment."""
from __future__ import annotations

import math

CUES = (
    "Isolated or established activity",
    "Recent purchase cluster",
    "Large deviation with novelty or a cluster",
)


def neural_cue(features: dict) -> dict:
    """Map existing numeric features to a cue, without inferring a fraud label."""
    large = features["log_amount_ratio"] >= math.log1p(3)
    cluster = features["recent_count"] >= 2
    novel = bool(features["new_merchant"])
    if large and (novel or cluster):
        index = 2
        why = "Amount exceeds 3x the historical median with merchant novelty or recent activity."
    elif cluster:
        index = 1
        why = "At least two earlier purchases occurred in the prior 24 hours."
    else:
        index = 0
        why = "Neither the large-deviation rule nor the recent-cluster rule matched."
    return {"index": index, "name": CUES[index], "basis": why,
            "synthetic_rule_only": True}
