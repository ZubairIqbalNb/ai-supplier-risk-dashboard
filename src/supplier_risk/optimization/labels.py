"""Deterministic expected-verdict labels for Module 7 evaluation.

The Module 2 synthetic dataset carries raw evidence and partition membership but
**no verdict label**: it has no ``verdict``, ``risk``, or ``label`` field. Module 6
introduced the ``APPROVE`` / ``FLAG`` verdict vocabulary and a deterministic
*validator* (the judge), but not a labeler. Module 7 therefore needs an explicit,
documented way to say which verdict a supplier's own evidence supports, so that
"correct verdict", "accuracy", and per-class precision/recall/F1 are defined.

That mapping is a **Module 7 provisional engineering choice**, in the same
register as the other provisional decisions in ``docs/schema.md``. It is a total,
deterministic function of the raw record only. It is not derived from retrieval
outputs, from any model, or from the held-out partition's evaluation results, so
using it cannot introduce data leakage.

Policy (conservative — a supplier is approved only when its own numbers are
clearly comfortable):

- ``FLAG`` when the supplier has at least one ``HIGH`` severity complaint.
- ``FLAG`` when ``defect_rate`` exceeds :data:`FLAG_DEFECT_RATE_THRESHOLD`.
- ``FLAG`` when ``on_time_delivery_rate`` falls below
  :data:`FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD`.
- ``APPROVE`` otherwise.

The thresholds are fixed constants, not tuned on any partition. ``LABEL_POLICY_VERSION``
is recorded in the compiled-student metadata so a change to the policy is
detectable.
"""

from __future__ import annotations

from typing import Final

from supplier_risk.contracts import SupplierRecord
from supplier_risk.decision.verdict import Verdict

# Bump when the policy below changes in any way that could alter a label.
LABEL_POLICY_VERSION: Final = "module7.labels.v1"

# High-severity complaints, a defect rate above this fraction, or an on-time
# delivery rate below this fraction all flag a supplier.
FLAG_SEVERITIES: Final[tuple[str, ...]] = ("HIGH",)
FLAG_DEFECT_RATE_THRESHOLD: Final = 0.02
FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD: Final = 0.85


def expected_verdict(record: SupplierRecord) -> Verdict:
    """Return the deterministic expected verdict for one supplier record.

    Args:
        record: A validated Module 2 supplier record. Partition membership is
            not read; the label is a property of the supplier's own evidence.

    Returns:
        ``"FLAG"`` when the record carries high-severity, high-defect, or
        low-on-time evidence; ``"APPROVE"`` otherwise.
    """
    metrics = record.delivery_metrics
    if any(complaint.severity in FLAG_SEVERITIES for complaint in record.complaints):
        return "FLAG"
    if metrics.defect_rate > FLAG_DEFECT_RATE_THRESHOLD:
        return "FLAG"
    if metrics.on_time_delivery_rate < FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD:
        return "FLAG"
    return "APPROVE"


def label_policy_description() -> dict[str, object]:
    """Return the label policy as deterministic, serializable metadata."""
    return {
        "policy_version": LABEL_POLICY_VERSION,
        "flag_severities": list(FLAG_SEVERITIES),
        "flag_defect_rate_threshold": FLAG_DEFECT_RATE_THRESHOLD,
        "flag_on_time_delivery_rate_threshold": FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD,
    }
