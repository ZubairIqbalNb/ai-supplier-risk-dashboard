"""DSPy Signature for the Module 6 supplier-risk verdict.

The signature has exactly three evidence-context inputs — SQL, FAISS, and GRAPH —
matching the three independent Module 5 stores. The three-source contract is
deliberate: adding a fourth context would let retrieval provenance collapse, so
it is enforced by tests rather than left to convention.

Only module-level facts are given to the model. Retrieval, identity resolution,
contract validation, and business rules stay in code; the Signature decides.
"""

from __future__ import annotations

import dspy

from supplier_risk.decision.verdict import Verdict


class SupplierRiskSignature(dspy.Signature):
    """Decide whether one supplier should be APPROVED or FLAGGED for continued
    sourcing, using only the three supplied evidence contexts.

    The inputs are exactly three labeled evidence contexts for the same supplier:

    - sql_context: SQLite delivery and defect metrics tagged ``[SQL]``, or a
      tagged ``[SQL ERROR]`` marker when the store could not be queried.
    - faiss_context: contract and complaint excerpts tagged ``[FAISS]``, or a
      tagged ``[FAISS ERROR]`` marker.
    - graph_context: subsidiary and joint-venture ownership evidence tagged
      ``[GRAPH]``, or a tagged ``[GRAPH ERROR]`` marker.

    Rules:

    1. Use only the supplied evidence. Never invent supplier facts, metrics,
       contracts, complaints, relationships, dates, identifiers, or citations.
    2. Separate adverse evidence, favorable evidence, and missing evidence.
    3. A source tagged ``[SQL ERROR]``, ``[FAISS ERROR]``, or ``[GRAPH ERROR]``
       is missing evidence. A failed source is never evidence that the supplier
       has no risk, and it must never be silently dropped: name it in
       ``missing_evidence``.
    4. If the sources conflict, or an empty result could mean either "none
       found" or "not collected", say which source is affected and treat the
       conflict as uncertainty.
    5. Be conservative. Prefer FLAG when the evidence is incomplete, ambiguous,
       or silent on a material point, and state what is missing.
    6. APPROVE only when all three sources were queried and the supplied
       evidence supports continued sourcing. FLAG is the default under
       uncertainty.
    7. Do not assert sanctions, blacklisting, guarantees, legal control, or the
       absence of risk. The three stores cannot establish those claims.
    8. The rationale must be a concise, decision-oriented explanation grounded in
       the supplied evidence, not a transcript of internal reasoning. Each
       finding should name the source label and the metric, document id, or
       relationship it comes from.
    """

    sql_context: str = dspy.InputField(
        desc=(
            "Labeled [SQL] structured delivery/defect evidence for the supplier, or a "
            "[SQL ERROR] marker. Keep the label and every error marker intact."
        )
    )
    faiss_context: str = dspy.InputField(
        desc=(
            "Labeled [FAISS] contract and complaint excerpts for the supplier, or a "
            "[FAISS ERROR] marker. Keep the label and every error marker intact."
        )
    )
    graph_context: str = dspy.InputField(
        desc=(
            "Labeled [GRAPH] subsidiary and joint-venture ownership evidence for the "
            "supplier, or a [GRAPH ERROR] marker. Keep the label and every error marker intact."
        )
    )

    supplier_id: str = dspy.OutputField(
        desc=(
            "The supplier id (SUP-NNN) exactly as stated inside the supplied contexts. "
            "Repeat it unchanged; never infer a different supplier."
        )
    )
    verdict: Verdict = dspy.OutputField(
        desc="APPROVE or FLAG, uppercase. APPROVE only when the supplied evidence supports it."
    )
    rationale: str = dspy.OutputField(
        desc=(
            "Concise explanation of the decision, at most 800 characters, grounded only in "
            "the supplied evidence and naming the source labels used."
        )
    )
    findings: list[str] = dspy.OutputField(
        desc=(
            "One to eight material evidence items, each a short sentence that names its "
            "source label and metric, document, or relationship."
        )
    )
    missing_evidence: list[str] = dspy.OutputField(
        desc=(
            "Every failed, absent, or incomplete source that materially affects the "
            "decision, one item per gap. Empty only when all three sources returned evidence."
        )
    )
