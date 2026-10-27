"""Module 6 structured supplier-risk verdicts and the deterministic judge.

Importing this package imports DSPy and the Module 5 orchestration module but
creates no provider client, reads no environment variable, opens no database,
and loads no embedding model. The assignment-facing entry point is
:func:`decide_supplier`.
"""

from supplier_risk.decision.errors import (
    DecisionError,
    DecisionInputError,
    DecisionModelError,
    DecisionResponseError,
    SupplierMismatchError,
)
from supplier_risk.decision.judge import (
    BANNED_TERMS,
    JudgeResult,
    JudgeRule,
    banned_terms_in,
    judge_payload,
    judge_verdict,
)
from supplier_risk.decision.lm import (
    LM_ROLES,
    LMRole,
    RoleLanguageModels,
    build_role_language_models,
    dspy_model_name,
)
from supplier_risk.decision.module import VerdictGenerator
from supplier_risk.decision.pipeline import decide_supplier
from supplier_risk.decision.result import DecisionResult
from supplier_risk.decision.signature import SupplierRiskSignature
from supplier_risk.decision.verdict import (
    MAX_FINDINGS,
    MAX_RATIONALE_CHARS,
    VERDICT_FIELDS,
    VERDICT_VALUES,
    SupplierRiskVerdict,
    Verdict,
    parse_supplier_identity,
    parse_verdict_payload,
    require_matching_supplier,
)

__all__ = [
    "BANNED_TERMS",
    "LM_ROLES",
    "MAX_FINDINGS",
    "MAX_RATIONALE_CHARS",
    "VERDICT_FIELDS",
    "VERDICT_VALUES",
    "DecisionError",
    "DecisionInputError",
    "DecisionModelError",
    "DecisionResponseError",
    "DecisionResult",
    "JudgeResult",
    "JudgeRule",
    "LMRole",
    "RoleLanguageModels",
    "SupplierMismatchError",
    "SupplierRiskSignature",
    "SupplierRiskVerdict",
    "Verdict",
    "VerdictGenerator",
    "banned_terms_in",
    "build_role_language_models",
    "decide_supplier",
    "dspy_model_name",
    "judge_payload",
    "judge_verdict",
    "parse_supplier_identity",
    "parse_verdict_payload",
    "require_matching_supplier",
]
