"""Teacher/student language-model roles for the decision layer.

The assignment requires a stronger teacher model and a cheaper student model
(Guide Step 5). Module 6 builds both roles but uses neither by itself: the caller
supplies the LM for one decision, for example::

    settings = load_language_model_settings()
    models = build_role_language_models(settings)
    with dspy.context(lm=models.student):
        result = decide_supplier("SUP-007")

Nothing here calls a provider, changes global DSPy settings, or reads the real
``.env`` unless the caller asks the existing config loader to. Global
``dspy.configure`` is deliberately not used so configuration cannot leak between
tests or requests, and the teacher is never invoked for an ordinary student
prediction; it exists for the Module 7 optimizer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import dspy

from supplier_risk.config import ConfigError, LanguageModelSettings

LMRole = Literal["teacher", "student"]
LM_ROLES: Final[tuple[LMRole, ...]] = ("teacher", "student")

# ``GEMINI_*`` model values such as ``gemini-2.0-flash`` are completed to the
# LiteLLM/DSPy provider form. A value that already names a provider, for example
# ``gemini/gemini-2.0-flash`` or ``openai/gpt-4o-mini``, is used unchanged.
DEFAULT_PROVIDER_PREFIX: Final = "gemini/"


def dspy_model_name(configured_model: str) -> str:
    """Return the DSPy/LiteLLM model identifier for a configured model name.

    Raises:
        ConfigError: if the configured identifier is blank.
    """
    text = configured_model.strip()
    if not text:
        raise ConfigError("language model identifier must not be blank")
    return text if "/" in text else f"{DEFAULT_PROVIDER_PREFIX}{text}"


@dataclass(frozen=True, slots=True)
class RoleLanguageModels:
    """The teacher and student DSPy LMs, kept as distinct roles.

    ``masked()`` is safe to log; it never contains an API key.
    """

    teacher: object
    student: object

    def for_role(self, role: LMRole) -> object:
        """Return the LM for ``role``.

        Raises:
            ValueError: if ``role`` is not ``"teacher"`` or ``"student"``.
        """
        if role == "teacher":
            return self.teacher
        if role == "student":
            return self.student
        raise ValueError(f"unknown language-model role: {role!r}")

    def masked(self) -> dict[str, str]:
        """Return the resolved model identifiers without any credential."""
        return {
            "teacher_model": str(getattr(self.teacher, "model", "<unknown>")),
            "student_model": str(getattr(self.student, "model", "<unknown>")),
        }


def build_role_language_models(
    settings: LanguageModelSettings,
    *,
    lm_factory: Callable[..., object] | None = None,
) -> RoleLanguageModels:
    """Build one LM per role from validated settings.

    Args:
        settings: Validated language-model configuration.
        lm_factory: Optional constructor used instead of ``dspy.LM``. Tests
            inject a fake factory so no provider client is created.

    Returns:
        The teacher and student LMs.

    Raises:
        ConfigError: if a configured model identifier is blank.
    """
    factory: Callable[..., object] = dspy.LM if lm_factory is None else lm_factory
    api_key = settings.api_key.get_secret_value()
    teacher = factory(model=dspy_model_name(settings.teacher_model), api_key=api_key)
    student = factory(model=dspy_model_name(settings.student_model), api_key=api_key)
    return RoleLanguageModels(teacher=teacher, student=student)
