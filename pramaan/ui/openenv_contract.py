"""UI contract card generated from the OpenEnv models and result type."""

from __future__ import annotations

from dataclasses import fields
from html import escape

from openenv.core.client_types import StepResult

from pramaan_env import PramaanAction, PramaanObservation
from pramaan_env.server.environment import PramaanEnvironment


def _field_rows(model: type) -> str:
    rows = []
    for name, field in model.model_fields.items():
        annotation = getattr(field, "annotation", None)
        type_name = getattr(annotation, "__name__", str(annotation))
        description = getattr(field, "description", None) or ""
        rows.append(
            "<tr><th scope=\"row\"><code>" + escape(name) + "</code></th>"
            "<td><code>" + escape(type_name) + "</code></td>"
            "<td>" + escape(description) + "</td></tr>"
        )
    return "".join(rows)


def _step_result_rows() -> str:
    rows = []
    for field in fields(StepResult):
        type_name = getattr(field, "type", None)
        rows.append(
            "<li><code>" + escape(field.name) + "</code> · <code>"
            + escape(str(type_name)) + "</code></li>"
        )
    return "".join(rows)


def render_openenv_contract() -> str:
    """Render action, observation, step result, and reward/episode contract."""
    return (
        '<article class="pramaan-ui-card pramaan-openenv-contract" data-testid="openenv-contract">'
        '<p class="pramaan-ui-eyebrow">Environment contract</p><h3>What an agent sends and receives</h3>'
        '<h4>Action · <code>PramaanAction</code></h4>'
        '<div class="pramaan-ui-table-wrap"><table><thead><tr><th scope="col">Field</th>'
        '<th scope="col">Type</th><th scope="col">Description</th></tr></thead><tbody>'
        + _field_rows(PramaanAction) + '</tbody></table></div>'
        '<h4>Observation · <code>PramaanObservation</code></h4>'
        '<div class="pramaan-ui-table-wrap"><table><thead><tr><th scope="col">Field</th>'
        '<th scope="col">Type</th><th scope="col">Description</th></tr></thead><tbody>'
        + _field_rows(PramaanObservation) + '</tbody></table></div>'
        '<h4><code>StepResult</code> fields</h4><ul>' + _step_result_rows() + '</ul>'
        f'<p>Each episode allows at most {PramaanEnvironment.MAX_ATTEMPTS} attempts. '
        'Reward is 0 for unreadable or rejected input, 0.2 to under 0.8 for partial hard-check credit, '
        'and 1.0 when every hard check passes.</p></article>'
    )
