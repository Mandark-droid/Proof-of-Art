from __future__ import annotations

import hashlib
import json
import numpy as np
import PIL
import pydantic

from . import VALIDATOR_VERSION
from .checks.core import all_pass
from .dsl import Pattern
from .render import mask_digest


def canonical_design(pattern: Pattern) -> bytes:
    return json.dumps(pattern.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def make_certificate(pattern: Pattern, checks: list[dict], masks, grid) -> dict:
    design_bytes = canonical_design(pattern)
    return {
        "certificate_version": "0.1",
        "type": "auditable_deterministic_verification",
        "schema_version": pattern.schema_version,
        "dsl_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "mask_sha256": mask_digest(masks, grid),
        "validator_version": VALIDATOR_VERSION,
        "runtime": {"numpy": np.__version__, "pillow": PIL.__version__, "pydantic": pydantic.__version__},
        "grid": [64, 64],
        "checks": checks,
        "all_hard_constraints_pass": all_pass(checks),
        "exclusions": ["Not a proof of cultural authenticity", "Not a guarantee of physical manufacturing quality", "Raster checks do not prove continuous vector seam geometry"],
    }


def verify_certificate(certificate: dict, fresh_certificate: dict) -> tuple[bool, str]:
    if not isinstance(certificate, dict):
        return False, "Certificate must be a JSON object"
    for key, expected in fresh_certificate.items():
        # Runtime versions are provenance metadata. Replay verifies the design,
        # deterministic checks, and raster digest across compatible runtimes.
        if key == "runtime":
            continue
        if certificate.get(key) != expected:
            labels = {
                "dsl_sha256": "DSL hash does not match recomputed design",
                "schema_version": "Pattern schema version differs from replay",
                "mask_sha256": "Mask hash does not match recomputed geometry",
                "checks": "Saved checks do not match recomputed checks",
                "all_hard_constraints_pass": "Saved overall status does not match recomputed status",
                "validator_version": "Validator version differs from replay",
                "grid": "Certificate grid differs from replay",
            }
            return False, labels.get(key, f"Certificate field {key!r} does not match replay")
    if set(certificate) != set(fresh_certificate):
        return False, "Certificate fields do not match the current certificate schema"
    return True, "Certificate matches deterministic replay"
