"""Fixed vocabulary shared by the v2 builder and pages."""
from __future__ import annotations

from pramaan.traditions import ALL_STATES

STATE_ORDER: tuple[str, ...] = tuple(ALL_STATES)

STATE_CODES: dict[str, str] = {
    "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS", "Bihar": "BR",
    "Chhattisgarh": "CG", "Goa": "GA", "Gujarat": "GJ", "Haryana": "HR",
    "Himachal Pradesh": "HP", "Jharkhand": "JH", "Karnataka": "KA", "Kerala": "KL",
    "Madhya Pradesh": "MP", "Maharashtra": "MH", "Manipur": "MN", "Meghalaya": "ML",
    "Mizoram": "MZ", "Nagaland": "NL", "Odisha": "OD", "Punjab": "PB",
    "Rajasthan": "RJ", "Sikkim": "SK", "Tamil Nadu": "TN", "Telangana": "TG",
    "Tripura": "TR", "Uttar Pradesh": "UP", "Uttarakhand": "UK", "West Bengal": "WB",
    "Andaman and Nicobar Islands": "AN", "Chandigarh": "CH",
    "Dadra and Nagar Haveli and Daman and Diu": "DD", "Delhi": "DL",
    "Jammu and Kashmir": "JK", "Ladakh": "LA", "Lakshadweep": "LD", "Puducherry": "PY",
}

GRAMMAR_ORDER: tuple[str, ...] = (
    "counted_grid", "stripe_band", "periodic_raster", "jaali_lattice",
    "dot_field", "rotational_field", "kolam_loop",
)

GRAMMAR_LABELS: dict[str, str] = {
    "counted_grid": "Counted grid", "stripe_band": "Stripe bands", "periodic_raster": "Repeat tiles",
    "jaali_lattice": "Jaali lattice", "dot_field": "Dot field", "rotational_field": "Rotational motifs",
    "kolam_loop": "Kolam loops",
}

_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
          "nineteen", "twenty")


def number_word(n: int) -> str:
    """Spell small counts for prose ("eleven models"); larger counts stay as digits."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)
