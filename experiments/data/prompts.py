"""Prompt builders.

The candidate prompt replicates the `core` branch of
`codes/ANES_PAS/generate_finetune_data.py::_create_prompt_from_row` exactly,
so new splits remain comparable with the existing sampled training file.
"""
from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

PARTY_LABELS = {
    1: "Strong Democrat", 2: "Democrat", 3: "Lean Democrat",
    4: "Independent", 5: "Lean Republican", 6: "Republican",
    7: "Strong Republican",
}
IDEOLOGY_LABELS = {
    1: "Very Liberal", 2: "Liberal", 3: "Moderate",
    4: "Conservative", 5: "Very Conservative",
}
RACE_MAP = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}
EDUC_MAP = {
    1: "Elementary or less", 2: "Some high school", 3: "High school diploma",
    4: "Some college", 5: "Bachelor's degree", 6: "Graduate degree",
}
POSITION_MAP = {
    1: "President", 2: "Vice President", 3: "Candidate", 4: "Former President",
    5: "Speaker", 6: "Senator", 7: "Governor", 8: "Justice", 9: "Doctor",
    10: "Advisor",
}


def _int(val, default: int = 0) -> int:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _float(val, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def build_candidate_prompt(row: Mapping | pd.Series, level: str = "core") -> str:
    """Build the candidate PAS prompt. Mirrors the original `core` template."""
    age_resp = _int(row.get("age_resp"), 45)
    gender_resp = "Male" if _int(row.get("gender_resp"), 1) == 1 else "Female"
    race_resp = RACE_MAP.get(_int(row.get("race_resp"), 1), "Unknown")
    educ = EDUC_MAP.get(_int(row.get("educ"), 3), "High school")
    pid7 = _int(row.get("pid7"), 4)
    party_affiliation = PARTY_LABELS.get(pid7, "Independent")
    ideo5 = _int(row.get("ideo5"), 3)
    ideology = IDEOLOGY_LABELS.get(ideo5, "Moderate")

    candidate = row.get("candidate", "Unknown")
    year = _int(row.get("year"), 2020)
    age_cand = _int(row.get("age_cand"), 60)
    party_cand = "Democrat" if _int(row.get("party"), 1) == 1 else "Republican"
    position = POSITION_MAP.get(_int(row.get("position_encoded"), 3), "Candidate")

    if level == "name_only":
        return f"""Predict the support score (0-100) for this voter-candidate pair.

Voter Profile:
- Demographics: {age_resp}-year-old {gender_resp} {race_resp}
- Education: {educ}
- Political: {party_affiliation}, {ideology}

Candidate: {candidate}
- Year: {year}

Support score (0-100):"""

    if level == "minimal":
        return f"""Predict the support score (0-100) for this voter-candidate pair.

Voter: {age_resp}-year-old {gender_resp} {race_resp}, {party_affiliation}, {ideology}
Candidate: {candidate} ({party_cand} {position}, {age_cand} years old)
Year: {year}

Score (0-100):"""

    if level == "full":
        party_match = _int(row.get("party_match"), 0)
        ideology_dist = _float(row.get("ideology_distance_7"), 0)
        age_diff = _int(row.get("age_difference"), 0)
        race_match = _int(row.get("race_match"), 0)
        gender_match = _int(row.get("gender_match"), 0)
        demographic_sim = _float(row.get("demographic_similarity"), 0.5)
        is_incumbent = _int(row.get("is_incumbent"), 0)
        return f"""Predict the support score (0-100) for this voter-candidate pair in {year}.

Voter Profile:
- Demographics: {age_resp}-year-old {gender_resp} {race_resp}
- Education: {educ}
- Political: {party_affiliation}, {ideology}

Candidate: {candidate}
- Party: {party_cand}
- Position: {position}
- Age: {age_cand}

Key Interactions:
- Party alignment: {'Same party' if party_match == 1 else 'Independent voter' if party_match == -1 else 'Different party'}
- Ideology distance: {ideology_dist:.1f} (on 7-point scale)
- Age difference: {age_diff} years
- Race match: {'Yes' if race_match else 'No'}
- Gender match: {'Yes' if gender_match else 'No'}
- Demographic similarity: {demographic_sim:.2f}
- Incumbent: {'Yes' if is_incumbent else 'No'}
- Year: {year}

Support score (0-100):"""

    # default: core
    party_match = _int(row.get("party_match"), 0)
    party_match_text = "Same party" if party_match == 1 else \
        "Independent voter" if party_match == -1 else "Different party"
    ideology_dist = _float(row.get("ideology_distance_7"), 0)
    age_diff = _int(row.get("age_difference"), 0)
    return f"""Predict the support score (0-100) for this voter-candidate pair.

Voter Profile:
- Demographics: {age_resp}-year-old {gender_resp} {race_resp}
- Education: {educ}
- Political: {party_affiliation}, {ideology}

Candidate: {candidate}
- Party: {party_cand}
- Position: {position}
- Age: {age_cand}

Key Interactions:
- Party alignment: {party_match_text}
- Ideology distance: {ideology_dist:.1f} (on 7-point scale)
- Age difference: {age_diff} years
- Year: {year}

Support score (0-100):"""


def build_group_prompt(row: Mapping | pd.Series, category: str) -> str:
    """Group attitude prompt, matching the existing group_*.jsonl template."""
    pid7 = _int(row.get("pid7"), 4)
    party = PARTY_LABELS.get(pid7, "Independent")
    ideo5 = _int(row.get("ideo5"), 3)
    ideology = IDEOLOGY_LABELS.get(ideo5, "Moderate")
    year = _int(row.get("year"), 2020)
    birthyr = _int(row.get("birthyr"), 0)
    age = year - birthyr if birthyr else 0
    group = row.get("group_target", "Unknown")
    return f"""Predict the feeling thermometer score (0-100) for this voter-group pair.

Voter Profile:
- Political identity: {party}
- Ideology: {ideology}
- Age: {age} years old

Social Group: {group}
- Category: {category}

Key Factors:
- Year: {year}

Feeling thermometer (0-100):"""
