from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspn.components import spn_visualization

from copy import deepcopy

from model_extraction.distributions import DistributionFit

import json
import re

from copy import deepcopy


def remove_duplicate_prefix(label):
    """
    Remove duplicated resource prefixes anywhere in a name.
    """

    if not isinstance(label, str):
        return label

    parts = label.split("_")

    cleaned = []

    i = 0

    while i < len(parts):

        # Detect pattern:
        # something_something_something_something
        # where first two tokens repeat next two tokens
        if (
            i + 3 < len(parts)
            and parts[i] != "XOR"
            and parts[i : i + 2] == parts[i + 2 : i + 4]
        ):

            cleaned.extend(parts[i : i + 2])
            i += 4

        else:

            cleaned.append(parts[i])
            i += 1

    return "_".join(cleaned)


def normalize_path_name(label):
    """
    Removes duplicated path prefixes.
    """

    if not isinstance(label, str):
        return label

    label = label.replace("/", "_")

    # remove repeated underscores
    label = re.sub(r"_+", "_", label)

    return label.strip("_")


def clean_pyspn_name(label):
    """
    Complete name cleanup pipeline.
    """

    label = label.lower()

    label = remove_duplicate_prefix(label)

    label = normalize_path_name(label)

    return label


def prepare_for_pyspn_visualization(spn):
    """
    Prepare SPN for PySPN visualization.
    """

    spn = deepcopy(spn)

    # -----------------------------------------
    # Clean transitions
    # -----------------------------------------

    for transition in spn.transitions:

        if hasattr(transition, "label"):

            old_label = transition.label

            transition.label = clean_pyspn_name(transition.label)


        dist = getattr(transition, "distribution", None)

        if dist is not None and hasattr(dist, "name") and hasattr(dist, "parameters"):

            transition.distribution = {dist.name: dist.parameters}

    # -----------------------------------------
    # Clean places
    # -----------------------------------------

    for place in spn.places:

        if hasattr(place, "label"):

            old_label = place.label

            place.label = clean_pyspn_name(place.label)


    return spn


# =============================================================================
# Conflict loading and matching
# =============================================================================
# Links conflict_detection.py's output back to SPN places/transitions by
# identifier: production-rule formalizations name the same components and
# events ("VGR_1_Pick_Up_And_Transport_Run3", "HBW_1_Store", ...) that
# become place/transition labels once the SPN is extracted, so matching
# on those identifiers — after running them through the SAME
# clean_pyspn_name() cleanup used for the SPN's own labels — is a direct,
# reliable link requiring no separate ID-mapping step to maintain.
#
# This is a self-contained re-implementation of the identifier-matching
# idea from conflict_detection.py, not an import from it: that script
# lives in a separate, unrelated codebase. If you do have it importable
# here, its `_similar()` (sequence-based, handles minor typos and the
# "_RunN" occurrence-marker distinction) is a strictly better matcher
# than the substring check below and can be swapped in directly.

# An identifier looks like a compound, underscore-joined component/event
# name (VGR_1_Pick_Up_And_Transport, HBW_1_Store, SM_2_Transport, ...).
# Plain words from surrounding NL phrasing ("pressure", "bar") won't
# match this and are correctly ignored.
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")

CONFLICT_NOTE_MAX_ENTRIES = 3  # cap how many conflict entries get listed per node label


def load_conflict_records(conflicts_path):
    """
    Loads conflict_detection.py's output JSONL and returns only the
    records that actually have a conflict — the rest carry nothing worth
    matching.
    """
    conflicting = []
    with open(conflicts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("has_conflict") and record.get("conflicts"):
                conflicting.append(record)
    return conflicting


def _formalization_text(record):
    """formalization is either a dict of slot values or an
    "IF ... THEN ... [IN ...]" string (see conflict_detection.py) —
    either way, flatten it to one string to pull identifiers out of."""
    formalization = record.get("formalization")
    if isinstance(formalization, dict):
        return " ".join(str(v) for v in formalization.values() if v)
    if isinstance(formalization, str):
        return formalization
    return ""


def extract_identifiers(record):
    """All compound identifiers mentioned in a conflict record's
    formalization (and, as a fallback, its original text), cleaned with
    the same pipeline used for SPN labels so matching is apples-to-apples."""
    text = _formalization_text(record) + " " + (record.get("original") or record.get("text") or "")
    return {clean_pyspn_name(match) for match in _IDENTIFIER_RE.findall(text)}


def build_conflict_index(conflict_records):
    """identifier -> list of conflict entries mentioning it, where each
    entry summarizes one conflicting statement and what it conflicts with."""
    index = {}
    for record in conflict_records:
        identifiers = extract_identifiers(record)
        if not identifiers:
            continue
        entry = {
            "statement_id": record.get("statement_id"),
            "conflicts_with": [c["statement_id"] for c in record["conflicts"]],
            "methods": sorted({c.get("method", "?") for c in record["conflicts"]}),
            "max_confidence": max((c.get("confidence", 0.0) for c in record["conflicts"]), default=0.0),
        }
        for identifier in identifiers:
            index.setdefault(identifier, []).append(entry)
    return index


def _matches(label, identifier):
    """Substring match in either direction — handles an SPN label that's
    a fragment of a longer formalization identifier, or vice versa."""
    if not label or not identifier:
        return False
    return label in identifier or identifier in label


def _format_conflict_note(entries):
    """entries: list of conflict-index entries that matched one node.
    Renders a short, readable label suffix, capped so it doesn't blow up
    the diagram for a heavily-conflicted node."""
    # de-duplicate by statement_id, keep the highest-confidence entry per statement
    by_statement = {}
    for entry in entries:
        sid = entry["statement_id"]
        if sid not in by_statement or entry["max_confidence"] > by_statement[sid]["max_confidence"]:
            by_statement[sid] = entry

    # de-duplicate mirrored pairs (A-vs-B and B-vs-A are the same conflict,
    # just reported from each side) so a mutual conflict isn't shown twice
    seen_pair_keys = set()
    deduped = []
    for entry in sorted(by_statement.values(), key=lambda e: -e["max_confidence"]):
        pair_key = frozenset([entry["statement_id"], *entry["conflicts_with"]])
        if pair_key in seen_pair_keys:
            continue
        seen_pair_keys.add(pair_key)
        deduped.append(entry)

    shown = deduped[:CONFLICT_NOTE_MAX_ENTRIES]
    lines = [
        f"{e['statement_id']} vs {', '.join(e['conflicts_with'])} "
        f"({'/'.join(e['methods'])}, {e['max_confidence']:.2f})"
        for e in shown
    ]
    remaining = len(deduped) - len(shown)
    if remaining > 0:
        lines.append(f"+{remaining} more")
    return "; ".join(lines)


def annotate_spn_with_conflicts(spn, conflict_records):
    """
    Mutates `spn` in place: sets `is_conflicted = True` and a human
    -readable `conflict_note` string on every place/transition whose
    (already-cleaned) label matches an identifier mentioned in a
    conflicting statement. Safe to call on an SPN whose nodes have never
    had these attributes before — draw_spn() only reads them via
    getattr() with a default, so unmatched nodes are simply left alone
    and render exactly as before.

    Call this AFTER prepare_for_pyspn_visualization(), so matching happens
    against the exact labels that will actually be rendered.
    """
    conflict_index = build_conflict_index(conflict_records)
    if not conflict_index:
        return spn

    n_flagged = 0
    for node in list(spn.places) + list(spn.transitions):
        label = getattr(node, "label", None)
        if not label:
            continue

        matched_entries = [
            entry
            for identifier, entries in conflict_index.items()
            for entry in entries
            if _matches(label, identifier)
        ]
        if not matched_entries:
            continue

        try:
            node.is_conflicted = True
            node.conflict_note = _format_conflict_note(matched_entries)
        except AttributeError:
            # The underlying Place/Transition class defines __slots__
            # without these names — add "is_conflicted" and
            # "conflict_note" to its __slots__ to enable this feature.
            print(f"Could not annotate '{label}': class does not allow new attributes "
                  f"(add 'is_conflicted'/'conflict_note' to its __slots__)")
            continue

        n_flagged += 1

    print(f"Conflict annotation: {n_flagged} place(s)/transition(s) flagged "
          f"from {len(conflict_records)} conflicting statement(s)")
    return spn


def visualize_pyspn_model_with_conflicts(spn, file="default_model", conflict_detected_knowledge=None, conflict_color_override=None):
    """
    conflicts_path: optional path to conflict_detection.py's output
    JSONL. When given, matching places/transitions are drawn with a
    dashed, thicker border and a conflict-detail label suffix (see
    spn_visualization.draw_spn) instead of rendering as normal.
    """

    visualization_spn = prepare_for_pyspn_visualization(spn)

    conflict_records = conflict_detected_knowledge[conflict_detected_knowledge["has_conflict"]].copy()
    #conflict_records = conflict_detected_knowledge
    conflict_records = conflict_records.to_dict(orient="records")

    #print(conflict_records)
    annotate_spn_with_conflicts(visualization_spn, conflict_records)

    # from model_extraction.distributions import DistributionFit

    for t in visualization_spn.transitions:
        if isinstance(getattr(t, "distribution", None), DistributionFit):
            print("NOT CONVERTED:", t.label)

    for t in visualization_spn.transitions:

        if t.t_type == "T" and t.distribution is not None:
            print(t.label, type(t.distribution))

    spn_visualization.draw_spn(visualization_spn, file=file, show=True, conflict_color_override=conflict_color_override)

    return spn