"""
conflict_detection.py

"""
from __future__ import annotations

import copy
import functools
import html
import itertools
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger("conflict_detection")


# --- Taxonomy -----------------------------------------------------------------

DIVERGING_OUTCOME = "Diverging Outcome"
DIVERGING_PRECONDITION = "Diverging Precondition"
INTERNAL_INCONSISTENCY = "Internal Inconsistency"
# only reported when report_knowledge_quality=True
REDUNDANCY = "Redundancy (knowledge quality)"

SEVERITY_RANK = {"observation": 0, "potential conflict": 1, "conflict": 2}

# knowledge type -> (slot names of the condition, slot names of the outcome)
CATEGORIES = {
    "state deviation": (
        ["deviation", "condition", "deviations", "conditions"],
        ["state", "mode", "states", "modes"],
    ),
    "maintenance": (
        ["state", "condition", "states", "conditions"],
        ["decision", "action", "decisions", "actions"],
    ),
    "operational": (
        ["condition", "operational_condition", "conditions"],
        ["state", "effect", "states", "effects"],
    ),
    "process flow": (
        ["predecessor", "predecessor_event", "predecessors"],
        ["successor", "successor_event", "successors"],
    ),
    "structural": (
        ["event", "system", "events", "systems", "group"],
        ["associated_event", "associated_events"],
    ),
    "temporal and probabilistic": (
        ["event", "events"],
        ["distribution", "duration", "probability", "weight",
         "distributions", "durations", "probabilities", "weights"],
    ),
}
SCOPED = {"state deviation", "maintenance", "operational"}
# for these types the conditions are events, not value constraints
EVENT_CONDITIONS = {"process flow", "structural", "temporal and probabilistic"}
SCOPE_KEYS = ["event", "component", "events", "components", "scope"]
# outcomes of these types are mutually exclusive states/modes
STRICT_SEVERITY_TYPES = {"state deviation", "operational"}

# Severity of a diverging precondition per knowledge type (section 3.2). None means valid.
PRECONDITION_SEVERITY = {
    "state deviation": "potential conflict",
    "maintenance": "potential conflict",
    "operational": "observation",
    "temporal and probabilistic": "observation",
    "structural": "observation",
    "process flow": "observation",
}


# --- Regexes for exclusiveness, prohibitions and limits -------------------------

_MARKER_WORDS = (
    r"must\s+not|shall\s+not|only|solely|exclusively|exactly\s+one|sole|unique|not\s+allowed|never|"
    r"forbidden|prohibited|at\s+most\s+one|(?:maximum|minimum|upper|lower|hard)\s+(?:boundary|limit)|"
    r"must\s+(?:be|remain|stay)(?:\s+maintained)?\s+(?:between|within)"
)
EXCLUSIVENESS_RE = re.compile(r"\b(?:" + _MARKER_WORDS + r")\b", re.I)
_EMPHASIS_RE = re.compile(r"\b(?:only|solely|exclusively|sole)\b", re.I)
_MARKER_PAREN_RE = re.compile(r"\(\s*(?:" + _MARKER_WORDS + r")(?:\s+\w+)?\s*\)", re.I)
_TRIGGER_ONLY_RE = re.compile(r"\s+(?:only|solely|exclusively)\s*$", re.I)

# A prohibition inside an outcome ("X MUST NOT be stopped"). Modal limits like
# "must not exceed 200" are bounds, not prohibitions - _LIMIT_AFTER_RE catches those.
_PROHIBITION_RE = re.compile(
    r"\b(?:must\s+not|shall\s+not|should\s+not|may\s+not|never|not\s+allowed|forbidden|prohibited)\b", re.I
)
_LIMIT_AFTER_RE = re.compile(
    r"\b(?:must|shall|should|may)\s+not\s+(?:be\s+)?(?:exceed|above|below|under|over|greater|less|more|fall|drop|go)\w*",
    re.I,
)

# words that carry no action when comparing a prohibition with an instruction
_PROHIB_FILLER = {
    "be", "is", "are", "been", "being", "the", "a", "an", "to", "of", "and", "or", "in", "on", "at",
    "for", "with", "this", "that", "it", "while", "when", "if", "then", "condition", "conditions",
    "persist", "persists", "remain", "remains", "event", "events", "also", "any", "all", "not", "no",
    "must", "shall", "should", "may", "never", "allowed", "forbidden", "prohibited", "by", "as",
    "than", "until", "during", "case", "automatic", "automatically", "immediately",
}

# modal limits become comparators, e.g. "must not exceed 200" -> "<= 200"
_LIMIT_REWRITES = [
    (r"\b(?:must|shall|should|may)\s+(?:not|never)\s+(?:exceed|be\s+(?:above|greater\s+than|more\s+than|over))\b"
     r"|\b(?:not|never)\s+exceed\b", "<="),
    (r"\b(?:must|shall|should|may)\s+(?:not|never)\s+(?:fall|drop|go|be)\s+(?:below|under|less\s+than)\b", ">="),
    (r"\b(?:must|shall|should)\s+(?:be|remain|stay)\s+(?=between\b|within\b)", ""),
]

# (positive words, negative words) - the order matters for the reason text
ANTONYM_FAMILIES = [
    (["open", "opens", "opened", "opening"],
     ["close", "closes", "closed", "closing"]),
    (["start", "starts", "started", "starting", "run", "runs", "running", "resume", "resumes", "resumed",
      "continue", "continues", "continued", "keep running", "turn on", "turned on", "power on", "operational"],
     ["stop", "stops", "stopped", "stopping", "halt", "halts", "halted", "shut down", "shutdown", "shut off",
      "turn off", "turned off", "power off", "faulty", "fault", "abnormal", "fail", "fails", "failed", "failing",
      "failure", "failures", "breakdown"]),
    (["increase", "increases", "increased", "increasing"],
     ["decrease", "decreases", "decreased", "decreasing"]),
    (["enable", "enables", "enabled", "activate", "activates", "activated"],
     ["disable", "disables", "disabled", "deactivate", "deactivates", "deactivated"]),
    (["allow", "allows", "allowed"],
     ["prevent", "prevents", "prevented", "forbid", "forbids"]),
    (["above", "exceed", "exceeds", "exceeding", "over"],
     ["below", "under", "falls below", "falling below"]),
    (["present", "available"],
     ["absent", "unavailable", "missing"]),
]
_NEGATION_RE = re.compile(r"\b(?:not|never|no)\b|n't\b")
_BOOL_RE = re.compile(r"^([a-z][a-z_0-9. ]*?)\s*=\s*(true|false)$")

# everything time-like is converted to seconds
TIME_FACTORS = {"ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
UNIT_ALIASES = {
    "seconds": "s", "second": "s", "sec": "s", "secs": "s",
    "minutes": "min", "minute": "min", "mins": "min",
    "hours": "h", "hour": "h", "hrs": "h", "hr": "h",
    "days": "d", "day": "d",
    "milliseconds": "ms", "millisecond": "ms",
    "celsius": "c", "°c": "c", "°": "c", "degree": "c", "degrees": "c", "deg": "c",
    "percent": "%",
}


# --- Small text helpers -------------------------------------------------------

def text(value: Any) -> str:
    """Turn any cell value (None, NaN, list, dict, html-escaped str) into a clean string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(text(v) for v in value if text(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return html.unescape(str(value)).strip()


def norm(value: Any) -> str:
    s = text(value).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def joined(d: Dict[str, Any], keys: Sequence[str]) -> str:
    return "; ".join(text(d[k]) for k in keys if k in d and text(d[k]))


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_NUM_WORDS = {word: value for value, word in enumerate(_ONES)}
_NUM_WORDS.update({word: 10 * (i + 2) for i, word in enumerate(_TENS)})
_NUM_RUN_RE = re.compile(r"\b(?:(?:%s|hundred|thousand)\b[\s-]*)+" % "|".join(_NUM_WORDS), re.I)


def words_to_numbers(s: str) -> str:
    """Replace number words with digits: 'seventy' -> '70', 'twenty five' -> '25'.

    Only simple runs are handled - something like 'one hundred and five' is not.
    """
    def convert(match):
        run = match.group(0)
        # an all-caps token is a variable name (TEN = tension), not a number
        if run.strip().isupper():
            return run
        trailing = run[len(run.rstrip()):]
        total = 0
        current = 0
        for word in re.split(r"[\s-]+", run.strip().lower()):
            if word in _NUM_WORDS:
                current += _NUM_WORDS[word]
            elif word == "hundred":
                current = (current or 1) * 100
            elif word == "thousand":
                total += (current or 1) * 1000
                current = 0
        return str(total + current) + trailing

    return _NUM_RUN_RE.sub(convert, s)


def unit_of(u: Optional[str]) -> str:
    u = (u or "").strip().lower()
    return UNIT_ALIASES.get(u, u)


def _to_base(value: float, u: str) -> Tuple[float, str]:
    if u in TIME_FACTORS:
        return value * TIME_FACTORS[u], "s"
    return value, u


# --- Numerical constraints ----------------------------------------------------
# SMT (Z3) when available, otherwise exact interval arithmetic.

@dataclass(frozen=True)
class Interval:
    lo: float = -math.inf
    hi: float = math.inf
    lc: bool = False   # lower bound closed?
    hc: bool = False   # upper bound closed?

    def subset(self, o: "Interval") -> bool:
        lower_ok = self.lo > o.lo or (self.lo == o.lo and (not self.lc or o.lc))
        upper_ok = self.hi < o.hi or (self.hi == o.hi and (not self.hc or o.hc))
        return lower_ok and upper_ok

    def contains(self, x: float) -> bool:
        above_lo = x > self.lo or (x == self.lo and self.lc)
        below_hi = x < self.hi or (x == self.hi and self.hc)
        return above_lo and below_hi

    def overlaps(self, o: "Interval") -> bool:
        lo = max(self.lo, o.lo)
        hi = min(self.hi, o.hi)
        if lo < hi:
            return True
        return lo == hi and self.contains(lo) and o.contains(lo)

    def intersect(self, o: "Interval") -> "Interval":
        lo = max(self.lo, o.lo)
        hi = min(self.hi, o.hi)
        return Interval(lo, hi,
                        self.contains(lo) and o.contains(lo),
                        self.contains(hi) and o.contains(hi))

    def empty(self) -> bool:
        if self.lo > self.hi:
            return True
        return self.lo == self.hi and not (self.lc and self.hc)


NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
VAR = r"[A-Za-z_][A-Za-z_0-9.]*"
UNIT = r"[%A-Za-z°/]+"
CMP = re.compile(rf"^(?:({VAR})\s*)?(>=|<=|>|<|=)\s*({NUM})(?:\s*±\s*({NUM}))?\s*({UNIT})?$", re.I)
RANGE = re.compile(rf"^({VAR})\s+between\s+({NUM})\s*({UNIT})?\s*(?:-|to)\s*({NUM})\s*({UNIT})?$", re.I)
UNKNOWN_VAR = "_"   # placeholder for a bare comparator like "< 80 C"

Box = Dict[Tuple[str, str], Interval]   # (variable, unit) -> allowed values


@dataclass
class Condition:
    """A parsed condition in DNF.

    disjuncts is a list of OR-branches; each branch is a list of endpoint variants (one per
    combination of tolerance endpoints) and each variant is a Box (the AND of all clauses).
    nominal holds one Box per branch, using the nominal values without tolerance.
    """
    disjuncts: List[List[Box]] = field(default_factory=list)
    nominal: List[Box] = field(default_factory=list)
    has_range: bool = False
    has_comparison: bool = False
    error: str = ""
    unsatisfiable: bool = False

    @property
    def parsed(self) -> bool:
        return bool(self.disjuncts) and not self.error

    def variables(self) -> set:
        return {key[0] for branch in self.disjuncts for box in branch for key in box}

    def holds(self, row: Dict[str, float]) -> Optional[bool]:
        """Evaluate the nominal condition on one log record. Returns None if a variable is missing."""
        if not self.nominal:
            return None
        missing = False
        for box in self.nominal:
            satisfied = True
            for (var, _unit), interval in box.items():
                value = row.get(var)
                if value is None:
                    missing = True
                    satisfied = False
                    break
                if not interval.contains(float(value)):
                    satisfied = False
                    break
            if satisfied:
                return True
        return None if missing else False


def _clause_boxes(clause: str):
    """Parse one clause ("T > 80 ± 2 C", "p between 2 and 4 bar").

    Returns (endpoint alternatives, nominal item, is_range, is_comparison), where every item is
    a ((variable, unit), Interval) tuple.
    """
    m = RANGE.fullmatch(clause)
    if m:
        var, lo, unit_lo, hi, unit_hi = m.groups()
        if unit_lo and unit_hi and unit_of(unit_lo) != unit_of(unit_hi):
            raise ValueError("Range endpoints use different units")
        u = unit_of(unit_hi or unit_lo)
        lo_base, base_unit = _to_base(float(lo), u)
        hi_base, _ = _to_base(float(hi), u)
        if lo_base > hi_base:
            raise ValueError("Reversed numeric range")
        item = ((var.lower(), base_unit), Interval(lo_base, hi_base, True, True))
        return [item], item, True, False

    m = CMP.fullmatch(clause)
    if not m:
        raise ValueError("Unsupported or nonnumeric condition clause: " + clause)
    var, op, value, tolerance, u = m.groups()
    u = unit_of(u)
    value, base_unit = _to_base(float(value), u)
    tolerance = _to_base(abs(float(tolerance or 0)), u)[0]

    def bound(threshold):
        if op == ">":
            return Interval(threshold, math.inf)
        if op == ">=":
            return Interval(threshold, math.inf, True)
        if op == "<":
            return Interval(-math.inf, threshold)
        if op == "<=":
            return Interval(-math.inf, threshold, False, True)
        return Interval(threshold, threshold, True, True)   # "="

    key = ((var or UNKNOWN_VAR).lower(), base_unit)
    # with a tolerance both endpoints are tried; without one this is a single alternative
    endpoints = sorted({value - tolerance, value + tolerance})
    alternatives = [(key, bound(t)) for t in endpoints]
    return alternatives, (key, bound(value)), False, True


_REWRITES = [
    (r"\bgreater than or equal to\b|\bat or above\b|\bat least\b|\bno less than\b", ">="),
    (r"\bless than or equal to\b|\bat or below\b|\bat most\b|\bno more than\b", "<="),
    (r"\blarger than\b|\bgreater than\b|\bmore than\b|\babove\b|\bexceeds?\b|\bover\b", ">"),
    (r"\bsmaller than\b|\bless than\b|\bbelow\b|\bunder\b|\bfewer than\b", "<"),
    (r"\bis equal to\b|\bequal to\b|\bequals?\b", "="),
]


def prepare_condition_text(s: str) -> str:
    """Normalize a condition so the clause regexes can read it."""
    s = html.unescape(s).replace("≥", ">=").replace("≤", "<=").replace("−", "-")
    s = _MARKER_PAREN_RE.sub("", s)
    s = words_to_numbers(s)
    for pattern, repl in _LIMIT_REWRITES:
        s = re.sub(pattern, repl, s, flags=re.I)
    s = re.sub(rf"\bwithin\s+(?={NUM})", "between ", s, flags=re.I)
    for pattern, repl in _REWRITES:
        s = re.sub(pattern, repl, s, flags=re.I)
    # "temperature drifts > 5" -> "temperature > 5"
    s = re.sub(r"\b(?:drifts?|deviat(?:es?|ion)|readings?|values?|drops?|falls?|dips?|rises?|goes)\s+(?=[<>=])",
               "", s, flags=re.I)
    s = re.sub(rf"(between\s+{NUM}\s*(?:{UNIT})?)\s+and\s+(?={NUM})", r"\1 to ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _merge_into(box: Box, key, interval: Interval) -> None:
    box[key] = box[key].intersect(interval) if key in box else interval


def _has_empty(box: Box) -> bool:
    return any(interval.empty() for interval in box.values())


@functools.lru_cache(maxsize=4096)
def parse_condition(raw: str) -> Condition:
    """Parse AND/OR, ranges, open/closed bounds and tolerances, with unit normalization."""
    s = prepare_condition_text(raw)
    result = Condition()
    if not s:
        result.error = "empty"
        return result

    try:
        for branch in re.split(r"\s+OR\s+", s, flags=re.I):
            alternatives = []
            nominal_items = []
            for clause in re.split(r"\s+AND\s+", branch, flags=re.I):
                alts, nominal, is_range, is_cmp = _clause_boxes(clause.strip())
                alternatives.append(alts)
                nominal_items.append(nominal)
                result.has_range |= is_range
                result.has_comparison |= is_cmp
            if len(alternatives) > 10:
                raise ValueError("Too many numeric clauses (limit 10)")

            variants = []
            for combo in itertools.product(*alternatives):
                box: Box = {}
                for key, interval in combo:
                    _merge_into(box, key, interval)
                variants.append(box)

            nominal_box: Box = {}
            for key, interval in nominal_items:
                _merge_into(nominal_box, key, interval)

            if all(_has_empty(v) for v in variants) or _has_empty(nominal_box):
                # this branch can never be true; drop it
                result.unsatisfiable = True
                continue
            result.disjuncts.append([v for v in variants if not _has_empty(v)])
            result.nominal.append(nominal_box)
    except ValueError as exc:
        return Condition(error=str(exc))

    if not result.disjuncts and result.unsatisfiable:
        return Condition(error="Unsatisfiable numeric condition", unsatisfiable=True)
    return result


# Relation between two boxes. Z3 is optional.
try:  # pragma: no cover - depends on the environment
    import z3 as _z3
except Exception:  # noqa: BLE001
    _z3 = None


def _z3_box(box: Box, variables: Dict):
    constraints = []
    for key, iv in box.items():
        v = variables.setdefault(key, _z3.Real(f"{key[0]}__{key[1]}"))
        if iv.lo != -math.inf:
            constraints.append(v >= iv.lo if iv.lc else v > iv.lo)
        if iv.hi != math.inf:
            constraints.append(v <= iv.hi if iv.hc else v < iv.hi)
    return _z3.And(*constraints) if constraints else _z3.BoolVal(True)


def _relation_name(x_in_y: bool, y_in_x: bool) -> str:
    if x_in_y and y_in_x:
        return "equal"
    if x_in_y:
        return "a_subset"
    if y_in_x:
        return "b_subset"
    return "overlap"


def _smt_relation(x: Box, y: Box) -> str:  # pragma: no cover - needs z3
    variables: Dict = {}
    fx = _z3_box(x, variables)
    fy = _z3_box(y, variables)

    def sat(formula):
        solver = _z3.Solver()
        solver.add(formula)
        return solver.check() == _z3.sat

    if not sat(_z3.And(fx, fy)):
        return "disjoint"
    x_in_y = not sat(_z3.And(fx, _z3.Not(fy)))
    y_in_x = not sat(_z3.And(fy, _z3.Not(fx)))
    return _relation_name(x_in_y, y_in_x)


def _interval_relation(x: Box, y: Box) -> str:
    shared = set(x) & set(y)
    if any(not x[k].overlaps(y[k]) for k in shared):
        return "disjoint"
    x_in_y = all(k in x and x[k].subset(bound) for k, bound in y.items())
    y_in_x = all(k in y and y[k].subset(bound) for k, bound in x.items())
    return _relation_name(x_in_y, y_in_x)


def box_relation(x: Box, y: Box, smt: bool = True) -> str:
    if not (set(x) & set(y)):
        return "unrelated"   # no shared variable, so both can hold at the same time
    if smt and _z3 is not None:
        try:
            return _smt_relation(x, y)
        except Exception as exc:  # noqa: BLE001 pragma: no cover
            logger.debug("Z3 failed (%s); using interval arithmetic", exc)
    return _interval_relation(x, y)


def _unify_unknown(a: Box, b: Box) -> Tuple[Box, Box]:
    """A bare comparator ('<80 C') has no variable, so bind it to the single variable on the other side."""
    def rename(box, other):
        unknown = [k for k in box if k[0] == UNKNOWN_VAR]
        named = [k for k in other if k[0] != UNKNOWN_VAR]
        if len(unknown) == 1 and len(box) == 1 and len(named) == 1 and named[0][1] == unknown[0][1]:
            return {named[0]: box[unknown[0]]}
        return box

    return rename(a, b), rename(b, a)


def numeric_condition_relation(a: Condition, b: Condition, smt: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Relation of two parsed conditions: equivalent, overlapping, incompatible or undefined (+ evidence)."""
    if not a.parsed or not b.parsed:
        return "undefined", {}

    # one entry per (branch of a, branch of b), a-major
    pair_relations = []
    for branch_a, branch_b in itertools.product(a.disjuncts, b.disjuncts):
        relations = set()
        for x, y in itertools.product(branch_a, branch_b):
            x, y = _unify_unknown(x, y)
            relations.add(box_relation(x, y, smt))
        # if the tolerance endpoints disagree, the relation isn't well defined
        pair_relations.append(relations.pop() if len(relations) == 1 else None)

    if None in pair_relations:
        return "undefined", {"reason": "relation depends on the tolerance endpoint"}

    evidence = {
        "smt": "z3" if (smt and _z3 is not None) else "interval arithmetic",
        "shared_variables": bool(a.variables() & b.variables()),
        "box_relations": sorted(set(pair_relations)),
    }
    if all(r == "disjoint" for r in pair_relations):
        return "incompatible", evidence
    if len(pair_relations) == 1 and pair_relations[0] == "equal":
        return "equivalent", evidence

    # several OR-branches: equivalent if every branch of a has an equal branch in b
    n_b = len(b.disjuncts)
    if len(a.disjuncts) == n_b:
        rows = [pair_relations[i * n_b:(i + 1) * n_b] for i in range(len(a.disjuncts))]
        if all("equal" in row for row in rows):
            return "equivalent", evidence

    evidence["nesting"] = next((r for r in pair_relations if r in ("a_subset", "b_subset")), None)
    return "overlapping", evidence


# --- Rules --------------------------------------------------------------------

@dataclass
class Rule:
    index: int
    sid: str
    category: str
    condition: str
    outcome: str
    scope: str
    original: str
    source: str
    numeric: Condition
    markers: List[str]
    parse_issues: List[str]
    is_decision: bool = False
    decision: str = ""
    origin_id: str = ""
    # a limit marker inside the condition, e.g. "(hard limit)"
    cond_marked: bool = False
    # "IF <condition> ONLY THEN ...": the outcome is stated for this condition alone
    exclusive_trigger: bool = False

    @property
    def scoped(self) -> bool:
        return bool(self.scope.strip())


_SUBJECT_ROLES = ("operator", "technician", "supervisor", "manager", "visitor", "security", "engineer", "contractor")
_SUBJECT_LEVELS = ("warning", "critical")


def condition_subjects(condition: str) -> Tuple[frozenset, frozenset]:
    """Roles (operator, visitor, ...) and alarm levels (WARNING, CRITICAL) mentioned in a condition."""
    t = condition.lower()
    roles = frozenset(r for r in _SUBJECT_ROLES if re.search(rf"\b{r}s?\b", t))
    levels = frozenset(lvl for lvl in _SUBJECT_LEVELS if re.search(rf"\b{lvl}\b", t))
    return roles, levels


FORMALIZATION_RE = re.compile(r"^\s*IF\s+(?P<c>.+?)\s+THEN\s+(?P<o>.+?)\s*$", re.I | re.S)


def extract_rule(record: Dict[str, Any], index: int) -> Rule:
    category = norm(record.get("category") or record.get("knowledge_type"))
    formalization = record.get("formalization")
    if formalization is None or text(formalization) == "":
        formalization = record.get("formalized_knowledge")

    condition = outcome = scope = ""
    issues: List[str] = []
    slots = CATEGORIES.get(category)

    if isinstance(formalization, dict) and slots:
        condition_keys, outcome_keys = slots
        condition = joined(formalization, condition_keys + ["condition"])
        outcome = joined(formalization, outcome_keys + ["outcome"])
        if category in SCOPED:
            scope = joined(formalization, SCOPE_KEYS)
    else:
        raw = text(formalization)
        if not raw:
            # no formalization: fall back to the first non-empty text field
            for key in ("text", "original", "statement", "sentence"):
                if text(record.get(key)):
                    raw = text(record.get(key))
                    break
        m = FORMALIZATION_RE.fullmatch(html.unescape(raw))
        if m:
            condition, tail = m.group("c"), m.group("o")
            # the scope is whatever follows the last " IN "
            pieces = re.split(r"\s+IN\s+", tail, flags=re.I)
            if len(pieces) > 1:
                outcome = " IN ".join(pieces[:-1])
                scope = pieces[-1]
            else:
                outcome = tail
        else:
            issues.append("Missing or unparseable IF/THEN formalization")

    if category not in CATEGORIES:
        issues.append("Missing or unsupported knowledge category")
    if category in SCOPED and not scope.strip():
        issues.append("Missing component/event scope")

    original = text(record.get("original"))

    # "... 30 C ONLY THEN": remember the marker, but parse the condition without it
    exclusive_trigger = False
    only_match = _TRIGGER_ONLY_RE.search(condition)
    if only_match:
        condition = condition[:only_match.start()].rstrip()
        exclusive_trigger = True

    # markers come from the formalization only - `original` may contain neighbouring sentences
    markers = {m.group(0).lower() for m in EXCLUSIVENESS_RE.finditer(" ".join([condition, outcome]))}
    if exclusive_trigger:
        markers.add("only")
    markers = sorted(markers)

    numeric = parse_condition(condition) if condition else Condition(error="empty")
    sid = text(record.get("statement_id")) or f"EKS_{index + 1}"
    decision = norm(record.get("decision"))

    return Rule(
        index, sid, category, condition.strip(), outcome.strip(), scope.strip(), original,
        text(record.get("source")), numeric, markers, issues,
        bool(decision), decision, text(record.get("origin_statement_id")),
        bool(EXCLUSIVENESS_RE.search(condition)), exclusive_trigger,
    )


# --- Outcome atoms and lexical comparison --------------------------------------

def _atoms(outcome: str, stop_equivalence: bool = True) -> List[str]:
    """Split an outcome into normalized parts (at AND / ';'). Emphasis words like 'only' are dropped."""
    atoms = []
    for part in re.split(r"\s+AND\s+|;|\s+and\s+", html.unescape(outcome), flags=re.I):
        part = _EMPHASIS_RE.sub(" ", _MARKER_PAREN_RE.sub("", part))
        atom = norm(words_to_numbers(part))
        atom = re.sub(r"\b(?:(?:keeps?|continues?|remains?|stays?)\s+)?operat(?:ion|ional|ing|es?)\b",
                      "operational", atom)
        if stop_equivalence:
            atom = re.sub(r"\b(?:e stop|emergency stop|estop)\b", "stop", atom)
        atom = re.sub(r"\s+", " ", atom).strip()
        if atom:
            atoms.append(atom)
    return atoms


def _opposes(a: str, b: str) -> Optional[str]:
    """Check whether two normalized atoms contradict each other. Returns the reason, or None."""
    for positives, negatives in ANTONYM_FAMILIES:
        for pos in positives:
            for neg in negatives:
                pos_re = rf"\b{re.escape(pos)}\b"
                neg_re = rf"\b{re.escape(neg)}\b"
                if ((re.search(pos_re, a) and re.search(neg_re, b))
                        or (re.search(neg_re, a) and re.search(pos_re, b))):
                    return f"antonyms '{pos}' / '{neg}'"

    bool_a, bool_b = _BOOL_RE.match(a), _BOOL_RE.match(b)
    if bool_a and bool_b and bool_a.group(1) == bool_b.group(1) and bool_a.group(2) != bool_b.group(2):
        return f"{bool_a.group(1)} = true vs false"

    # same words, but only one side is negated
    if bool(_NEGATION_RE.search(a)) != bool(_NEGATION_RE.search(b)):
        if _NEGATION_RE.sub("", a).split() == _NEGATION_RE.sub("", b).split():
            return "explicit negation"
    return None


def _stem(w: str) -> str:
    """Very rough stemmer, just good enough to match action words."""
    if w.startswith("failur"):
        return "fail"
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            w = w[:-len(suffix)]
            break
    if w.endswith("y") and len(w) > 3:
        w = w[:-1] + "i"
    elif w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def _action_stems(s: str) -> set:
    t = _PROHIBITION_RE.sub(" ", html.unescape(s).lower())
    t = re.sub(r"shut[\s-]*(?:down|off)", "shutdown", t)
    words = re.findall(r"[a-z0-9]+", t)
    return {_stem(w) for w in words if w not in _PROHIB_FILLER}


def _prohibition_opposes(a: str, b: str, threshold: float = 0.75) -> Optional[str]:
    """One outcome forbids an action ("... MUST NOT be emergency-stopped"), the other one asks for it
    ("emergency stop ...").

    We compare the action words (stems). If the instructed action is contained in the prohibited
    one, they oppose each other. Modal limits ("must not exceed ...") don't count as prohibitions.
    """
    a_prohibits = bool(_PROHIBITION_RE.search(a))
    b_prohibits = bool(_PROHIBITION_RE.search(b))
    if a_prohibits == b_prohibits:
        return None
    negative, positive = (a, b) if a_prohibits else (b, a)
    if _LIMIT_AFTER_RE.search(negative):
        return None
    if len(_action_stems(positive)) < 2 or not _action_stems(negative):
        return None

    def steps(t):
        # the whole outcome plus each of its individual steps
        return [t] + re.split(r"\s+and\s+|;|,\s+(?=[a-z]+\s)", t, flags=re.I)

    best = 0.0
    for x in steps(negative):
        for y in steps(positive):
            sx, sy = _action_stems(x), _action_stems(y)
            shared = sx & sy
            if len(shared) >= 2:
                best = max(best, len(shared) / min(len(sx), len(sy)))
    if best >= threshold:
        return f"prohibition vs instruction ({best:.0%} of the smaller action is shared)"
    return None


def _keywords(s: str) -> set:
    stopwords = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "at", "for", "with",
                 "this", "that", "it"}
    return {w for w in re.findall(r"[a-z0-9%°]+", s.lower()) if w not in stopwords and len(w) > 1}


def text_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the keywords."""
    x, y = _keywords(a), _keywords(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _setpoint_numbers(t: str) -> List[str]:
    """Numbers in a text, ignoring identifiers like 'M12' or 'P3-01'."""
    without_ids = re.sub(r"[A-Za-z]+\d+(?:[-_]\w+)*", " ", t)
    return sorted(re.findall(r"\d+(?:\.\d+)?", without_ids))


# --- Numeric outcomes (durations, probabilities, setpoints) -----------------------

def _dimension(base_unit: str) -> str:
    if base_unit == "s":
        return "time"
    if base_unit == "%":
        return "percent"
    return base_unit


def parse_outcome_interval(outcome: str, allow_bare: bool = False) -> Optional[Tuple[Interval, str]]:
    """Interval and dimension of a numeric outcome ('12-16 minutes', 'min 12 max 16 min', 'p = 0.3')."""
    s = words_to_numbers(html.unescape(outcome)).lower()
    num = r"(\d+(?:\.\d+)?)"
    unit_re = r"\s*(ms|milliseconds?|s|sec|secs|seconds?|min|mins|minutes?|h|hrs?|hours?|d|days?|%|percent)?"
    range_patterns = [
        rf"(?:triangular|uniform)\w*\s*\(\s*{num}\s*,[^)]*?{num}\s*\)\s*{unit_re}",
        rf"min(?:imum)?\s*[=:]?\s*{num}\D+?max(?:imum)?\s*[=:]?\s*{num}{unit_re}",
        rf"between\s+{num}\s*(?:to|and|-)\s*{num}{unit_re}",
        rf"{num}\s*(?:-|to)\s*{num}{unit_re}",
    ]
    for pattern in range_patterns:
        m = re.search(pattern, s)
        if m:
            u = unit_of(m.group(3))
            lo, base_unit = _to_base(float(m.group(1)), u)
            hi, _ = _to_base(float(m.group(2)), u)
            return Interval(min(lo, hi), max(lo, hi), True, True), _dimension(base_unit)

    # a single value - needs a unit unless bare numbers are allowed
    m = re.search(rf"(?:^|[=:\s]){num}{unit_re}\b", s)
    if m and (allow_bare or m.group(2)) and re.fullmatch(r"[\sa-z_:=.\-]*\d[\d.\s%a-z]*", s):
        value, base_unit = _to_base(float(m.group(1)), unit_of(m.group(2)))
        return Interval(value, value, True, True), _dimension(base_unit)
    return None


# --- Models (loaded lazily, optional, can be injected for tests) -------------------

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)


class NLIClassifier:
    """probabilities(pairs) -> [{'contradiction': p, 'entailment': p, 'neutral': p}, ...] (softmax)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def probabilities(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        if not pairs:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

        config = getattr(getattr(self._model, "model", None), "config", None)
        id2label = getattr(config, "id2label", None) or {0: "contradiction", 1: "entailment", 2: "neutral"}

        results: List[Dict[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = list(pairs[start:start + self.batch_size])
            scores = self._model.predict(batch, apply_softmax=True, convert_to_numpy=True)
            for row in scores:
                results.append({str(id2label[i]).lower(): float(row[i]) for i in range(len(row))})
        return results


_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _model(kind: str, spec: Any):
    """spec can be a model name, an already built object, or None/False to disable the model."""
    if spec is None or spec is False:
        return None
    if not isinstance(spec, str):
        return spec
    key = (kind, spec)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = Embedder(spec) if kind == "embedder" else NLIClassifier(spec)
    return _MODEL_CACHE[key]


# --- Detector: relations, categories and severity --------------------------------

@dataclass
class Relation:
    relation: str   # equivalent | overlapping | incompatible | undefined
    method: str = ""
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


class Detector:
    """Classifies rule pairs (phase 2) for one rule base. Keeps the options, models and caches."""

    def __init__(self, rules: List[Rule], backend: str = "nli",
                 embedding_model: Any = "sentence-transformers/all-mpnet-base-v2",
                 nli_model: Any = "cross-encoder/nli-deberta-v3-base",
                 contradiction_threshold: float = 0.90, entailment_threshold: float = 0.90,
                 equivalence_threshold: float = 0.90, numeric_diff_threshold: float = 0.0,
                 require_scope_match: bool = True, cross_category: bool = True,
                 stop_equivalence: bool = True, smt: bool = True, unrelated_conditions: str = "undefined",
                 enable_diverging_precondition: bool = True, report_knowledge_quality: bool = False,
                 subject_guard: bool = True, lexical_check: bool = True):
        if backend not in {"heuristic", "nli"}:
            raise ValueError(f"Unknown backend: {backend!r}")
        if unrelated_conditions not in {"undefined", "potential", "overlapping"}:
            raise ValueError("unrelated_conditions must be undefined, potential or overlapping")

        self.rules = rules
        self.backend = backend
        use_models = backend == "nli"
        self.embedder = _model("embedder", embedding_model) if use_models else None
        self.nli = _model("nli", nli_model) if use_models else None

        self.th_contra = contradiction_threshold
        self.th_entail = entailment_threshold
        self.th_equiv = equivalence_threshold
        self.numeric_diff_threshold = numeric_diff_threshold
        self.require_scope_match = require_scope_match
        self.cross_category = cross_category
        self.subject_guard = subject_guard
        self.lexical_check = lexical_check
        self.stop_equivalence = stop_equivalence
        self.smt = smt
        self.unrelated = unrelated_conditions
        self.enable_dp = enable_diverging_precondition
        self.report_quality = report_knowledge_quality

        self._nli_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._vec_cache: Dict[str, Any] = {}
        self._or_cond, self._or_out = self._or_groups()

    # model helpers

    def _nli_pair(self, premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
        if self.nli is None:
            return None
        key = (premise, hypothesis)
        if key not in self._nli_cache:
            self._nli_cache[key] = self.nli.probabilities([key])[0]
        return self._nli_cache[key]

    def _nli_both(self, a: str, b: str) -> Optional[Dict[str, Any]]:
        """Run NLI in both directions and keep the stronger score."""
        forward = self._nli_pair(a, b)
        backward = self._nli_pair(b, a)
        if forward is None or backward is None:
            return None
        return {
            "contradiction": max(forward.get("contradiction", 0.0), backward.get("contradiction", 0.0)),
            "entailment": max(forward.get("entailment", 0.0), backward.get("entailment", 0.0)),
            "forward": forward,
            "backward": backward,
        }

    def _cosine(self, a: str, b: str) -> Optional[float]:
        if self.embedder is None:
            return None
        for s in (a, b):
            if s not in self._vec_cache:
                self._vec_cache[s] = [float(x) for x in self.embedder.encode([s])[0]]
        # vectors are normalized, so the dot product is the cosine
        return sum(x * y for x, y in zip(self._vec_cache[a], self._vec_cache[b]))

    # textual relation (section 1.2)

    def _lexical_relation(self, a: str, b: str, kind: str) -> Optional[Relation]:
        """Deterministic checks: prohibitions, antonyms/negation, shared parts, same action."""
        if kind == "outcome":
            why = _prohibition_opposes(a, b)
            if why:
                return Relation("incompatible", "lexical", why, {"prohibition": True})

        atoms_a = _atoms(a, self.stop_equivalence)
        atoms_b = _atoms(b, self.stop_equivalence)
        set_a, set_b = set(atoms_a), set(atoms_b)
        if set_a == set_b:
            return Relation("equivalent", "lexical", "identical after normalization")

        for x in atoms_a:
            for y in atoms_b:
                why = _opposes(x, y)
                if why:
                    return Relation("incompatible", "lexical", why, {"atoms": [x, y]})

        if set_a and set_b and (set_a <= set_b or set_b <= set_a):
            return Relation("overlapping", "lexical", "one expression contains the other")
        shared = set_a & set_b
        if shared:
            return Relation("overlapping", "lexical", "shared elements: " + ", ".join(sorted(shared)))

        if kind == "outcome":
            # same action in different words, with the same setpoints and no prohibition
            sx, sy = _action_stems(a), _action_stems(b)
            common = sx & sy
            if (len(common) >= 2 and len(common) / min(len(sx), len(sy)) >= 0.75
                    and _setpoint_numbers(a) == _setpoint_numbers(b)
                    and not _PROHIBITION_RE.search(a) and not _PROHIBITION_RE.search(b)):
                return Relation("equivalent", "lexical",
                                "same action, different wording (shared words: " + ", ".join(sorted(common)) + ")")
        return None

    def _textual(self, a: str, b: str, kind: str) -> Relation:
        """Lexical comparison first; if that's inconclusive, embeddings (equivalence) and NLI."""
        if norm(a) == norm(b):
            return Relation("equivalent", "lexical", "identical expressions")

        if self.lexical_check:   # can be switched off for the ablation study
            found = self._lexical_relation(a, b, kind)
            if found is not None:
                return found

        cos = self._cosine(a, b)
        if cos is None and text_overlap(a, b) >= self.th_equiv:
            return Relation("equivalent", "keyword_overlap", "near-identical wording")
        if cos is not None and cos >= self.th_equiv:
            return Relation("equivalent", "embedding", f"cosine similarity {cos:.2f}", {"cosine": round(cos, 4)})

        probs = self._nli_both(a, b)
        if probs is None:
            return Relation("undefined", "lexical", "no deterministic relation (NLI not used)")

        evidence = {
            "nli_probabilities": {"contradiction": round(probs["contradiction"], 4),
                                  "entailment": round(probs["entailment"], 4)},
            "nli_threshold": self.th_contra,
        }
        if probs["contradiction"] >= self.th_contra:
            return Relation("incompatible", "nli", f"NLI contradiction {probs['contradiction']:.2f}", evidence)
        if probs["entailment"] >= self.th_entail:
            return Relation("overlapping", "nli", f"NLI entailment {probs['entailment']:.2f}", evidence)
        evidence["reason"] = "scores below threshold"
        return Relation("undefined", "nli", "NLI scores below threshold", evidence)

    # condition relation

    def condition_relation(self, a: Rule, b: Rule) -> Relation:
        if a.numeric.parsed and b.numeric.parsed:
            rel, ev = numeric_condition_relation(a.numeric, b.numeric, self.smt)
            no_shared_vars = ev.get("shared_variables") is False
            if rel == "overlapping" and no_shared_vars and self.unrelated != "overlapping":
                if self.unrelated == "undefined":
                    return Relation("undefined", "smt",
                                    "conditions share no variable; co-occurrence is not established", ev)
                ev["unrelated_variables"] = True   # "potential": keep it, but flag it
            return Relation(rel, "smt", "numerical constraints", ev)

        if a.category in EVENT_CONDITIONS and b.category in EVENT_CONDITIONS:
            return self._event_relation(a.condition, b.condition)
        if a.numeric.parsed != b.numeric.parsed:
            return Relation("undefined", "mixed", "numerical vs textual condition")
        return self._textual(a.condition, b.condition, "condition")

    def _event_relation(self, a: str, b: str) -> Relation:
        """Predecessor/group/trigger events: shared events overlap, different events are incompatible."""
        rel = self._textual(a, b, "condition")
        if rel.relation != "undefined":
            return rel
        return Relation("incompatible", "event", "different events", {"events": [a, b]})

    # outcome relation

    def outcome_relation(self, a: Rule, b: Rule) -> Relation:
        # "X MUST NOT ..." vs "X ..."
        why = _prohibition_opposes(a.outcome, b.outcome) if self.lexical_check else None
        if why:
            return Relation("incompatible", "lexical", why, {"prohibition": True})

        both_temporal = a.category == "temporal and probabilistic" and b.category == "temporal and probabilistic"
        interval_a = parse_outcome_interval(a.outcome, both_temporal)
        interval_b = parse_outcome_interval(b.outcome, both_temporal)
        if interval_a and interval_b:
            (x, dim_x), (y, dim_y) = interval_a, interval_b
            if dim_x != dim_y:
                return Relation("undefined", "smt", f"dimensions differ ({dim_x} vs {dim_y})")
            intervals = [(x.lo, x.hi), (y.lo, y.hi)]
            if x == y:
                return Relation("equivalent", "smt", "identical values", {"intervals": intervals})
            if x.overlaps(y):
                return Relation("overlapping", "smt", "intervals overlap", {"intervals": intervals})
            scale = max(abs(x.lo), abs(x.hi), abs(y.lo), abs(y.hi), 1e-12)
            gap = (max(x.lo, y.lo) - min(x.hi, y.hi)) / scale
            ev = {"intervals": intervals, "relative_gap": round(gap, 4)}
            if gap >= self.numeric_diff_threshold:
                return Relation("incompatible", "smt", f"disjoint intervals, relative gap {gap:.2f}", ev)
            return Relation("overlapping", "smt", "disjoint but within tolerance", ev)

        # outcomes that are constraints themselves ("pressure <= 4 bar")
        cond_a, cond_b = parse_condition(a.outcome), parse_condition(b.outcome)
        if cond_a.parsed and cond_b.parsed and cond_a.has_comparison and cond_b.has_comparison:
            rel, ev = numeric_condition_relation(cond_a, cond_b, self.smt)
            if rel != "undefined":
                return Relation(rel, "smt", "numerical outcome constraints", ev)

        return self._textual(a.outcome, b.outcome, "outcome")

    # OR-branches

    def _or_groups(self):
        """Collect the OR-alternatives used anywhere in the rule base (for conditions and outcomes)."""
        cond_groups, out_groups = [], []
        for rule in self.rules:
            parts = [norm(p) for p in re.split(r"\s+OR\s+", rule.condition, flags=re.I) if p.strip()]
            if len(parts) > 1:
                cond_groups.append(set(parts))
            parts = [norm(p) for p in re.split(r"\s+OR\s+", rule.outcome, flags=re.I) if p.strip()]
            if len(parts) > 1:
                out_groups.append(set(parts))
        return cond_groups, out_groups

    @staticmethod
    def _members(s: str) -> set:
        return {norm(p) for p in re.split(r"\s+OR\s+", s, flags=re.I) if p.strip()}

    def shares_or_branch(self, a: Rule, b: Rule) -> bool:
        """True if another rule lists parts of a and b as alternatives of one OR."""
        checks = ((self._or_cond, a.condition, b.condition),
                  (self._or_out, a.outcome, b.outcome))
        for groups, text_a, text_b in checks:
            members_a, members_b = self._members(text_a), self._members(text_b)
            for group in groups:
                if any(x != y for x in members_a & group for y in members_b & group):
                    return True
        return False

    # pair classification

    def comparable(self, a: Rule, b: Rule) -> bool:
        if a.category != b.category and not self.cross_category:
            return False
        if self.require_scope_match and a.scoped and b.scoped and norm(a.scope) != norm(b.scope):
            return False   # different context, valid by default
        if self.subject_guard:
            # a different role or alarm level means a separate scope, also valid by default
            roles_a, levels_a = condition_subjects(a.condition)
            roles_b, levels_b = condition_subjects(b.condition)
            if roles_a and roles_b and not roles_a & roles_b:
                return False
            if levels_a and levels_b and not levels_a & levels_b:
                return False
        return True

    @staticmethod
    def _unrelated(cond: "Relation") -> bool:
        return bool(cond.evidence.get("unrelated_variables")) or cond.evidence.get("shared_variables") is False

    def _category_for(self, a: Rule, b: Rule, cond: Relation, out: Relation, markers: List[str]) -> Optional[str]:
        """Step 2: anomaly category of a pair, or None if there is none."""
        if cond.relation in ("equivalent", "overlapping"):
            unrelated = self._unrelated(cond)
            nesting = cond.evidence.get("nesting")
            if out.relation == "incompatible":
                return DIVERGING_OUTCOME
            if out.relation == "overlapping" and markers and not unrelated:
                return DIVERGING_OUTCOME   # exclusiveness stated -> conflict (section 3.1)
            if (markers and cond.relation == "overlapping" and out.relation == "equivalent"
                    and not nesting and not unrelated):
                # an exclusive limit is exceeded by a partially overlapping region
                return DIVERGING_PRECONDITION
            if (cond.relation == "overlapping" and out.relation == "equivalent" and nesting
                    and not unrelated and self.lexical_check
                    and self._exclusive_trigger_exceeded(a, b, cond.evidence["nesting"])):
                # "ONLY THEN": the other rule triggers the same outcome outside that region
                return DIVERGING_PRECONDITION
            return None
        if cond.relation == "incompatible" and self.enable_dp and out.relation in ("equivalent", "overlapping"):
            return DIVERGING_PRECONDITION
        return None

    def classify_pair(self, a: Rule, b: Rule) -> Optional[Dict[str, Any]]:
        """Returns a finding or a valid discrepancy, or None if the pair isn't comparable / can't be decided."""
        if a.parse_issues or b.parse_issues or not self.comparable(a, b):
            return None
        if (a.numeric.unsatisfiable and not a.numeric.parsed) or (b.numeric.unsatisfiable and not b.numeric.parsed):
            return None

        cond = self.condition_relation(a, b)
        out = self.outcome_relation(a, b)

        markers = sorted(set(a.markers) | set(b.markers)) if self.lexical_check else []
        if _PROHIBITION_RE.search(a.outcome) and _PROHIBITION_RE.search(b.outcome):
            # two prohibitions don't exclude each other
            markers = [m for m in markers if not _PROHIBITION_RE.fullmatch(m)]

        base = {
            "condition_relation": cond.relation,
            "outcome_relation": out.relation,
            "evidence": {
                "condition": {"method": cond.method, "reason": cond.reason, **cond.evidence},
                "outcome": {"method": out.method, "reason": out.reason, **out.evidence},
            },
        }
        # lift the NLI scores to the top level of the evidence
        for rel in (cond, out):
            for key, value in rel.evidence.items():
                if key in ("nli_probabilities", "nli_threshold"):
                    base["evidence"][key] = value
        methods = sorted({m for m in (cond.method, out.method) if m})

        category = self._category_for(a, b, cond, out, markers)
        if category is None:
            if cond.relation == "undefined" or (out.relation == "undefined" and cond.relation != "incompatible"):
                return None
            if cond.relation == "incompatible":
                return None   # a case split / alternative scenario
            if cond.relation == "equivalent" and out.relation == "equivalent":
                subtype = "duplicate"
            elif out.relation == "equivalent" or cond.relation == "equivalent" or cond.evidence.get("nesting"):
                subtype = "subsumption"
            else:
                subtype = "overlap"
            return {"kind": "valid", "subtype": subtype, "reason": "Valid discrepancy: " + subtype, **base}

        severity, why = self._severity(category, a, b, cond, out, markers)
        if severity is None:
            return {"kind": "valid", "subtype": "alternative", "reason": why, **base}
        return {"kind": "finding", "category": category, "severity": severity, "reason": why,
                "method": ", ".join(methods), "exclusiveness_markers": markers, **base}

    @staticmethod
    def _exclusive_trigger_exceeded(a: Rule, b: Rule, nesting: str) -> bool:
        """Nested conditions with equal outcomes, where one rule marks its condition as exclusive.

        'IF x > 30 ONLY THEN o' (narrow, exclusive) vs 'IF x > 26 THEN o' (wide):
            the wide rule triggers outside the exclusive region.
        'IF x > 28 (hard limit) THEN o' (wide, limit) vs 'IF x > 30 THEN o' (narrow):
            the narrow rule only acts after the limit has been passed.
        """
        if nesting == "a_subset":
            narrow, wide = a, b
        elif nesting == "b_subset":
            narrow, wide = b, a
        else:
            return False
        if narrow.exclusive_trigger and not wide.exclusive_trigger:
            return True
        return (wide.cond_marked and not wide.exclusive_trigger
                and not narrow.cond_marked and not narrow.exclusive_trigger)

    def _severity(self, category, a, b, cond, out, markers) -> Tuple[Optional[str], str]:
        """Step 3: severity class and the reason for it. A None severity means the pair is valid."""
        shared_or = self.shares_or_branch(a, b)
        unrelated = bool(cond.evidence.get("unrelated_variables"))

        if markers:
            return "conflict", "Mutual exclusiveness stated (" + ", ".join(markers) + "): conflict for all knowledge types."

        if category == DIVERGING_OUTCOME:
            if shared_or:
                return "observation", "Rules share an OR-branch through another rule."
            if a.category in STRICT_SEVERITY_TYPES and b.category in STRICT_SEVERITY_TYPES:
                if unrelated:
                    return "potential conflict", "Conditions share no variable; their co-occurrence is not established."
                return "conflict", "Conditions can hold together and the outcomes describe incompatible states/modes."
            return "potential conflict", ("Outcomes of this knowledge-type combination can coexist (branching, "
                                          "parallelism, sequence); the weaker severity is chosen.")

        # Diverging Precondition: take the weaker severity of the two knowledge types
        levels = []
        for rule in (a, b):
            level = PRECONDITION_SEVERITY.get(rule.category, "observation")
            if rule.category == "process flow" and shared_or:
                level = None
            levels.append(level)
        if None in levels and all(rule.category == "process flow" for rule in (a, b)):
            return None, "Alternative predecessors stated as an OR-branch: valid process path."
        levels = [lvl for lvl in levels if lvl is not None] or ["observation"]
        severity = min(levels, key=SEVERITY_RANK.get)

        if a.category in ("state deviation", "maintenance") and b.category in ("state deviation", "maintenance"):
            why = "A different trigger changes when the state, event or action occurs."
        else:
            why = "Different predecessors lead to the same outcome; limits are often stated with margins or as associations."
        return severity, why

    # internal inconsistency (section 2.3)

    def internal_inconsistency(self, r: Rule) -> Optional[Dict[str, Any]]:
        if r.parse_issues:
            return None
        if r.numeric.unsatisfiable and not r.numeric.parsed:
            return self._internal(
                r, "Condition constraints are unsatisfiable (no value can satisfy them): the rule can never be activated.",
                "smt")

        for kind, rule_text in (("condition", r.condition), ("outcome", r.outcome)):
            atoms = _atoms(rule_text, self.stop_equivalence)
            use_nli = self.nli is not None and len(atoms) <= 6 and kind == "outcome"
            for x, y in itertools.combinations(atoms, 2):
                why = _opposes(x, y)
                if why:
                    return self._internal(r, f"The {kind} contains contradictory parts ({why}).", "lexical",
                                          {"atoms": [x, y]})
                if use_nli:
                    probs = self._nli_both(x, y)
                    if probs and probs["contradiction"] >= self.th_contra:
                        evidence = {"atoms": [x, y],
                                    "nli_probabilities": {"contradiction": round(probs["contradiction"], 4)}}
                        return self._internal(r, f"NLI contradiction {probs['contradiction']:.2f} within the {kind}.",
                                              "nli", evidence)
        return None

    @staticmethod
    def _internal(r: Rule, reason: str, method: str, evidence: Optional[Dict] = None) -> Dict[str, Any]:
        return {"kind": "finding", "category": INTERNAL_INCONSISTENCY, "severity": "conflict", "reason": reason,
                "method": method, "condition_relation": "n/a", "outcome_relation": "n/a",
                "evidence": evidence or {}, "exclusiveness_markers": []}


# --- Phase 1: candidate pairs --------------------------------------------------

def select_candidate_pairs(rules: List[Rule], detector: Detector, top_k: int = 8, similarity_threshold: float = 0.45,
                           related_pairs: Optional[Sequence[Tuple[str, str]]] = None) -> List[Tuple[int, int]]:
    """Union of deterministic matching and semantic retrieval (k nearest neighbours above a cosine threshold)."""
    n = len(rules)
    pairs = set()
    related = {frozenset(p) for p in (related_pairs or [])}

    # deterministic matching
    for i, j in itertools.combinations(range(n), 2):
        a, b = rules[i], rules[j]
        if not detector.comparable(a, b):
            continue
        both_scoped = a.scoped and b.scoped
        if frozenset((a.sid, b.sid)) in related:
            pairs.add((i, j))
        elif both_scoped and norm(a.scope) == norm(b.scope):
            pairs.add((i, j))
        elif not both_scoped and (norm(a.condition) == norm(b.condition) or _atoms(a.outcome) == _atoms(b.outcome)):
            pairs.add((i, j))

    # semantic retrieval
    if n > 1:
        texts = [f"{r.condition} -> {r.outcome} {r.scope}".strip() for r in rules]
        if detector.embedder is not None:
            vectors = detector.embedder.encode(texts)

            def similarity(i, j):
                return float(sum(float(x) * float(y) for x, y in zip(vectors[i], vectors[j])))
        else:
            def similarity(i, j):
                return text_overlap(texts[i], texts[j])

        for i in range(n):
            scored = sorted(((similarity(i, j), j) for j in range(n)
                             if j != i and detector.comparable(rules[i], rules[j])), reverse=True)
            for score, j in scored[:top_k]:
                if score >= similarity_threshold:
                    pairs.add((min(i, j), max(i, j)))

    return sorted(pairs)


# --- Public API ---------------------------------------------------------------

RESULT_COLUMNS = [
    "anomalies", "conflicts", "valid_discrepancies", "rejected_equivalents",
    "has_anomaly", "has_conflict", "has_severe_conflict", "max_severity", "resulting_severity",
    "resulting_anomaly_category", "resulting_anomaly_categories", "resulting_severity_classes",
    "resulting_classifications",
]


def _finding_for(result: Dict[str, Any], me: Rule, other: Optional[Rule]) -> Dict[str, Any]:
    """Turn a classification result into the finding stored on `me` (other=None for internal findings)."""
    finding = {
        "statement_id": other.sid if other else None,
        "conflict_type": result["category"],
        "anomaly_category": result["category"],
        "severity": result["severity"],
        "severity_class": result["severity"],
        "condition_relation": result["condition_relation"],
        "outcome_relation": result["outcome_relation"],
        "method": result["method"],
        "reason": result["reason"],
        "evidence": copy.deepcopy(result["evidence"]),
        "category": other.category if other else me.category,
    }
    if result.get("exclusiveness_markers"):
        finding["exclusiveness_markers"] = result["exclusiveness_markers"]
    if other is not None and other.is_decision:
        finding["counterpart_is_expert_decision"] = True
        finding["expert_decision"] = other.decision
        finding["expert_decision_id"] = other.sid
    return finding


def _summarize(row: Dict[str, Any]) -> None:
    found = row["anomalies"]
    found.sort(key=lambda h: (-SEVERITY_RANK[h["severity"]], h["conflict_type"], str(h.get("statement_id") or "")))
    highest = max((h["severity"] for h in found), key=SEVERITY_RANK.get, default=None)
    primary = sorted({h["conflict_type"] for h in found if h["severity"] == highest})

    row["conflicts"] = copy.deepcopy(found)
    row["has_anomaly"] = bool(found)
    row["has_conflict"] = bool(found)
    row["has_severe_conflict"] = any(h["severity"] in ("conflict", "potential conflict") for h in found)
    row["max_severity"] = highest
    row["resulting_severity"] = highest
    row["resulting_anomaly_category"] = primary[0] if primary else None
    row["resulting_anomaly_categories"] = sorted({h["conflict_type"] for h in found})
    row["resulting_severity_classes"] = sorted({h["severity"] for h in found}, key=SEVERITY_RANK.get, reverse=True)
    row["resulting_classifications"] = [
        {"anomaly_category": h["conflict_type"], "severity": h["severity"], "conflict_with": h["statement_id"]}
        for h in found
    ]


def _prepare(records: Any):
    """Validate the input, deep-copy it into a list of dicts and fill in missing statement ids."""
    is_df = isinstance(records, pd.DataFrame)
    if not is_df and not isinstance(records, list):
        raise TypeError("records must be a pandas DataFrame or a list of dictionaries")
    if is_df and not records.columns.is_unique:
        raise ValueError("DataFrame column names must be unique")

    rows = copy.deepcopy(records.to_dict("records") if is_df else records)
    if any(not isinstance(r, dict) for r in rows):
        raise TypeError("Each record must be a dictionary")

    seen = set()
    for i, row in enumerate(rows, 1):
        sid = text(row.get("statement_id")) or f"EKS_{i}"
        if sid in seen:
            logger.warning("Duplicate statement_id %r; partner ids may be ambiguous.", sid)
        seen.add(sid)
        row["statement_id"] = sid
    return is_df, rows


def _as_rows(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return [dict(v) for v in value]


def _print_pair(i, j, a, b, res, pair_no, total, source):
    """Print one candidate pair with relations, category, severity and reason (verbose=True)."""
    line = "-" * 100
    print(f"\n{line}\n[pair {pair_no}/{total}] candidate via {source}: {a.sid}  <->  {b.sid}")
    for r in (a, b):
        print(f"   {r.sid:<10} [{r.category}] source={r.source or '-'} scope={r.scope or '-'}")
        print(f"              IF {r.condition}  THEN {r.outcome}")
    if res is None:
        print("   => not comparable / relation undetermined: no finding")
        return

    ev = res.get("evidence", {})
    cond_ev = ev.get("condition", {})
    out_ev = ev.get("outcome", {})
    print(f"   condition relation : {res['condition_relation']:<11} "
          f"({cond_ev.get('method', '-')}: {cond_ev.get('reason', '')})")
    print(f"   outcome relation   : {res['outcome_relation']:<11} "
          f"({out_ev.get('method', '-')}: {out_ev.get('reason', '')})")

    if res["kind"] != "finding":
        print(f"   => VALID DISCREPANCY ({res['subtype']}): {res['reason']}")
        return
    print(f"   => ANOMALY  category={res['category']} | severity={res['severity'].upper()}")
    print(f"      conflicts with: {b.sid} (for {a.sid}) / {a.sid} (for {b.sid})")
    if res.get("exclusiveness_markers"):
        print(f"      exclusiveness markers: {', '.join(res['exclusiveness_markers'])}")
    print(f"      method: {res.get('method') or '-'}\n      reason: {res['reason']}")


def _print_detection_summary(rows) -> None:
    """Print the findings per severity class and per statement (verbose=True)."""
    line = "=" * 100
    seen = set()
    counts = {}
    print(f"\n{line}\nDETECTION SUMMARY")
    for r in rows:
        for f in r["anomalies"]:
            partner = f.get("statement_id")
            # every pair shows up on both statements, so only count it once
            key = (tuple(sorted([str(r["statement_id"]), str(partner)])), f["conflict_type"])
            if key in seen:
                continue
            seen.add(key)
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
            partner_label = partner if partner is not None else "(internal)"
            print(f"  {f['severity'].upper():<17} {f['conflict_type']:<24} {r['statement_id']} <-> {partner_label}")
    valid = sum(1 for r in rows for v in r["valid_discrepancies"] if str(r["statement_id"]) < str(v["statement_id"]))
    print(f"  totals: {counts or 'no findings'}; valid discrepancies: {valid}\n{line}")


def detect_conflicts(records, backend="nli", embedding_model="sentence-transformers/all-mpnet-base-v2",
                     nli_model="cross-encoder/nli-deberta-v3-base", similarity_threshold=0.45, top_k=8,
                     contradiction_threshold=0.90, numeric_diff_threshold=0.0, require_scope_match=True,
                     cross_category=True, outcome_match_threshold=None, condition_relatedness_threshold=None,
                     original_text_threshold=None, enable_diverging_predecessor=True, enable_formalization_check=None,
                     *, stop_equivalence=True, inplace=None, expert_decisions=None, related_pairs=None,
                     entailment_threshold=0.90, equivalence_threshold=0.90, smt=True, unrelated_conditions="undefined",
                     report_knowledge_quality=False, candidate_selection=True, verbose=False, subject_guard=True,
                     lexical_check=True, **ignored):
    """Detect conflicts, potential conflicts and observations and annotate the records with them.

    For DataFrame input a copy is returned. Lists are updated in place unless inplace=False.

    backend="nli" uses sentence embeddings and an NLI cross-encoder (the models are downloaded on
    first use). backend="heuristic" runs fully offline with the deterministic checks only.
    embedding_model / nli_model can also be objects with .encode(texts) / .probabilities(pairs).

    candidate_selection=False compares every comparable pair (the resolution step uses this).
    verbose=True prints every candidate pair with its relations, category, severity and reason.

    outcome_match_threshold, condition_relatedness_threshold, original_text_threshold and
    enable_formalization_check no longer do anything; they are only accepted so old calls keep working.
    """
    if ignored:
        logger.warning("Ignoring unknown detect_conflicts options: %s", ", ".join(sorted(ignored)))

    is_df, rows = _prepare(records)
    n = len(rows)
    rules = [extract_rule(r, i) for i, r in enumerate(rows)]

    # expert decisions are appended after the statements (indices n, n+1, ...)
    for k, decision_record in enumerate(_as_rows(expert_decisions)):
        decision_record.setdefault("statement_id", f"EXP_{k + 1}")
        rule = extract_rule(decision_record, n + k)
        rule.is_decision = True
        rule.decision = norm(decision_record.get("decision"))
        rules.append(rule)

    detector = Detector(rules, backend, embedding_model, nli_model, contradiction_threshold, entailment_threshold,
                        equivalence_threshold, numeric_diff_threshold, require_scope_match, cross_category,
                        stop_equivalence, smt, unrelated_conditions, enable_diverging_predecessor,
                        report_knowledge_quality, subject_guard, lexical_check)

    for row in rows:
        row["anomalies"] = []
        row["valid_discrepancies"] = []
        row["rejected_equivalents"] = []

    # Phase 1
    statement_rules = rules[:n]
    if candidate_selection:
        pairs = select_candidate_pairs(statement_rules, detector, top_k, similarity_threshold, related_pairs)
        pair_source = "deterministic/embedding candidate selection"
    else:
        pairs = list(itertools.combinations(range(n), 2))
        pair_source = "exhaustive comparison"

    # Phase 2 + 3
    counter = itertools.count(1)
    total = len(pairs) + (len(rules) - n) * n

    def record_pair(i: int, j: int) -> None:
        a, b = rules[i], rules[j]
        res = detector.classify_pair(a, b)
        if verbose:
            source = "expert decision" if max(i, j) >= n else pair_source
            _print_pair(i, j, a, b, res, next(counter), total, source)
        if res is None:
            return

        for me, other, idx in ((a, b, i), (b, a, j)):
            if idx >= n:
                continue   # expert decisions don't get annotated
            row = rows[idx]
            if res["kind"] == "finding":
                row["anomalies"].append(_finding_for(res, me, other))
                continue

            row["valid_discrepancies"].append({
                "statement_id": other.sid,
                "subtype": res["subtype"],
                "reason": res["reason"],
                "condition_relation": res["condition_relation"],
                "outcome_relation": res["outcome_relation"],
                "category": other.category,
                "source": other.source,
            })
            if report_knowledge_quality and res["subtype"] in ("duplicate", "subsumption"):
                row["anomalies"].append({
                    "statement_id": other.sid,
                    "conflict_type": REDUNDANCY,
                    "anomaly_category": REDUNDANCY,
                    "severity": "observation",
                    "severity_class": "observation",
                    "condition_relation": res["condition_relation"],
                    "outcome_relation": res["outcome_relation"],
                    "method": "knowledge_quality",
                    "reason": res["reason"],
                    "evidence": {},
                    "category": other.category,
                })

    for i, j in pairs:
        record_pair(i, j)

    for i, rule in enumerate(statement_rules):
        res = detector.internal_inconsistency(rule)
        if res:
            rows[i]["anomalies"].append(_finding_for(res, rule, None))

    for k, decision in enumerate(rules[n:]):
        for i in range(n):
            if not decision.decision.startswith("reject"):
                record_pair(i, n + k)
                continue
            # a rejected decision: only check whether the statement is the same thing again
            cond = detector.condition_relation(rules[i], decision)
            out = detector.outcome_relation(rules[i], decision)
            if detector.comparable(rules[i], decision) and cond.relation == "equivalent" and out.relation == "equivalent":
                rows[i]["rejected_equivalents"].append({
                    "expert_decision_id": decision.sid,
                    "statement_id": decision.origin_id or decision.sid,
                    "relation": "equivalent",
                })

    for row in rows:
        _summarize(row)
    if verbose:
        _print_detection_summary(rows)

    mutate = (not is_df) if inplace is None else bool(inplace)
    if is_df:
        result = records if mutate else records.copy(deep=True)
        for col in RESULT_COLUMNS:
            result[col] = [r[col] for r in rows]
        return result
    if mutate:
        for target, source in zip(records, rows):
            target.update(source)
        return records
    return rows


def detect_pairs(records: Any, index_pairs: Sequence[Tuple[int, int]], **options) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Run phase 2 on selected record pairs only (used by the camp compatibility check in phase 5).

    Returns {(i, j): result} with result["kind"] being "finding" or "valid". Pairs that
    can't be decided are left out.
    """
    _, rows = _prepare(records)
    rules = [extract_rule(r, i) for i, r in enumerate(rows)]
    detector = Detector(rules, **_detector_options(options))
    results = {}
    for i, j in index_pairs:
        res = detector.classify_pair(rules[i], rules[j])
        if res is not None:
            results[(i, j)] = res
    return results


# options that detect_pairs forwards to the Detector (all keep their name except one)
_DETECTOR_OPTION_NAMES = [
    "backend", "embedding_model", "nli_model", "contradiction_threshold", "entailment_threshold",
    "equivalence_threshold", "numeric_diff_threshold", "require_scope_match", "cross_category",
    "stop_equivalence", "smt", "unrelated_conditions", "enable_diverging_predecessor", "report_knowledge_quality",
]
_DETECTOR_OPTION_RENAMES = {"enable_diverging_predecessor": "enable_diverging_precondition"}


def _detector_options(o: Dict[str, Any]) -> Dict[str, Any]:
    return {_DETECTOR_OPTION_RENAMES.get(name, name): o[name] for name in _DETECTOR_OPTION_NAMES if name in o}


def findings_dataframe(enriched) -> pd.DataFrame:
    """One row per unique finding (per pair and category)."""
    rows = enriched.to_dict("records") if isinstance(enriched, pd.DataFrame) else enriched
    columns = ["statement_id", "conflict_with", "conflict_type", "severity", "condition_relation",
               "outcome_relation", "method", "reason"]
    result = []
    seen = set()
    for row in rows:
        for hit in row.get("anomalies", []):
            partner = hit.get("statement_id")
            if partner is not None:
                ids = tuple(sorted([str(row["statement_id"]), str(partner)]))
            else:
                ids = (str(row["statement_id"]),)
            key = (ids, hit["conflict_type"])
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "statement_id": row["statement_id"],
                "conflict_with": partner,
                "conflict_type": hit["conflict_type"],
                "severity": hit["severity"],
                "condition_relation": hit.get("condition_relation"),
                "outcome_relation": hit.get("outcome_relation"),
                "method": hit.get("method"),
                "reason": hit.get("reason"),
            })
    return pd.DataFrame(result, columns=columns)


def analyze_statements(records, **options):
    enriched = detect_conflicts(records, **options)
    return {"records": enriched, "findings": findings_dataframe(enriched)}


def detect_conflicts_in_statement(records=None, **options):
    """Old entry point: detect_conflicts, but with the offline backend as the default."""
    if records is None:
        raise ValueError("records must be provided (command line execution has been removed)")
    options.setdefault("backend", "heuristic")
    return detect_conflicts(records, **options)


def read_jsonl(path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
