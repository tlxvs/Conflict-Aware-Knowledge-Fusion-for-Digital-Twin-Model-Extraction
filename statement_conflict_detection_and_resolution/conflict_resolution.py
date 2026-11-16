"""
conflict_resolution.py

"""

from __future__ import annotations

import itertools
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:  # installed as a package
    from statement_conflict_detection_and_resolution.conflict_detection import (
        INTERNAL_INCONSISTENCY, SEVERITY_RANK, _atoms, detect_pairs, extract_rule, norm, parse_condition, text)
except ImportError:  # running from the source folder
    from conflict_detection import (
        INTERNAL_INCONSISTENCY, SEVERITY_RANK, _atoms, detect_pairs, extract_rule, norm, parse_condition, text)

logger = logging.getLogger("conflict_resolution")

CRITERIA = ["expert_decisions", "iot_support", "source_corroboration", "prudent_or_specific", "fewer_classifications"]
CRITERIA_LABELS = {
    "expert_decisions": "no conflict with previous expert decisions",
    "iot_support": "IoT overlap >= 0.20",
    "source_corroboration": "more source corroboration",
    "prudent_or_specific": "more prudent / more specific rule",
    "fewer_classifications": "fewer taxonomy classifications",
}
IOT_MIN = 0.20        # minimum IoT overlap (PoC score)
IOT_TIE = 0.10        # IoT differences up to this are treated as a tie
MAX_CAMPS = 10_000    # with more maximal cliques than this we fall back to a greedy partition
UNDER_SPECIFIED_TYPES = {"process flow", "structural"}

# protective actions (stop, close, ...) and checks (inspect, alarm, ...) for the prudence criterion
STOP_RE = re.compile(r"\b(?:stop|halt|shut ?down|shut off|close|closed|disable|reduce|prevent|reject|isolate|e stop)\b")
CHECK_RE = re.compile(r"\b(?:inspect|verify|check|alarm|alert|warn|maintenance|replace)\b")
PROHIB_RE = re.compile(r"\b(?:(?:must|shall|should|may|can|cannot|can ?not) not|never|not allowed|forbidden|prohibited|no\b)")

CLUSTER_SEVERITIES = ("conflict", "potential conflict", "observation")


# --- Input handling -----------------------------------------------------------

def _is_dataframe(obj: Any) -> bool:
    return hasattr(obj, "to_dict") and hasattr(obj, "columns")


def _normalize_records(records: Any) -> List[Dict[str, Any]]:
    if records is None:
        return []
    if _is_dataframe(records):
        records = records.to_dict(orient="records")
    if not isinstance(records, list):
        raise TypeError("records must be a list of statement dictionaries or a pandas DataFrame.")
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise TypeError(f"Record {i} is {type(r).__name__}, expected a dictionary.")
    return records


def _rows(log: Any) -> List[Dict[str, Any]]:
    if log is None:
        return []
    if _is_dataframe(log):
        return log.to_dict(orient="records")
    return [dict(r) for r in log]


# --- Phase 5: conflict graph, clusters and camps ----------------------------------

def _findings(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = record.get("anomalies")
    if value is None:
        value = record.get("conflicts")
    return [f for f in (value or []) if isinstance(f, dict)]


def build_conflict_graph(records: List[Dict[str, Any]], severities: Sequence[str] = CLUSTER_SEVERITIES
                         ) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Map each edge (i, j) with i < j to the findings of that pair. Only detected anomalies are used."""
    id_to_idx = {r.get("statement_id"): i for i, r in enumerate(records)}
    edges: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for i, record in enumerate(records):
        for f in _findings(record):
            j = id_to_idx.get(f.get("statement_id"))
            if j is None or j == i:
                continue
            # findings against expert decisions are not edges between statements
            if f.get("counterpart_is_expert_decision") or f.get("severity") not in severities:
                continue
            bucket = edges.setdefault((min(i, j), max(i, j)), [])
            # each pair is reported on both statements, so skip the copy
            already_there = any(x.get("conflict_type") == f.get("conflict_type")
                                and x.get("severity") == f.get("severity") for x in bucket)
            if not already_there:
                bucket.append(f)
    return edges


def _components(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Connected components with at least one edge (isolated nodes are skipped)."""
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    seen = set()
    components = []
    for start in range(n):
        if start in seen or not adjacency[start]:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node] - seen)
        components.append(sorted(component))
    return components


def build_clusters(records: List[Dict[str, Any]], severities: Sequence[str] = CLUSTER_SEVERITIES) -> List[List[int]]:
    return _components(len(records), list(build_conflict_graph(records, severities)))


def maximal_cliques(nodes: List[int], adjacency: Dict[int, Set[int]], limit: int = MAX_CAMPS) -> Optional[List[List[int]]]:
    """Bron-Kerbosch with pivoting. Returns None if there are more than `limit` cliques."""
    cliques: List[List[int]] = []

    def expand(r: List[int], p: Set[int], x: Set[int]) -> bool:
        if not p and not x:
            cliques.append(sorted(r))
            return len(cliques) <= limit
        pivot = max(p | x, key=lambda v: len(adjacency[v] & p))
        for v in list(p - adjacency[pivot]):
            if not expand(r + [v], p & adjacency[v], x & adjacency[v]):
                return False   # too many cliques, stop
            p.remove(v)
            x.add(v)
        return True

    if expand([], set(nodes), set()):
        return cliques
    return None


def partition_into_camps(cluster: List[int], incompatible: Set[Tuple[int, int]]) -> Tuple[List[List[int]], str]:
    """Camps are the maximal cliques of the compatibility graph (greedy partition above MAX_CAMPS)."""
    adjacency = {}
    for v in cluster:
        adjacency[v] = {w for w in cluster if w != v and (min(v, w), max(v, w)) not in incompatible}

    cliques = maximal_cliques(cluster, adjacency)
    if cliques is not None:
        return sorted(cliques), "maximal_cliques"

    # fallback: put each statement (most compatible first) into the first camp it fits in
    camps: List[List[int]] = []
    for v in sorted(cluster, key=lambda node: -len(adjacency[node])):
        for camp in camps:
            if all(w in adjacency[v] for w in camp):
                camp.append(v)
                break
        else:
            camps.append([v])
    return sorted(sorted(c) for c in camps), "greedy_partition"


# --- IoT overlap (event log / state log) ------------------------------------------

class IoTEvidence:
    """How well the production rules agree with the IoT logs.

    event_log: rows with `event` (the name) and optionally `timestamp` and `case`. Events are read
               in order.
    state_log: rows with numeric variable columns (named like in the conditions), an optional scope
               column (`scope` / `component` / `station`) and text columns (state, mode, action,
               event, ...).

    For Process Flow / Structural rules the score is the share of predecessor occurrences that are
    directly followed by a successor. For state-type rules it is the share of records where the
    numeric condition holds (and the scope matches) that also show the outcome state.
    """

    SCOPE_COLUMNS = ("scope", "component", "station", "event_scope")

    def __init__(self, event_log: Any = None, state_log: Any = None):
        self.sequences: List[List[str]] = []
        events = _rows(event_log)
        if events:
            # only sort if every event has a timestamp, otherwise keep the given order
            if all("timestamp" in e for e in events):
                events = sorted(events, key=lambda e: str(e["timestamp"]))
            by_case: Dict[Any, List[str]] = {}
            for e in events:
                case = e.get("case", e.get("case_id", 0))
                by_case.setdefault(case, []).append(norm(e.get("event")))
            self.sequences = list(by_case.values())

        self.states = []
        for row in _rows(state_log):
            numeric = {}
            words = []
            scope = ""
            for column, value in row.items():
                name = str(column).lower()
                if name in self.SCOPE_COLUMNS:
                    scope = norm(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:   # skips NaN
                    numeric[name] = float(value)
                elif name not in ("timestamp", "time") and isinstance(value, str):
                    words.append(norm(value))
            self.states.append((numeric, " ".join(words), scope))

    @property
    def available(self) -> bool:
        return bool(self.sequences or self.states)

    @staticmethod
    def _alternatives(s: str) -> Set[str]:
        parts = (norm(part) for part in re.split(r"\s+OR\s+|\s+AND\s+", s, flags=re.I))
        return {p for p in parts if p}

    def score(self, rule) -> Optional[float]:
        if rule.category in ("process flow", "structural"):
            return self._event_overlap(rule)
        if rule.category == "temporal and probabilistic":
            return None
        return self._state_overlap(rule)

    def _event_overlap(self, rule) -> Optional[float]:
        predecessors = self._alternatives(rule.condition)
        successors = self._alternatives(rule.outcome)
        total = 0
        hits = 0
        for seq in self.sequences:
            for k, event in enumerate(seq):
                if event not in predecessors:
                    continue
                total += 1
                if k + 1 < len(seq) and seq[k + 1] in successors:
                    hits += 1
        return hits / total if total else None

    def _state_overlap(self, rule) -> Optional[float]:
        if not rule.numeric.parsed:
            return None
        atoms = [a for a in _atoms(rule.outcome) if a]
        holds = 0
        shows = 0
        for numeric, words, scope in self.states:
            if rule.scoped and scope and norm(rule.scope) != scope:
                continue
            if not rule.numeric.holds(numeric):   # False or None (variable missing)
                continue
            holds += 1
            if any(re.search(rf"\b{re.escape(a)}\b", words) for a in atoms):
                shows += 1
        return shows / holds if holds else None


# --- Phase 6: camp profiles and criteria --------------------------------------------

class _Context:
    """Records, parsed rules and IoT evidence, shared by the phase 6 helpers."""

    def __init__(self, records, rules, iot: IoTEvidence):
        self.records = records
        self.rules = rules
        self.iot = iot
        self.ids = [r.get("statement_id") for r in records]
        self._iot_cache: Dict[int, Optional[float]] = {}

    def iot_score(self, i: int) -> Optional[float]:
        if i not in self._iot_cache:
            self._iot_cache[i] = self.iot.score(self.rules[i]) if self.iot.available else None
        return self._iot_cache[i]

    def prudence(self, i: int) -> Tuple[int, int]:
        """(prudence points, specificity) of one rule."""
        rule = self.rules[i]
        outcome = " ".join(_atoms(rule.outcome))
        prohibits = bool(PROHIB_RE.search(outcome))

        # a forbidden stop ("MUST NOT be stopped") isn't a protective action, so it gets no points
        points = 0
        if not prohibits:
            if STOP_RE.search(outcome):
                points += 2
            if CHECK_RE.search(outcome):
                points += 1
            # only stated limits count here - exclusiveness and prohibitions restrict, they don't protect
            limit_markers = [m for m in rule.markers if not PROHIB_RE.search(m) and m != "only"]
            if limit_markers:
                points += 1

        clauses = sum(len(box) for box in rule.numeric.nominal) if rule.numeric.parsed else 0
        return points, clauses + len(_atoms(rule.outcome))

    def corroboration(self, members: Sequence[int]) -> int:
        # statements without a source each count as their own source
        sources = {text(self.records[i].get("source")) or f"?{i}" for i in members}
        # duplicates / subsumed rules from a different source also back the camp up
        extra: Set[Tuple[Any, ...]] = set()
        for i in members:
            own_source = text(self.records[i].get("source"))
            for v in self.records[i].get("valid_discrepancies") or []:
                if v.get("subtype") in ("duplicate", "subsumption") and text(v.get("source")) != own_source:
                    extra.add((v.get("statement_id"),))
        return len(sources) + len(extra)

    def expert_violations(self, members: Sequence[int]) -> int:
        count = 0
        for i in members:
            for f in _findings(self.records[i]):
                if f.get("counterpart_is_expert_decision") and f.get("severity") in ("conflict", "potential conflict"):
                    count += 1
            count += len(self.records[i].get("rejected_equivalents") or [])
        return count

    def classifications(self, members: Sequence[int]) -> int:
        return sum(len(_findings(self.records[i])) for i in members)


def camp_profile(members: List[int], unshared: List[int], ctx: _Context) -> Dict[str, Any]:
    scores = [ctx.iot_score(i) for i in unshared]
    scores = [s for s in scores if s is not None]
    prudence = [ctx.prudence(i) for i in unshared]
    return {
        "statement_ids": [ctx.ids[i] for i in members],
        "unshared_statement_ids": [ctx.ids[i] for i in unshared],
        "expert_decision_violations": ctx.expert_violations(unshared),
        "iot_support": (sum(scores) / len(scores)) if scores else None,
        "iot_rules_evaluated": len(scores),
        "source_corroboration": ctx.corroboration(unshared),
        "independent_sources": len({text(ctx.records[i].get("source")) for i in unshared}),
        "prudence": sum(points for points, _ in prudence),
        "specificity": sum(spec for _, spec in prudence),
        "taxonomy_count": ctx.classifications(unshared),
    }


def _mark_extreme(matched: List[Set[str]], values: List[Any], criterion: str, best=max) -> None:
    """Give `criterion` to every camp with the best value - but only if the camps actually differ."""
    if max(values) > min(values):
        target = best(values)
        for k, value in enumerate(values):
            if value == target:
                matched[k].add(criterion)


def _has_severity(edges: Dict[Tuple[int, int], List[Dict]], severity: str) -> bool:
    return any(f.get("severity") == severity for findings in edges.values() for f in findings)


def evaluate_camps(cluster: List[int], camps: List[List[int]], edges: Dict[Tuple[int, int], List[Dict]], ctx: _Context
                   ) -> Dict[str, Any]:
    """Count the matched criteria per camp (on the unshared statements) and pick a recommendation."""
    n = len(camps)
    # statements in every camp don't help to tell the camps apart
    core = set.intersection(*[set(c) for c in camps]) if n > 1 else set()
    unshared = [[i for i in camp if i not in core] or list(camp) for camp in camps]
    profiles = [camp_profile(camps[k], unshared[k], ctx) for k in range(n)]
    has_conflict = _has_severity(edges, "conflict")

    matched: List[Set[str]] = [set() for _ in range(n)]

    # 1: no conflict with earlier expert decisions
    for k, p in enumerate(profiles):
        if p["expert_decision_violations"] == 0:
            matched[k].add("expert_decisions")

    # 2: IoT support - the best camps, within the tie margin
    supported = [(k, p["iot_support"]) for k, p in enumerate(profiles)
                 if p["iot_support"] is not None and p["iot_support"] >= IOT_MIN]
    if supported:
        best = max(score for _, score in supported)
        for k, score in supported:
            if score >= best - IOT_TIE:
                matched[k].add("iot_support")

    # 3: source corroboration
    _mark_extreme(matched, [p["source_corroboration"] for p in profiles], "source_corroboration")

    # 4: prudence / specificity, only when there is a real conflict
    if has_conflict:
        _mark_extreme(matched, [(p["prudence"], p["specificity"]) for p in profiles], "prudent_or_specific")

    # 5: fewer classifications
    _mark_extreme(matched, [p["taxonomy_count"] for p in profiles], "fewer_classifications", best=min)

    counts = [len(m) for m in matched]
    top = max(counts) if counts else 0
    winners = [k for k, c in enumerate(counts) if c == top]
    recommended = winners if len(winners) == 1 and top > 0 else []

    categories = {ctx.rules[i].category for i in cluster}
    under_specified = categories <= UNDER_SPECIFIED_TYPES and not has_conflict
    status = "recommendation_available" if recommended else "pending"
    only_fewer = False
    if under_specified:
        iot_backed = [k for k, p in enumerate(profiles) if p["iot_support"] is not None and p["iot_support"] >= IOT_MIN]
        if len(iot_backed) > 1:
            recommended = iot_backed
            status = "candidate_split"

    decided_by: List[str] = []
    if recommended and status == "recommendation_available":
        winner = recommended[0]
        others = [k for k in range(n) if k != winner]
        # the criteria the winner has and at least one other camp doesn't
        decided_by = [c for c in CRITERIA if c in matched[winner] and any(c not in matched[k] for k in others)]
        if decided_by == ["fewer_classifications"]:
            # counting taxonomy classifications alone isn't a good enough reason - leave it to the human
            recommended = []
            status = "pending"
            decided_by = []
            only_fewer = True

    return {
        "profiles": profiles,
        "matched": matched,
        "counts": counts,
        "recommended": recommended,
        "status": status,
        "decided_by": decided_by,
        "under_specified": under_specified,
        "has_conflict": has_conflict,
        "unshared": unshared,
        "only_fewer_classifications": only_fewer,
    }


def _reason(ev: Dict[str, Any], camps_ids: List[List[str]]) -> str:
    rec = ev["recommended"]
    if ev["status"] == "candidate_split":
        camps = ", ".join(f"camp {k + 1}" for k in rec)
        return ("The IoT data support more than one camp (overlap >= 0.20): " + camps
                + ". The conflict may result from under-specified conditions; both camps are recommended.")
    if not rec and ev.get("only_fewer_classifications"):
        return ("Only the criterion 'fewer taxonomy classifications' favours one camp; it is not sufficient for a "
                "recommendation. The cluster remains for human review.")
    if not rec:
        counts = ", ".join(f"camp {k + 1}: {c}" for k, c in enumerate(ev["counts"]))
        return f"No camp matches more criteria than all others ({counts}); no recommendation."
    k = rec[0]
    names = "; ".join(CRITERIA_LABELS[c] for c in sorted(ev["matched"][k], key=CRITERIA.index))
    return (f"Camp {k + 1} ({', '.join(camps_ids[k])}) matches the most criteria ({ev['counts'][k]}: {names}), "
            "more than any other camp.")


# --- Building the cluster dicts -------------------------------------------------

_PAIR_FINDING_KEYS = ("conflict_type", "severity", "condition_relation", "outcome_relation", "method", "reason")


def _pair_entries(cluster, edges, ctx, camps, ev) -> List[Dict[str, Any]]:
    recommended_members = {i for k in ev["recommended"] for i in camps[k]}
    has_recommendation = bool(ev["recommended"]) and ev["status"] == "recommendation_available"
    entries = []
    for (i, j), findings in sorted(edges.items()):
        if i not in cluster or j not in cluster:
            continue
        severity = max((f["severity"] for f in findings), key=SEVERITY_RANK.get)

        # if exactly one side of the pair is in the recommended camp, that side is preferred
        preferred = None
        if has_recommendation:
            if i in recommended_members and j not in recommended_members:
                preferred = ctx.ids[i]
            elif j in recommended_members and i not in recommended_members:
                preferred = ctx.ids[j]

        entries.append({
            "statement_ids": [ctx.ids[i], ctx.ids[j]],
            "sources": [text(ctx.records[i].get("source")), text(ctx.records[j].get("source"))],
            "severity_class": severity,
            "findings": [{key: f.get(key) for key in _PAIR_FINDING_KEYS} for f in findings],
            "preferred": preferred,
            "criteria": list(ev["decided_by"]) if preferred else [],
        })
    return entries


def _build_cluster(cluster_id, cluster, camps, method, edges, ctx, ev) -> Dict[str, Any]:
    ids = ctx.ids
    camp_ids = [[ids[i] for i in camp] for camp in camps]
    rec = ev["recommended"]
    keep_idx = sorted({i for k in rec for i in camps[k]})
    keep = [ids[i] for i in keep_idx]
    remove = [ids[i] for i in cluster if i not in keep_idx] if keep else []
    severity = max((f["severity"] for fs in edges.values() for f in fs), key=SEVERITY_RANK.get, default=None)
    reason = _reason(ev, camp_ids)

    recommendations = []
    for i in cluster:
        if keep:
            action = "keep" if ids[i] in keep else "remove"
        else:
            action = "manual_review"
        my_camps = [k for k, camp in enumerate(camps) if i in camp]
        if my_camps:
            matched = sorted(set().union(*[ev["matched"][k] for k in my_camps]), key=CRITERIA.index)
        else:
            matched = []
        recommendations.append({
            "statement_id": ids[i],
            "source": text(ctx.records[i].get("source")),
            "action": action,
            "reason": reason,
            "strategy": "criteria_count",
            "cluster_id": cluster_id,
            "recommended_statement_ids": keep,
            "recommended_rejected_statement_ids": remove,
            "camps_containing_statement": my_camps,
            "criteria": {"matched": matched, "decided_by": ev["decided_by"]},
            "severity_class": severity,
        })

    camp_matches = [
        {"camp": k, "matched": sorted(ev["matched"][k], key=CRITERIA.index), "count": len(ev["matched"][k])}
        for k in range(len(camps))
    ]
    return {
        "cluster_id": cluster_id,
        "kind": "conflict_cluster",
        "statement_ids": [ids[i] for i in cluster],
        "camps": camp_ids,
        "camp_method": method,
        "severity_class": severity,
        "status": ev["status"],
        "strategy": "criteria_count",
        "candidate_split": ev["status"] == "candidate_split",
        "recommended_camps": list(rec),
        "decided_by": ev["decided_by"],
        "camp_profiles": ev["profiles"],
        "camp_matches": camp_matches,
        "pairs": _pair_entries(set(cluster), edges, ctx, camps, ev),
        "recommended_keep": keep,
        "recommended_remove": remove,
        "recommendations": recommendations,
        "reason": reason,
        "under_specification_hint": ev["under_specified"],
        "statements": [ctx.records[i] for i in cluster],
    }


def _statement_issue_cluster(cluster_id, i, finding, ctx) -> Dict[str, Any]:
    """A one-statement 'cluster' for an internal inconsistency. Always left to the expert."""
    sid = ctx.ids[i]
    source = text(ctx.records[i].get("source"))
    severity = finding.get("severity")
    reason = "Internal inconsistency within the statement: " + str(finding.get("reason", ""))
    pair = {
        "statement_ids": [sid],
        "sources": [source],
        "severity_class": severity,
        "findings": [{key: finding.get(key) for key in ("conflict_type", "severity", "method", "reason")}],
        "preferred": None,
        "criteria": [],
    }
    recommendation = {
        "statement_id": sid,
        "source": source,
        "action": "manual_review",
        "reason": reason,
        "strategy": "none",
        "cluster_id": cluster_id,
        "recommended_statement_ids": [],
        "recommended_rejected_statement_ids": [],
        "camps_containing_statement": [0],
        "criteria": {"matched": [], "decided_by": []},
        "severity_class": severity,
    }
    return {
        "cluster_id": cluster_id,
        "kind": "statement_issue",
        "statement_ids": [sid],
        "camps": [[sid]],
        "camp_method": "single",
        "severity_class": severity,
        "status": "pending",
        "strategy": "none",
        "candidate_split": False,
        "recommended_camps": [],
        "decided_by": [],
        "camp_profiles": [],
        "camp_matches": [],
        "pairs": [pair],
        "recommended_keep": [],
        "recommended_remove": [],
        "reason": reason,
        "recommendations": [recommendation],
        "under_specification_hint": False,
        "statements": [ctx.records[i]],
    }


# --- Public API ---------------------------------------------------------------

def _print_camps(c: Dict[str, Any], texts: Dict, sources: Dict) -> None:
    print(f"  camps ({len(c['camps'])}, method: {c.get('camp_method')}):")
    matches = {m["camp"]: m for m in c.get("camp_matches", [])}
    profiles = c.get("camp_profiles", [])
    for k, camp in enumerate(c["camps"]):
        star = "  <== RECOMMENDED" if k in c.get("recommended_camps", []) else ""
        print(f"    Camp {k + 1}: {', '.join(camp)}{star}")
        for sid in camp:
            print(f"        {sid} (source: {sources.get(sid, '?')}): {texts.get(sid, '')}")
        m = matches.get(k, {})
        print(f"        criteria matched ({m.get('count', 0)}): {', '.join(m.get('matched', [])) or 'none'}")
        if k < len(profiles):
            pr = profiles[k]
            iot = pr.get("iot_support")
            iot_text = "n/a" if iot is None else f"{iot:.2f}"
            print(f"        profile: IoT overlap {iot_text}"
                  f", source corroboration {pr.get('source_corroboration')}, "
                  f"taxonomy classifications {pr.get('taxonomy_count')}, "
                  f"expert-decision violations {pr.get('expert_decision_violations', 0)}, "
                  f"prudence {pr.get('prudence', 'n/a')}, specificity {pr.get('specificity', 'n/a')}")


def print_resolution_report(clusters: List[Dict[str, Any]], records: Optional[List[Dict[str, Any]]] = None) -> None:
    """Print the clusters, camps, criteria and recommendations to the console."""
    bar = "=" * 90
    line = "-" * 90
    records = records or []
    texts = {r.get("statement_id"): (r.get("original") or "") for r in records}
    sources = {r.get("statement_id"): r.get("source") for r in records}

    print(f"\n{bar}\nPhase 5/6: {len(clusters)} cluster(s) requiring resolution\n{bar}")
    for c in clusters:
        is_issue = c.get("kind") == "statement_issue"
        title = "STATEMENT ISSUE" if is_issue else "CONFLICT CLUSTER"
        print(f"\n{title} {c['cluster_id']}   severity: {c.get('severity_class')}   status: {c.get('status')}"
              f"   statements: {len(c['statement_ids'])}")
        print(f"  members: {', '.join(c['statement_ids'])}")

        print("  conflict edges:")
        for p in c.get("pairs", []):
            kinds = sorted({f.get("conflict_type") or "?" for f in p.get("findings", [])})
            preference = f"   prefer {p['preferred']}" if p.get("preferred") else "   (no preference)"
            print(f"    {' <-> '.join(p['statement_ids'])}  [{p.get('severity_class')}]  {', '.join(kinds)}{preference}")
            if p.get("criteria"):
                print(f"       deciding criteria: {', '.join(p['criteria'])}")

        if not is_issue:
            _print_camps(c, texts, sources)

        print(line)
        decided = f"  (decided by: {', '.join(c['decided_by'])})" if c.get("decided_by") else ""
        print(f"  RECOMMENDATION: {c.get('status')}{decided}")
        if c.get("recommended_keep"):
            print(f"    keep  : {', '.join(c['recommended_keep'])}")
            print(f"    remove: {', '.join(c['recommended_remove']) or '-'}")
        else:
            print("    no recommendation -> cluster stays pending unless the expert decides")
        if c.get("under_specification_hint"):
            print("    note: under-specified Process Flow/Structural rules; several camps may be valid")
        print(f"    reason: {c.get('reason')}")


def _recheck_cluster(records, cluster, local, options, cluster_severities) -> Set[Tuple[int, int]]:
    """Run phase 2 on the cluster pairs detection didn't look at. Adds new findings to `local`
    and returns the set of incompatible pairs."""
    incompatible = set(local)
    todo = [(i, j) for i, j in itertools.combinations(cluster, 2) if (i, j) not in local]
    if not todo:
        return incompatible
    for (i, j), res in detect_pairs(records, todo, **options).items():
        if res["kind"] != "finding" or res["severity"] not in cluster_severities:
            continue
        local[(i, j)] = [{
            "conflict_type": res["category"],
            "severity": res["severity"],
            "condition_relation": res["condition_relation"],
            "outcome_relation": res["outcome_relation"],
            "method": res["method"],
            "reason": res["reason"],
            "newly_found": True,
        }]
        incompatible.add((i, j))
    return incompatible


def apply_recommendations(records: Any, strategy_chain: Optional[Sequence[str]] = None, *, event_log: Any = None,
                          state_log: Any = None, detection_kwargs: Optional[Dict[str, Any]] = None,
                          exhaustive_compatibility: bool = True,
                          cluster_severities: Sequence[str] = CLUSTER_SEVERITIES,
                          verbose: bool = False
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Cluster the detected anomalies and create resolution recommendations (advisory only).

    records:             output of detect_conflicts()
    strategy_chain:      old argument (confidence/corroboration/conservative). It is ignored now,
                         the camp with the most matched criteria is recommended.
    event_log/state_log: IoT logs for the overlap criterion (DataFrame or list of dicts, see IoTEvidence)
    detection_kwargs:    options for the exhaustive compatibility check. Pass the same options you
                         used for detect_conflicts (default: the offline heuristic backend).
    verbose:             print a console report of the clusters, camps, criteria and recommendations

    Returns (records, clusters).
    """
    if strategy_chain:
        logger.info("strategy_chain %s is ignored; recommendations use the criteria count.", list(strategy_chain))
    records = _normalize_records(records)
    if not records:
        return [], []

    # drop results from an earlier run
    for r in records:
        r.pop("recommendation", None)
        r.pop("resolution", None)

    rules = [extract_rule(r, i) for i, r in enumerate(records)]
    ctx = _Context(records, rules, IoTEvidence(event_log, state_log))
    edges = build_conflict_graph(records, cluster_severities)
    components = _components(len(records), list(edges))
    options = dict(detection_kwargs or {})
    options.setdefault("backend", "heuristic")

    clusters: List[Dict[str, Any]] = []
    for cluster in components:
        local = {(i, j): fs for (i, j), fs in edges.items() if i in cluster and j in cluster}
        if exhaustive_compatibility:
            incompatible = _recheck_cluster(records, cluster, local, options, cluster_severities)
        else:
            incompatible = set(local)
        camps, method = partition_into_camps(cluster, incompatible)
        ev = evaluate_camps(cluster, camps, local, ctx)
        clusters.append(_build_cluster(len(clusters) + 1, cluster, camps, method, local, ctx, ev))

    # statements outside any cluster can still contradict themselves
    clustered = {i for c in components for i in c}
    for i, record in enumerate(records):
        if i in clustered:
            continue
        for f in _findings(record):
            if f.get("conflict_type") == INTERNAL_INCONSISTENCY and f.get("severity") in cluster_severities:
                clusters.append(_statement_issue_cluster(len(clusters) + 1, i, f, ctx))
                clustered.add(i)
                break

    by_id = {r.get("statement_id"): r for r in records}
    for cluster in clusters:
        for rec in cluster["recommendations"]:
            r = by_id[rec["statement_id"]]
            r["recommendation"] = rec
            r["resolution"] = {
                "status": "recommended",
                "cluster_id": cluster["cluster_id"],
                "cluster_members": cluster["statement_ids"],
                "method": cluster["strategy"],
                "recommendation_only": True,
            }

    # everything else is conflict-free and simply kept
    for r in records:
        if "recommendation" in r:
            continue
        sid = r.get("statement_id")
        r["recommendation"] = {
            "statement_id": sid,
            "source": text(r.get("source")),
            "action": "keep",
            "reason": "No conflict was detected for this expert statement.",
            "strategy": "no_conflict",
            "cluster_id": None,
            "recommended_statement_ids": [sid],
            "recommended_rejected_statement_ids": [],
            "camps_containing_statement": [],
            "criteria": {"matched": [], "decided_by": []},
        }
        r["resolution"] = {
            "status": "not_in_conflict",
            "cluster_id": None,
            "cluster_members": [],
            "method": "no_conflict",
            "recommendation_only": True,
        }

    logger.info("%d cluster(s); recommendations created.", len(clusters))
    if verbose:
        print_resolution_report(clusters, records)
    return records, clusters


def apply_resolutions(records: Any, strategy_chain: Optional[Sequence[str]] = None, **kwargs):
    """Old name, kept for backward compatibility (only creates recommendations)."""
    return apply_recommendations(records, strategy_chain, **kwargs)