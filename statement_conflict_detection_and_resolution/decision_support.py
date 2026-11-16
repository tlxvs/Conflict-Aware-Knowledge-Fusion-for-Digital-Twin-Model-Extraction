"""
decision_support.py

"""
from __future__ import annotations

import json
import logging
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

logger = logging.getLogger("decision_support")

# =============================================================================
# Styling
# =============================================================================

SEVERITY_STYLES: Dict[str, Dict[str, str]] = {
    "conflict":           {"label": "CONFLICT",           "bg": "#fadbd8", "fg": "#922b21", "accent": "#e74c3c"},
    "potential conflict": {"label": "POTENTIAL CONFLICT", "bg": "#fdebd0", "fg": "#a04000", "accent": "#e67e22"},
    "observation":        {"label": "OBSERVATION",        "bg": "#d6eaf8", "fg": "#21618c", "accent": "#5dade2"},
    "anomaly":            {"label": "ANOMALY",            "bg": "#ebedef", "fg": "#566573", "accent": "#95a5a6"},
}
SEVERITY_RANK = {"conflict": 0, "potential conflict": 1, "observation": 2, "anomaly": 3}
NO_FINDING_ACCENT = "#58d68d"

STATUS_STYLES = {
    "recommendation_available": {"label": "RECOMMENDATION AVAILABLE", "bg": "#d5f5e3", "fg": "#196f3d"},
    "candidate_split":          {"label": "CAMPS SUPPORTED BY IoT DATA", "bg": "#e8daef", "fg": "#6c3483"},
    "pending":                  {"label": "NO RECOMMENDATION",        "bg": "#ebedef", "fg": "#566573"},
}
ACTION_STYLES = {
    "keep":          {"label": "SYSTEM: KEEP",          "bg": "#d5f5e3", "fg": "#196f3d"},
    "remove":        {"label": "SYSTEM: REMOVE",        "bg": "#fadbd8", "fg": "#922b21"},
    "manual_review": {"label": "SYSTEM: MANUAL REVIEW", "bg": "#fdebd0", "fg": "#a04000"},
}
CRITERIA_LABELS = {
    "expert_decisions":      "no conflict with previous expert decisions",
    "iot_support":           "IoT overlap >= 0.20",
    "source_corroboration":  "more source corroboration",
    "prudent_or_specific":   "more prudent / more specific rule",
    "fewer_classifications": "fewer taxonomy classifications",
}
INTEGRATION_MODES = {"decision (OR)": "decision", "parallelism": "parallelism"}
DEFAULT_INTEGRATION = "decision (OR)"

STATEMENT_TEXT_FIELDS = ["original", "text", "statement", "sentence"]
DERIVED_KEYS = {"anomalies", "conflicts", "valid_discrepancies", "rejected_equivalents", "has_anomaly", "has_conflict",
                "has_severe_conflict", "max_severity", "resulting_severity", "resulting_anomaly_category",
                "resulting_anomaly_categories", "resulting_severity_classes", "resulting_classifications",
                "duplication_count", "duplicate_statement_ids", "recommendation", "resolution"}


def _sev_style(severity: Optional[str]) -> Dict[str, str]:
    return SEVERITY_STYLES.get(severity or "", SEVERITY_STYLES["anomaly"])


def _sev_rank(severity: Optional[str]) -> int:
    return SEVERITY_RANK.get(severity or "", SEVERITY_RANK["anomaly"])


def _severity_of(finding: Dict[str, Any]) -> Optional[str]:
    return finding.get("severity") or finding.get("severity_class")


# =============================================================================
# Text helpers (no Tk needed)
# =============================================================================

def _present(statement: Dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = statement.get(field)
        if value is not None and value == value and value != "":
            return value
    return None


def formalization_text(statement: Dict[str, Any]) -> str:
    value = _present(statement, "formalization", "formalized_knowledge")
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def statement_text(statement: Dict[str, Any]) -> str:
    """Original statement; the formalization if no original text exists."""
    for field in STATEMENT_TEXT_FIELDS:
        value = statement.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return formalization_text(statement)


def criteria_text(names) -> str:
    return "; ".join(CRITERIA_LABELS.get(n, n) for n in names) if names else "none"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def evidence_summary(finding: Dict[str, Any]) -> str:
    """One-line summary of the evidence stored by conflict_detection."""
    ev = finding.get("evidence") or {}
    parts: List[str] = []
    if finding.get("condition_relation") not in (None, "n/a"):
        parts.append(f"conditions {finding['condition_relation']}, outcomes {finding.get('outcome_relation')}")
    for side in ("condition", "outcome"):
        detail = ev.get(side) or {}
        if detail.get("reason"):
            parts.append(f"{side}: {detail['reason']} [{detail.get('method', '')}]")
    nli = ev.get("nli_probabilities")
    if isinstance(nli, dict) and "contradiction" in nli:
        text = f"NLI contradiction {_num(nli['contradiction'])}"
        if "entailment" in nli:
            text += f", entailment {_num(nli['entailment'])}"
        if "nli_threshold" in ev:
            text += f" (threshold {_num(ev['nli_threshold'])})"
        parts.append(text)
    if finding.get("exclusiveness_markers"):
        parts.append("exclusiveness stated: " + ", ".join(finding["exclusiveness_markers"]))
    if finding.get("counterpart_is_expert_decision"):
        parts.append(f"previous expert decision ({finding.get('expert_decision', '?')})")
    if not parts and finding.get("reason"):
        parts.append(str(finding["reason"]))
    elif finding.get("reason"):
        parts.append(str(finding["reason"]))
    return "  •  ".join(parts)


def profile_line(profile: Dict[str, Any]) -> str:
    iot = profile.get("iot_support")
    iot_text = "n/a" if iot is None else f"{iot:.2f} ({profile.get('iot_rules_evaluated', 0)} rules)"
    return (f"IoT overlap {iot_text}  •  source corroboration {profile.get('source_corroboration', '?')}  •  "
            f"taxonomy classifications {profile.get('taxonomy_count', '?')}  •  prudence {profile.get('prudence', 'n/a')}, "
            f"specificity {profile.get('specificity', 'n/a')}  •  expert-decision violations "
            f"{profile.get('expert_decision_violations', 0)}")


def worst_severity(statement: Dict[str, Any]) -> Optional[str]:
    findings = statement.get("anomalies") or []
    return min((_severity_of(f) for f in findings), key=_sev_rank) if findings else None


# =============================================================================
# Dialog
# =============================================================================

class ConflictResolver(tk.Toplevel):
    """Single popup with all clusters that require resolution."""

    WRAP = 1040

    def __init__(self, parent, clusters, cluster_numbers=None, cluster_metadata=None):
        super().__init__(parent)
        self.clusters = clusters
        self.cluster_numbers = cluster_numbers or [c.get("cluster_id", i) for i, c in enumerate(clusters, start=1)]
        self.cluster_metadata = cluster_metadata or []
        self.result = None
        self._current_cluster: Dict[str, Any] = {}

        self.keep_vars: Dict[str, tk.BooleanVar] = {}         # statement_id -> accepted?
        self.decision_vars: Dict[Any, tk.StringVar] = {}      # cluster -> selected / all / none / pending
        self.camp_vars: Dict[Any, List[tk.BooleanVar]] = {}   # cluster -> accepted camps
        self.integration_vars: Dict[Any, tk.StringVar] = {}   # cluster -> decision (OR) / parallelism

        self.title("Conflict Resolution — Human Decision Support")
        self.configure(bg="#f4f6f8")
        width = min(1250, self.winfo_screenwidth() - 100)
        height = min(760, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}")
        self.minsize(900, 600)
        self._build_ui()
        try:
            self.grab_set()
        except tk.TclError:
            logger.debug("grab_set failed (window not yet viewable)")
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    # ------------------------------------------------------------------ small widgets

    @staticmethod
    def _chip(parent, text, bg, fg, **pack):
        label = tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"), bg=bg, fg=fg, padx=8, pady=3)
        label.pack(**pack)
        return label

    def _toggle(self, parent, content_builder, show_text, hide_text, **pack):
        button = ttk.Button(parent, text=show_text)
        button.pack(**pack)
        frame = tk.Frame(parent, bg=parent["bg"])
        state = {"open": False, "built": False}

        def toggle():
            if state["open"]:
                frame.pack_forget()
                button.configure(text=show_text)
            else:
                if not state["built"]:
                    content_builder(frame)
                    state["built"] = True
                frame.pack(fill="x", after=button, padx=pack.get("padx", 0))
                button.configure(text=hide_text)
            state["open"] = not state["open"]
        button.configure(command=toggle)

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        self._build_header()
        self._build_buttons()
        outer = tk.Frame(self, bg="#f4f6f8")
        outer.pack(fill="both", expand=True, padx=20, pady=15)
        self.canvas = tk.Canvas(outer, bg="#f4f6f8", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg="#f4f6f8")
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(3, "units"))
        for number, cluster in zip(self.cluster_numbers, self.clusters):
            self._add_cluster(self.content, cluster, number)

    def _on_mousewheel(self, event):
        step = -event.delta if sys.platform == "darwin" else -event.delta / 120
        self.canvas.yview_scroll(int(step), "units")

    def destroy(self):
        try:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.unbind_all(sequence)
        except tk.TclError:
            pass
        super().destroy()

    def _build_header(self):
        header = tk.Frame(self, bg="#17202a")
        header.pack(fill="x")
        tk.Label(header, text="Conflict Resolution", font=("Segoe UI", 19, "bold"), fg="white",
                 bg="#17202a").pack(anchor="w", padx=25, pady=(14, 0))
        total = sum(len(c.get("statement_ids", [])) for c in self.clusters)
        by_severity: Dict[str, int] = {}
        for c in self.clusters:
            sev = c.get("severity_class") or "anomaly"
            by_severity[sev] = by_severity.get(sev, 0) + 1
        summary = "  •  ".join(f"{n} {sev}" for sev, n in sorted(by_severity.items(), key=lambda t: _sev_rank(t[0])))
        tk.Label(header, text=f"{len(self.clusters)} clusters requiring resolution ({summary})  •  {total} statements",
                 font=("Segoe UI", 10), fg="#d5d8dc", bg="#17202a").pack(anchor="w", padx=25)
        tk.Label(header, text="System recommendations are advisory only. Clusters without a decision stay pending and are "
                              "excluded from the active simulation rule base.", font=("Segoe UI", 9), fg="#aeb6bf",
                 bg="#17202a").pack(anchor="w", padx=25, pady=(2, 6))
        legend = tk.Frame(header, bg="#17202a")
        legend.pack(anchor="w", padx=25, pady=(0, 10))
        tk.Label(legend, text="Severity:", font=("Segoe UI", 9), fg="#aeb6bf", bg="#17202a").pack(side="left", padx=(0, 6))
        for key in ("conflict", "potential conflict", "observation"):
            style = SEVERITY_STYLES[key]
            self._chip(legend, style["label"], style["bg"], style["fg"], side="left", padx=3)

    def _build_buttons(self):
        frame = tk.Frame(self, bg="#e5e7e9", height=70)
        frame.pack(fill="x", side="bottom")
        frame.pack_propagate(False)
        ttk.Button(frame, text="Cancel", command=self.cancel).pack(side="right", padx=(5, 20), pady=15)
        ttk.Button(frame, text="Apply All Decisions", command=self.confirm).pack(side="right", padx=5, pady=15)

    # ------------------------------------------------------------------ cluster

    @staticmethod
    def _recommended_ids(cluster) -> set:
        camps = cluster.get("camps") or []
        ids: set = set()
        for index in cluster.get("recommended_camps") or []:
            if 0 <= index < len(camps):
                ids.update(camps[index])
        return ids or set(cluster.get("recommended_keep") or [])

    def _add_cluster(self, parent, cluster, number):
        sev = _sev_style(cluster.get("severity_class"))
        status = cluster.get("status", "pending")
        statements = cluster.get("statements", [])
        self._current_cluster = cluster

        frame = tk.Frame(parent, bg="#ffffff", highlightbackground=sev["accent"], highlightthickness=2)
        frame.pack(fill="x", pady=(0, 22))
        header = tk.Frame(frame, bg="#2c3e50")
        header.pack(fill="x")
        title = "Statement Issue" if cluster.get("kind") == "statement_issue" else "Conflict Cluster"
        tk.Label(header, text=f"{title} {number}", font=("Segoe UI", 14, "bold"), fg="white",
                 bg="#2c3e50").pack(side="left", padx=15, pady=10)
        self._chip(header, sev["label"], sev["bg"], sev["fg"], side="left", padx=4)
        st = STATUS_STYLES.get(status, STATUS_STYLES["pending"])
        self._chip(header, st["label"], st["bg"], st["fg"], side="left", padx=4)
        tk.Label(header, text=f"{len(statements)} statements  •  {len(cluster.get('camps', []))} camps",
                 font=("Segoe UI", 9), fg="#d5d8dc", bg="#2c3e50").pack(side="right", padx=15)

        self._add_recommendation(frame, cluster)
        self._add_pairs(frame, cluster, {x.get("statement_id"): x for x in statements})
        self._add_camps(frame, cluster, number)

        recommended_ids = self._recommended_ids(cluster)
        decision_frame = tk.Frame(frame, bg="#f4f6f8")
        decision_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.decision_vars[number] = tk.StringVar(value="selected" if recommended_ids else "pending")
        tk.Label(decision_frame, text="Decision:", font=("Segoe UI", 10, "bold"), fg="#17202a",
                 bg="#f4f6f8").pack(side="left", padx=(0, 10))
        for label, value in [("Accept selected camp(s)", "selected"), ("Accept all", "all"),
                             ("Reject all", "none"), ("Defer (pending)", "pending")]:
            ttk.Radiobutton(decision_frame, text=label, variable=self.decision_vars[number], value=value).pack(side="left", padx=5)
        if recommended_ids:
            ttk.Button(decision_frame, text="Use system recommendation",
                       command=lambda c=cluster, n=number: self._use_recommendation(c, n)).pack(side="left", padx=8)
        integration = tk.Frame(frame, bg="#f4f6f8")
        integration.pack(fill="x", padx=15, pady=(0, 10))
        tk.Label(integration, text="If several camps are accepted, integrate them into the SPN model as:",
                 font=("Segoe UI", 9), fg="#17202a", bg="#f4f6f8").pack(side="left", padx=(0, 8))
        self.integration_vars[number] = tk.StringVar(value=DEFAULT_INTEGRATION)
        ttk.Combobox(integration, textvariable=self.integration_vars[number], values=list(INTEGRATION_MODES),
                     state="readonly", width=16).pack(side="left")

        for statement in statements:
            self._current_cluster = cluster
            self._add_statement(frame, statement)

    def _use_recommendation(self, cluster, number):
        recommended = set(cluster.get("recommended_camps") or [])
        for index, var in enumerate(self.camp_vars.get(number, [])):
            var.set(index in recommended)
        self._select_statements(cluster, self._recommended_ids(cluster), number)

    def _camp_toggled(self, cluster, number):
        camps = cluster.get("camps") or []
        ids: set = set()
        for index, var in enumerate(self.camp_vars.get(number, [])):
            if var.get() and index < len(camps):
                ids.update(camps[index])
        self._select_statements(cluster, ids, number)

    def _select_statements(self, cluster, keep_ids, number):
        for statement in cluster.get("statements", []):
            sid = statement.get("statement_id")
            if sid in self.keep_vars:
                self.keep_vars[sid].set(sid in keep_ids)
        self.decision_vars[number].set("selected")

    # ------------------------------------------------------------------ recommendation

    def _add_recommendation(self, parent, cluster):
        frame = tk.Frame(parent, bg="#eaf2f8", highlightbackground="#3498db", highlightthickness=1)
        frame.pack(fill="x", padx=15, pady=15)
        keep, remove = cluster.get("recommended_keep") or [], cluster.get("recommended_remove") or []
        status = cluster.get("status", "pending")
        recommended = [i + 1 for i in cluster.get("recommended_camps") or []]

        tk.Label(frame, text="SYSTEM RECOMMENDATION (advisory)", font=("Segoe UI", 9, "bold"), fg="#21618c",
                 bg="#eaf2f8").pack(anchor="w", padx=15, pady=(10, 3))
        if cluster.get("kind") != "statement_issue":
            tk.Label(frame, text="Method: the camp that matches the most criteria is recommended (no weighted score)",
                     font=("Segoe UI", 9), fg="#566573", bg="#eaf2f8", justify="left",
                     wraplength=self.WRAP).pack(anchor="w", padx=15, pady=(0, 4))

        if status == "candidate_split" and recommended:
            tk.Label(frame, text="Recommended: camps " + ", ".join(map(str, recommended)) + ". The IoT data support more than one "
                     "camp (under-specified conditions). If you accept several camps, choose below how they enter the SPN model.",
                     font=("Segoe UI", 10, "bold"), fg="#6c3483", bg="#eaf2f8", justify="left",
                     wraplength=self.WRAP).pack(anchor="w", padx=15, pady=2)
        elif recommended and (keep or remove):
            tk.Label(frame, text=f"Recommended camp {', '.join(map(str, recommended))}  →  KEEP: " + ", ".join(keep),
                     font=("Segoe UI", 10, "bold"), fg="#196f3d", bg="#eaf2f8", justify="left",
                     wraplength=self.WRAP).pack(anchor="w", padx=15, pady=2)
            if remove:
                tk.Label(frame, text="REMOVE: " + ", ".join(remove), font=("Segoe UI", 10), fg="#922b21", bg="#eaf2f8",
                         justify="left", wraplength=self.WRAP).pack(anchor="w", padx=15, pady=2)
        else:
            tk.Label(frame, text="No recommendation. The cluster stays pending unless you decide; pending statements are "
                                 "excluded from the active simulation rule base.", font=("Segoe UI", 9, "italic"),
                     fg="#7b241c", bg="#eaf2f8", justify="left", wraplength=self.WRAP).pack(anchor="w", padx=15, pady=2)
        if cluster.get("reason"):
            tk.Label(frame, text="Reason: " + str(cluster["reason"]), font=("Segoe UI", 10), fg="#17202a", bg="#eaf2f8",
                     justify="left", wraplength=self.WRAP).pack(anchor="w", padx=15, pady=(3, 4))
        if cluster.get("under_specification_hint"):
            tk.Label(frame, text="All anomalies here are potential conflicts or observations of Process Flow / Structural rules: "
                                 "both successors may be valid under uncaptured sub-conditions.",
                     font=("Segoe UI", 9, "italic"), fg="#a04000", bg="#eaf2f8", justify="left",
                     wraplength=self.WRAP).pack(anchor="w", padx=15, pady=(2, 0))
        tk.Frame(frame, bg="#eaf2f8", height=8).pack()

    # ------------------------------------------------------------------ pairs

    def _add_pairs(self, parent, cluster, by_id):
        pairs = cluster.get("pairs") or []
        if not pairs:
            return
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", padx=15, pady=(0, 10))
        tk.Label(frame, text=f"ANOMALIES ({len(pairs)})", font=("Segoe UI", 9, "bold"), fg="#566573",
                 bg="white").pack(anchor="w", padx=12, pady=(4, 4))
        for pair in pairs:
            ids = list(pair.get("statement_ids") or [])
            s = _sev_style(pair.get("severity_class"))
            row = tk.Frame(frame, bg=s["bg"], highlightbackground=s["accent"], highlightthickness=1)
            row.pack(fill="x", padx=12, pady=3)
            head = tk.Frame(row, bg=s["bg"])
            head.pack(fill="x", padx=8, pady=(5, 0))
            self._chip(head, s["label"], "white", s["fg"], side="left")
            kinds = sorted({f.get("conflict_type") or "Anomaly" for f in pair.get("findings") or []})
            tk.Label(head, text="  " + ", ".join(kinds) + ("   (" + " ↔ ".join(ids) + ")" if len(ids) > 1 else f"   ({ids[0]})"),
                     font=("Segoe UI", 9, "bold"), fg=s["fg"], bg=s["bg"]).pack(side="left")
            for sid, source in zip(ids, pair.get("sources") or []):
                tk.Label(row, text=f"{sid}: {statement_text(by_id.get(sid, {}))}", font=("Segoe UI", 9), fg="#17202a",
                         bg=s["bg"], justify="left", wraplength=self.WRAP - 40).pack(anchor="w", padx=14)
                tk.Label(row, text=f"source: {source or '—'}", font=("Segoe UI", 8), fg="#566573", bg=s["bg"],
                         justify="left", wraplength=self.WRAP - 40).pack(anchor="w", padx=26)
            for f in pair.get("findings") or []:
                detail = evidence_summary(f)
                if detail:
                    tk.Label(row, text=detail, font=("Segoe UI", 8), fg="#566573", bg=s["bg"], justify="left",
                             wraplength=self.WRAP - 40).pack(anchor="w", padx=14)
            preferred = pair.get("preferred")
            rec = f"Recommendation: prefer {preferred}" if preferred else "Recommendation: none for this pair"
            tk.Label(row, text=rec + "  •  criteria: " + criteria_text(pair.get("criteria")), font=("Segoe UI", 9, "bold"),
                     fg="#196f3d" if preferred else "#7b241c", bg=s["bg"], justify="left",
                     wraplength=self.WRAP - 40).pack(anchor="w", padx=14, pady=(2, 5))

    # ------------------------------------------------------------------ camps

    def _add_camps(self, parent, cluster, number):
        camps = cluster.get("camps") or []
        if not camps or cluster.get("kind") == "statement_issue":
            return
        profiles = cluster.get("camp_profiles") or []
        matches = {m.get("camp"): m for m in cluster.get("camp_matches") or []}
        recommended = set(cluster.get("recommended_camps") or [])

        frame = tk.Frame(parent, bg="#f8f9f9")
        frame.pack(fill="x", padx=15, pady=(0, 10))
        tk.Label(frame, text="CAMPS (mutually compatible statements) - tick the camp(s) you accept", font=("Segoe UI", 9, "bold"),
                 fg="#566573", bg="#f8f9f9").pack(anchor="w", padx=12, pady=(8, 5))
        self.camp_vars[number] = []
        for index, camp in enumerate(camps):
            var = tk.BooleanVar(value=index in recommended)
            self.camp_vars[number].append(var)
            row = tk.Frame(frame, bg="#f8f9f9")
            row.pack(fill="x", padx=20, pady=3)
            top = tk.Frame(row, bg="#f8f9f9")
            top.pack(fill="x")
            tk.Checkbutton(top, text=f"Camp {index + 1}: " + ", ".join(str(s) for s in camp), variable=var,
                           command=lambda c=cluster, n=number: self._camp_toggled(c, n), font=("Segoe UI", 9, "bold"),
                           fg="#2c3e50", bg="#f8f9f9", activebackground="#f8f9f9", selectcolor="white", justify="left",
                           wraplength=self.WRAP - 260).pack(side="left")
            if index in recommended:
                self._chip(top, "★ RECOMMENDED", "#d5f5e3", "#196f3d", side="left", padx=8)
            match = matches.get(index) or {}
            self._chip(top, f"{match.get('count', 0)} criteria matched", "#d6eaf8", "#21618c", side="left", padx=4)
            tk.Label(row, text="Matched: " + criteria_text(match.get("matched")), font=("Segoe UI", 9), fg="#196f3d",
                     bg="#f8f9f9", justify="left", wraplength=self.WRAP).pack(anchor="w", padx=26)
            if index < len(profiles):
                tk.Label(row, text=profile_line(profiles[index]), font=("Segoe UI", 8), fg="#7f8c8d", bg="#f8f9f9",
                         justify="left", wraplength=self.WRAP).pack(anchor="w", padx=26)
        tk.Label(frame, text="Camps are compared on their unshared statements. Statements can also be ticked individually below.",
                 font=("Segoe UI", 9, "italic"), fg="#7f8c8d", bg="#f8f9f9", justify="left",
                 wraplength=self.WRAP).pack(anchor="w", padx=20, pady=(5, 8))

    # ------------------------------------------------------------------ statement

    def _add_statement(self, parent, statement):
        sid = statement.get("statement_id")
        if not sid:
            return
        recommendation = statement.get("recommendation") if isinstance(statement.get("recommendation"), dict) else {}
        action = recommendation.get("action", "")
        findings = statement.get("anomalies") or []
        worst = worst_severity(statement)
        accent = _sev_style(worst)["accent"] if worst else NO_FINDING_ACCENT

        keep_var = tk.BooleanVar(value=sid in self._recommended_ids(self._current_cluster))
        self.keep_vars[sid] = keep_var

        card = tk.Frame(parent, bg="white", highlightbackground="#d5d8dc", highlightthickness=1)
        card.pack(fill="x", padx=15, pady=6)
        tk.Frame(card, bg=accent, width=7).pack(side="left", fill="y")
        body = tk.Frame(card, bg="white")
        body.pack(side="left", fill="both", expand=True)

        top = tk.Frame(body, bg="white")
        top.pack(fill="x", padx=15, pady=(12, 6))
        tk.Checkbutton(top, text=f"Accept {sid}", variable=keep_var, font=("Segoe UI", 10, "bold"), fg="#17202a", bg="white",
                       activebackground="white", selectcolor="white").pack(side="left")
        style = ACTION_STYLES.get(action)
        if style:
            self._chip(top, style["label"], style["bg"], style["fg"], side="left", padx=(10, 0))
        for k in recommendation.get("camps_containing_statement") or []:
            self._chip(top, f"CAMP {k + 1}", "#d6eaf8", "#21618c", side="left", padx=(6, 0))
        counts: Dict[str, int] = {}
        for f in findings:
            key = _severity_of(f) or "anomaly"
            counts[key] = counts.get(key, 0) + 1
        for sev in sorted(counts, key=_sev_rank):
            s = _sev_style(sev)
            self._chip(top, f"{counts[sev]} {s['label']}", s["bg"], s["fg"], side="left", padx=(6, 0))
        tk.Label(top, text=_present(statement, "category", "knowledge_type") or "", font=("Segoe UI", 9), fg="#566573",
                 bg="#ebedef", padx=8, pady=3).pack(side="right")

        tk.Label(body, text="Original statement", font=("Segoe UI", 9, "bold"), fg="#566573", bg="white").pack(anchor="w", padx=15)
        tk.Label(body, text=statement_text(statement), font=("Segoe UI", 10), fg="#17202a", bg="white", justify="left",
                 wraplength=self.WRAP - 40).pack(anchor="w", padx=15, pady=(2, 8))
        formal = formalization_text(statement)
        if formal and formal != statement_text(statement):
            tk.Label(body, text="Rule", font=("Segoe UI", 9, "bold"), fg="#566573", bg="white").pack(anchor="w", padx=15)
            tk.Label(body, text=formal, font=("Consolas", 9), fg="#1b2631", bg="#f8f9f9", justify="left",
                     wraplength=self.WRAP - 40, padx=10, pady=8).pack(fill="x", padx=15, pady=(2, 8))

        info = []
        source = recommendation.get("source") or _present(statement, "source")
        if source:
            info.append(f"Source: {source}")
        criteria = recommendation.get("criteria") or {}
        if criteria.get("matched"):
            info.append("Criteria matched by its camp: " + criteria_text(criteria["matched"]))
        if criteria.get("decided_by"):
            info.append("Decisive: " + criteria_text(criteria["decided_by"]))
        if info:
            tk.Label(body, text="  •  ".join(info), font=("Segoe UI", 8), fg="#566573", bg="white", justify="left",
                     wraplength=self.WRAP - 40).pack(anchor="w", padx=15, pady=(0, 6))

        if findings:
            def build(inner):
                for f in findings:
                    s = _sev_style(_severity_of(f))
                    row = tk.Frame(inner, bg="white")
                    row.pack(fill="x", pady=3)
                    self._chip(row, s["label"], s["bg"], s["fg"], side="left", padx=(0, 8), anchor="n")
                    head = f.get("conflict_type") or f.get("anomaly_category") or "Anomaly"
                    if f.get("statement_id"):
                        head += "  ↔  " + str(f["statement_id"])
                    col = tk.Frame(row, bg="white")
                    col.pack(side="left", fill="x", expand=True)
                    tk.Label(col, text=head, font=("Segoe UI", 9, "bold"), fg=s["fg"], bg="white", justify="left",
                             wraplength=self.WRAP - 160).pack(anchor="w")
                    summary = evidence_summary(f)
                    if summary:
                        tk.Label(col, text=summary, font=("Segoe UI", 8), fg="#7f8c8d", bg="white", justify="left",
                                 wraplength=self.WRAP - 160).pack(anchor="w")
            self._toggle(body, build, f"Show {len(findings)} finding(s)", "Hide findings", anchor="w", padx=15, pady=(0, 10))
        else:
            tk.Label(body, text="No anomalies recorded for this statement.", font=("Segoe UI", 9, "italic"), fg="#7f8c8d",
                     bg="white").pack(anchor="w", padx=15, pady=(0, 10))

    # ------------------------------------------------------------------ confirm / cancel

    def confirm(self):
        decisions = []
        for number, cluster in zip(self.cluster_numbers, self.clusters):
            var = self.decision_vars.get(number)
            if var is None:
                continue
            decision = var.get()
            ids = list(cluster.get("statement_ids", []))
            camps = [list(c) for c in cluster.get("camps") or []]
            if decision == "selected":
                kept = [s for s in ids if s in self.keep_vars and self.keep_vars[s].get()]
                if not kept:
                    messagebox.showwarning("No statements selected",
                                           f"Cluster {number}: select at least one camp or statement, or choose "
                                           "'Reject all' or 'Defer (pending)'.", parent=self)
                    return
            elif decision == "all":
                kept = list(ids)
            else:
                kept = []

            ticked = [c for c, v in zip(camps, self.camp_vars.get(number, [])) if v.get()]
            if decision == "all":
                accepted_camps = camps
            elif decision == "selected":
                pool = ticked or camps
                accepted_camps = [c for c in pool if set(c) <= set(kept)]
            else:
                accepted_camps = []
            integration = None
            if len(accepted_camps) > 1:
                integration = INTEGRATION_MODES.get(self.integration_vars[number].get(), "decision")
            recommended = self._recommended_ids(cluster)
            decisions.append({
                "cluster": number, "decision": decision, "cluster_severity": cluster.get("severity_class"),
                "status": cluster.get("status"), "kept_statement_ids": kept,
                "rejected_statement_ids": [s for s in ids if s not in kept] if decision != "pending" else [],
                "accepted_camps": accepted_camps, "integration": integration,
                "followed_recommendation": bool(recommended) and set(kept) == recommended,
            })
        self.result = decisions
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


# =============================================================================
# Public function
# =============================================================================

def _as_expert_decision(statement: Dict[str, Any], verdict: str) -> Dict[str, Any]:
    decision = {k: v for k, v in statement.items() if k not in DERIVED_KEYS}
    decision["origin_statement_id"] = statement["statement_id"]
    decision["statement_id"] = f"EXP_{statement['statement_id']}"
    decision["decision"] = verdict
    return decision


def resolve_pending_clusters(pending_clusters, resolved_knowledge):
    """
    Show one UI with all clusters that require resolution.

    pending_clusters   clusters from conflict_resolution.apply_recommendations()
                       (lists of statement ids are still accepted)
    resolved_knowledge records from the same call (with anomalies and recommendation)
    """
    if not isinstance(resolved_knowledge, list) or any(not isinstance(s, dict) for s in resolved_knowledge):
        raise TypeError("resolved_knowledge must be a list of statement dictionaries.")
    if not isinstance(pending_clusters, list):
        raise TypeError("pending_clusters must be a list.")

    by_id = {s["statement_id"]: s for s in resolved_knowledge if s.get("statement_id") is not None}
    empty = {"kept": [], "rejected": [], "pending": [], "decisions": [], "spn_integration": [],
             "expert_decisions": [], "cancelled": False}

    clusters, numbers = [], []
    for fallback, data in enumerate(pending_clusters, start=1):
        if isinstance(data, dict):
            complete, ids = dict(data), list(data.get("statement_ids", []))
        elif isinstance(data, list):
            ids = [x for x in data if isinstance(x, str)]
            complete = {"cluster_id": fallback, "camps": [], "status": "pending", "recommended_keep": [],
                        "recommended_remove": [], "reason": ""}
        else:
            raise TypeError("Each pending cluster must be a cluster dictionary or a list of statement ids.")
        statements = [by_id[i] for i in ids if i in by_id]
        if not statements:
            continue
        complete["statements"] = statements
        complete["statement_ids"] = [s["statement_id"] for s in statements]
        complete.setdefault("cluster_id", fallback)
        clusters.append(complete)
        numbers.append(complete["cluster_id"])

    conflict_ids = {sid for c in clusters for sid in c["statement_ids"]}

    def in_order(ids):
        return [s for s in resolved_knowledge if s.get("statement_id") in ids]

    if not clusters:
        return dict(empty, kept=list(resolved_knowledge))

    root = tk.Tk()
    root.title("Conflict Resolution")
    root.geometry("1x1+0+0")
    try:
        root.attributes("-alpha", 0.0)
    except tk.TclError:
        pass
    dialog = ConflictResolver(root, clusters, numbers, clusters)
    dialog.deiconify()
    dialog.update_idletasks()
    dialog.lift()
    dialog.focus_force()
    root.wait_window(dialog)
    decisions = dialog.result
    root.destroy()

    if decisions is None:
        return dict(empty, cancelled=True)

    kept_ids, rejected_ids, pending_ids = set(), set(), set()
    by_number = dict(zip(numbers, clusters))
    for d in decisions:
        if d["decision"] == "pending":
            pending_ids.update(by_number[d["cluster"]]["statement_ids"])
        else:
            kept_ids.update(d["kept_statement_ids"])
            rejected_ids.update(d["rejected_statement_ids"])
    kept_ids |= set(by_id) - conflict_ids
    kept_ids -= pending_ids
    rejected_ids -= pending_ids

    expert_decisions = ([_as_expert_decision(by_id[i], "accepted") for i in sorted(kept_ids & conflict_ids)]
                        + [_as_expert_decision(by_id[i], "rejected") for i in sorted(rejected_ids)])
    return {"kept": in_order(kept_ids), "rejected": in_order(rejected_ids), "pending": in_order(pending_ids),
            "decisions": decisions,
            "spn_integration": [{"cluster": d["cluster"], "camps": d["accepted_camps"], "mode": d["integration"]}
                                for d in decisions if d.get("integration")],
            "expert_decisions": expert_decisions, "cancelled": False}
