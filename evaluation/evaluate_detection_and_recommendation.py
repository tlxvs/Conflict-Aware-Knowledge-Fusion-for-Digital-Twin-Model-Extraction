#!/usr/bin/env python3
"""Assess conflict detection and recommendation on the SOP-001 rule files.

  python evaluate_detection_and_recommendation.py \
      --rules sop001_pdf_rules.jsonl sop001_conflicting_rules_50.jsonl \
      --labels sop001_conflicting_rules_50_labels.csv \
      --ground-truth ground_truth.csv  [--backend heuristic|nli] [--out results]

Inputs
  --rules         JSONL rule files (original, formalization, category, timestamp, source, statement_id); the formalizations are
                  already in the detector's format, so nothing is rewritten.
  --labels        expected pair classification (id_a, id_b, proposed_category, proposed_severity, optional expert_label).
                  'Valid Discrepancy' rows are pairs that look conflicting but are consistent.
  --ground-truth  ground_truth.csv (the reference rules). It is used (1) to check that the rules extracted from the PDF agree with it
                  and (2) to derive which statements a recommendation must keep or remove.

Outputs (in --out): alignment_pdf_vs_ground_truth.csv, classification_results.csv, recommendation_results.csv, summary.json, evaluation.xlsx
"""
import argparse, json, os, re, sys
from typing import Any, Dict, List
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # makes statement_conflict_detection_and_resolution importable
sys.path.insert(0, str(Path(__file__).parent))  # keeps pdf_rules / iot_adapter importable

from statement_conflict_detection_and_resolution.conflict_detection import detect_conflicts
from statement_conflict_detection_and_resolution.conflict_resolution import apply_recommendations

def _import_iot_adapter():
    """iot_adapter lives either in the same package as the resolution file or next to this script."""
    import importlib
    pkg = apply_recommendations.__module__.rpartition(".")[0]
    for name in ([f"{pkg}.iot_adapter"] if pkg else []) + ["iot_adapter"]:
        try:
            return importlib.import_module(name).load_state_log
        except ImportError:
            continue
    sys.exit("iot_adapter.py not found: put it next to this script or into the package folder of conflict_resolution.py")


FINDING = ("Diverging Outcome", "Diverging Precondition")
SEV = {"conflict": 2, "potential conflict": 1, "observation": 0}
NUM = r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?"
ID_RE = re.compile(r"\b(?:[A-Z]{2,5}\d{1,3}(?:_[A-Z0-9_&]+)?|[A-Z]{2,4}-\d{3,5}[A-Za-z]*|RULE-[A-Z0-9-]+|MAINT-\d+|SOP-\d+)\b")


def natural(x: str):
    """sort key that works for numeric ids ('12', '2001') and for ids such as 'W_16'"""
    return (0, int(x)) if str(x).isdigit() else (1, str(x))


def numbers(t: str) -> set:
    return {float(x) for x in re.findall(NUM, ID_RE.sub(" ", str(t or "")))}


def load_rules(paths: List[str]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r["statement_id"] = str(r["statement_id"])
                    if r["statement_id"] in seen:
                        sys.exit(f"duplicate statement_id {r['statement_id']} (files must use distinct ids)")
                    seen.add(r["statement_id"]); r["file"] = os.path.basename(p); out.append(r)
    return out


# --------------------------------------------------------------------------- 1. PDF rules vs ground truth
def gt_rows_for(source: str, g: pd.DataFrame) -> pd.DataFrame:
    m = re.search(r"(?:RULE-|MAINT-)[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*", str(source))
    if not m:
        return g.iloc[0:0]
    return g[g.ruleId.str.startswith(m.group(0))]            # RULE-WRH01-01 -> RULE-WRH01-01a / 01b; RULE-THR-ST01-TMP-CRIT; MAINT-01


def alignment(rules, g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in rules:
        if ".pdf" not in str(r.get("source", "")).lower():            # only statements extracted from the PDF (not the test rules)
            continue
        c = gt_rows_for(f'{r.get("source", "")} {r.get("reference", "")}', g)
        if c.empty:
            continue
        gt_nums = set()
        for _, x in c.iterrows():
            gt_nums |= numbers(f"{x.condition} {x.action}") | {float(x[k]) for k in ("critHi", "warnHi", "critLo", "warnLo") if pd.notna(x[k])}
        mine = numbers(r["formalization"])
        extra = sorted(mine - gt_nums)
        rows.append(dict(statement_id=r["statement_id"], source=r["source"], gt_rules="; ".join(c.ruleId), formalization=r["formalization"],
                         numbers_in_statement=sorted(mine), numbers_not_in_ground_truth=extra, aligned=not extra))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. classification

# Older label files use "S_2001" for the test rules and "W_n" (numbering of Rules_written.txt) for the PDF rules.
# They are translated to the statement_ids of sop001_pdf_rules.jsonl so both versions of the labels file work.
W2P = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 18, 17: 19, 18: 21, 19: 23,
       20: 25, 21: 26, 22: 27, 23: 29, 24: 30, 29: 24, 30: 28}
W_OVERRIDE = {("OO-10", "W_22"): 28, ("DP-15", "W_9"): 10, ("DP-17", "W_9"): 9, ("VR-06", "W_9"): 9}


def normalise_ids(d: pd.DataFrame) -> pd.DataFrame:
    def conv(x, tag):
        x = str(x).strip()
        if x.startswith("W_") and x[2:].isdigit():
            return str(W_OVERRIDE.get((tag, x), W2P.get(int(x[2:]), x)))
        return re.sub(r"^S2?_", "", x)
    tags = d["tag"] if "tag" in d else pd.Series([""] * len(d))
    d["id_a"] = [conv(a, t) for a, t in zip(d.id_a, tags)]
    d["id_b"] = [conv(b, t) for b, t in zip(d.id_b, tags)]
    return d

def load_labels(path: str) -> pd.DataFrame:
    d = normalise_ids(pd.read_csv(path, dtype=str).fillna(""))
    cat, sev = [], []
    for _, r in d.iterrows():
        c, s = r.proposed_category, r.proposed_severity
        if r.get("expert_label", "").strip():
            c, _, s = r.expert_label.partition("/"); c, s = c.strip(), s.strip()
        cat.append(c); sev.append(s)
    d["lab_cat"], d["lab_sev"] = cat, sev
    return d


def pair_findings(ann) -> Dict[frozenset, Dict[str, str]]:
    out = {}
    for r in ann:
        for f in r.get("anomalies", []):
            p = f.get("statement_id")
            if p is None:
                continue
            k = frozenset((str(r["statement_id"]), str(p)))
            if k not in out or SEV[f["severity"]] > SEV[out[k]["severity"]]:
                out[k] = dict(category=f["conflict_type"], severity=f["severity"], reason=f.get("reason", ""),
                              cond_rel=f.get("condition_relation", ""), out_rel=f.get("outcome_relation", ""), method=f.get("method", ""))
    return out


def evaluate_classification(ann, labels: pd.DataFrame):
    found = pair_findings(ann)
    from_pdf = {str(r["statement_id"]) for r in ann if ".pdf" in str(r.get("source", "")).lower()}
    exp = {frozenset((r.id_a, r.id_b)): r for _, r in labels.iterrows()}
    rows = []
    for k in set(exp) | set(found):
        e, f = exp.get(k), found.get(k)
        is_finding = e is not None and e.lab_cat in FINDING
        is_valid = e is not None and e.lab_cat == "Valid Discrepancy"
        if is_finding and f: o = "TP"
        elif is_finding: o = "FN"
        elif f and is_valid: o = "FP (labelled valid)"
        elif f and all(str(x) in from_pdf for x in k): o = "FP (pair of PDF rules)"      # both rules are extracted from the SOPs
        elif f: o = "FP (unlabelled pair)"
        else: continue
        a, b = sorted(k, key=natural)
        rows.append(dict(id_a=a, id_b=b, tag=e.tag if e is not None else "", expected_category=e.lab_cat if is_finding else ("Valid Discrepancy" if is_valid else ""),
                         expected_severity=e.lab_sev if is_finding else "", detected_category=f["category"] if f else "", detected_severity=f["severity"] if f else "",
                         outcome=o, category_ok=bool(is_finding and f and f["category"] == e.lab_cat), severity_ok=bool(is_finding and f and f["severity"] == e.lab_sev),
                         rationale=e.rationale if e is not None else "", detected_reason=f["reason"] if f else "",
                         condition_relation=f["cond_rel"] if f else "", outcome_relation=f["out_rel"] if f else "", method=f["method"] if f else ""))
    df = pd.DataFrame(rows)
    s: Dict[str, Any] = {}
    n_exp = sum(1 for r in exp.values() if r.lab_cat in FINDING)
    tp = int((df.outcome == "TP").sum()) if len(df) else 0
    fn = n_exp - tp
    fp_valid = int((df.outcome == "FP (labelled valid)").sum()) if len(df) else 0
    fp_pdf = int((df.outcome == "FP (pair of PDF rules)").sum()) if len(df) else 0
    fp_unl = int((df.outcome == "FP (unlabelled pair)").sum()) if len(df) else 0 + 0
    fp_unl_all = fp_unl + fp_pdf
    n_valid = sum(1 for r in exp.values() if r.lab_cat == "Valid Discrepancy")
    p = tp / (tp + fp_valid + fp_unl_all) if tp + fp_valid + fp_unl_all else float("nan")
    r_ = tp / n_exp if n_exp else float("nan")
    det = df[df.outcome == "TP"] if len(df) else df
    s = dict(pairs_expected=n_exp, tp=tp, fn=fn, fp_labelled_valid=fp_valid, valid_pairs_labelled=n_valid, fp_unlabelled_pairs=fp_unl, fp_pairs_of_pdf_rules=fp_pdf,
             precision=p, recall=r_, f1=2 * p * r_ / (p + r_) if p + r_ else float("nan"),
             category_accuracy_given_detected=float(det.category_ok.mean()) if len(det) else None,
             severity_accuracy_given_detected=float(det.severity_ok.mean()) if len(det) else None,
             per_label={f"{c} / {v}": dict(n=int(len(x)), recall=float((x.outcome == 'TP').mean()))
                        for (c, v), x in (df[df.outcome.isin(['TP', 'FN'])].groupby(['expected_category', 'expected_severity']) if len(df) else [])})
    if n_exp:                                                           # FN rows are not in df when never detected -> add them
        have = set(map(frozenset, zip(df.id_a, df.id_b))) if len(df) else set()
        miss = []
        for k, e in exp.items():
            if e.lab_cat in FINDING and k not in have:
                a, b = sorted(k, key=natural)
                miss.append(dict(id_a=a, id_b=b, tag=e.tag, expected_category=e.lab_cat, expected_severity=e.lab_sev, outcome="FN", rationale=e.rationale))
        df = pd.concat([df, pd.DataFrame(miss)], ignore_index=True)
        s["per_label"] = {f"{c} / {v}": dict(n=int(len(x)), recall=float((x.outcome == "TP").mean()))
                          for (c, v), x in df[df.outcome.isin(["TP", "FN"])].groupby(["expected_category", "expected_severity"])}
    return df.sort_values(["outcome", "id_a"]).reset_index(drop=True), s


# --------------------------------------------------------------------------- 3. recommendation
def expected_status(rules, labels: pd.DataFrame, align: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """confirmed   : statement extracted from the PDF and numerically in line with ground_truth.csv
       contradicted: statement that conflicts (conflict / potential conflict) with a confirmed statement
       compatible  : statement whose only relation to a confirmed statement is valid / observation
       not covered : everything else"""
    st = {r["statement_id"]: dict(status="not covered", gt="") for r in rules}
    ok = {row.statement_id: row.gt_rules for row in align.itertuples() if row.aligned}
    for sid, gt in ok.items():
        st[sid] = dict(status="confirmed", gt=gt)
    for _, r in labels.iterrows():
        for x, y in ((r.id_a, r.id_b), (r.id_b, r.id_a)):
            if y in ok and x in st and x not in ok:
                if r.lab_cat in FINDING and r.lab_sev in ("conflict", "potential conflict"):
                    st[x] = dict(status="contradicted", gt=ok[y])
                elif st[x]["status"] == "not covered":
                    st[x] = dict(status="compatible", gt=ok[y])
    return st


def evaluate_recommendations(clusters, status):
    rows = []
    for c in clusters:
        m = [str(x) for x in c["statement_ids"]]
        conf = [x for x in m if status[x]["status"] == "confirmed"]
        contra = [x for x in m if status[x]["status"] == "contradicted"]
        keep = {str(x) for x in (c.get("recommended_keep") or [])}
        if not conf and not contra:
            v = "no ground truth"
        elif c.get("status") == "pending" or not keep:
            v = "abstain"
        elif keep & set(contra) or set(conf) - keep:
            v = "wrong"
        else:
            v = "correct"
        gt_kept = {status[x]["gt"] for x in conf if x in keep}                     # ground-truth rules still represented by a kept rule
        lost = [x for x in conf if x not in keep and status[x]["gt"] not in gt_kept]
        v_len = v if v in ("no ground truth", "abstain") else ("wrong" if keep & set(contra) or lost else "correct")
        rows.append(dict(cluster=c["cluster_id"], members="; ".join(m), status=c.get("status"), decided_by=", ".join(c.get("decided_by") or []),
                         recommended_keep="; ".join(sorted(keep)), must_keep_confirmed="; ".join(conf), must_remove_contradicted="; ".join(contra),
                         kept_a_contradicted_rule="; ".join(sorted(keep & set(contra))), removed_a_confirmed_rule="; ".join(sorted(set(conf) - keep)) if keep else "",
                         gt_rules="; ".join(sorted({status[x]["gt"] for x in m if status[x]["gt"]})), verdict=v, verdict_lenient=v_len))
    df = pd.DataFrame(rows)
    ev = df[df.verdict != "no ground truth"]; dec = ev[ev.verdict != "abstain"]
    dec_l = ev[ev.verdict_lenient != "abstain"]
    s = dict(clusters=len(df), with_ground_truth=len(ev), decided=len(dec), correct=int((ev.verdict == "correct").sum()),
             wrong=int((ev.verdict == "wrong").sum()), abstain=int((ev.verdict == "abstain").sum()),
             coverage=len(dec) / len(ev) if len(ev) else None, precision_when_decided=float((dec.verdict == "correct").mean()) if len(dec) else None,
             accuracy=float((ev.verdict == "correct").mean()) if len(ev) else None,
             correct_lenient=int((ev.verdict_lenient == "correct").sum()),
             precision_when_decided_lenient=float((dec_l.verdict_lenient == "correct").mean()) if len(dec_l) else None,
             accuracy_lenient=float((ev.verdict_lenient == "correct").mean()) if len(ev) else None)
    return df, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--ground-truth", required=True); ap.add_argument("--out", default="evaluation_results")
    ap.add_argument("--backend", default="heuristic", choices=["heuristic", "nli"])
    ap.add_argument("--no-subject-guard", action="store_true", help="switch the role / alarm-level compatibility check off (ablation)")
    ap.add_argument("--no-candidates", action="store_true", help="compare ALL pairs instead of the Phase-1 candidate pairs (ablation; slow with NLI)")
    ap.add_argument("--no-lexical", action="store_true", help="switch the lexical check off (negation / prohibition / keyword comparison and exclusiveness markers; ablation)")
    ap.add_argument("--no-smt", action="store_true", help="switch the SMT (z3) solver off and use interval arithmetic only")
    ap.add_argument("--timeseries", default=None, help="sensor CSV for the IoT criterion (timeseries_annotated.csv); if omitted the script looks in ., input_data, ../input_data")
    ap.add_argument("--expert-decisions", default=None, help="jsonl of previous expert decisions (rule fields + decision: accepted|rejected) for the expert-decision criterion")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rules, g, labels = load_rules(a.rules), pd.read_csv(a.ground_truth), load_labels(a.labels)
    print(f"{len(rules)} rules, {len(labels)} labelled pairs, {len(g)} ground-truth rules")
    known = {r["statement_id"] for r in rules}
    unknown = sorted({i for i in list(labels.id_a) + list(labels.id_b) if i not in known}, key=natural)
    if unknown:
        sys.exit(f"The labels file refers to ids that are not in the rule files: {unknown[:8]}{' ...' if len(unknown) > 8 else ''}\n"
                 "Use the current sop001_conflicting_rules_50_labels.csv (ids 1-41 and 2001-2050), not an older copy with ids such as W_16.")
    align = alignment(rules, g)

    keep = ["statement_id", "category", "formalization", "original", "source", "timestamp"]
    df = pd.DataFrame([{k: r.get(k) for k in keep} for r in rules])
    mod = sys.modules[detect_conflicts.__module__]
    smt_on = (not a.no_smt) and getattr(mod, "_z3", None) is not None
    print(f"backend: {a.backend}  |  SMT: {'z3 active' if smt_on else 'off' + (' (z3 is not installed: pip install z3-solver)' if not a.no_smt and getattr(mod, '_z3', None) is None else ' (--no-smt)')}")
    opts = dict(backend=a.backend, smt=not a.no_smt, subject_guard=not a.no_subject_guard,
                candidate_selection=not a.no_candidates, lexical_check=not a.no_lexical)

    # ---- criterion 1: previous expert decisions (optional input)
    decisions = None
    if a.expert_decisions:
        decisions = [json.loads(l) for l in open(a.expert_decisions, encoding="utf-8") if l.strip()]
        print(f"expert decisions: {len(decisions)} loaded from {a.expert_decisions}")
    ann_df = detect_conflicts(df, **opts, **({"expert_decisions": decisions} if decisions else {}))    # your detection
    ann = ann_df.to_dict("records")

    # ---- criterion 2: IoT log (explicit path, or found automatically)
    ts = a.timeseries
    if not ts:
        here = os.path.dirname(os.path.abspath(__file__))
        for d in (os.getcwd(), here, os.path.join(os.getcwd(), "input_data"), os.path.join(here, "input_data"),
                  os.path.join(os.path.dirname(here), "input_data")):
            if os.path.exists(os.path.join(d, "timeseries_annotated.csv")):
                ts = os.path.join(d, "timeseries_annotated.csv"); break
    state_log, iot_info = None, "NOT ACTIVE (no sensor file found: pass --timeseries <file>)"
    if ts:
        if not os.path.exists(ts):
            sys.exit(f"--timeseries file not found: {ts}\nFind it with:  Get-ChildItem -Path $HOME -Recurse -Filter timeseries_annotated.csv | Select FullName")
        load_state_log = _import_iot_adapter()
        state_log = load_state_log(ts)
        for c in [c for c in state_log.columns if str(c).endswith("_sensor")]:       # rules say TMP / HUM, the adapter names the columns tmp_sensor / hum_sensor
            if c[:-7] not in state_log.columns:
                state_log[c[:-7]] = state_log[c]
        IoTEvidence = sys.modules[apply_recommendations.__module__].IoTEvidence          # same module the recommendation uses
        extract_rule = sys.modules[detect_conflicts.__module__].extract_rule
        ev = IoTEvidence(state_log=state_log)
        sc = [ev.score(extract_rule(r, i)) for i, r in enumerate(ann)]
        n_sc, n_ok = sum(x is not None for x in sc), sum(x is not None and x >= 0.20 for x in sc)
        iot_info = f"active: {len(state_log)} log rows from {ts}; {n_sc} of {len(ann)} rules scored, {n_ok} reach the 0.20 support threshold"
    print("criterion 1 expert decisions:", f"{len(decisions)} previous decisions" if decisions else "NO previous decisions given (every camp trivially satisfies it)")
    print("criterion 2 IoT support:     ", iot_info)
    print("criteria 3-5 (source corroboration, prudent/specific, fewer classifications): active")
    recs, clusters = apply_recommendations(ann, detection_kwargs=opts, state_log=state_log)   # your recommendation

    cls_df, cls_s = evaluate_classification(ann, labels)
    status = expected_status(rules, labels, align)
    rec_df, rec_s = evaluate_recommendations(clusters, status)
    used = {}
    for x in (rec_df.decided_by.dropna() if "decided_by" in rec_df else []):
        for c in str(x).split(","):
            if c.strip(): used[c.strip()] = used.get(c.strip(), 0) + 1
    rec_s["criteria_in_decisions"] = used
    rec_s["iot_criterion"] = iot_info; rec_s["expert_decisions_given"] = len(decisions or [])
    summary = dict(rules=len(rules), backend=a.backend, smt=("z3" if smt_on else "interval arithmetic"), labels="expert-corrected" if (labels.get("expert_label", pd.Series(dtype=str)) != "").any() else "manual judgement (not expert-reviewed)",
                   pdf_rules_vs_ground_truth=dict(statements_checked=len(align), aligned=int(align.aligned.sum()) if len(align) else 0),
                   classification=cls_s, recommendation=rec_s)
    align.to_csv(os.path.join(a.out, "alignment_pdf_vs_ground_truth.csv"), index=False)
    cls_df.to_csv(os.path.join(a.out, "classification_results.csv"), index=False)
    rec_df.to_csv(os.path.join(a.out, "recommendation_results.csv"), index=False)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    try:
        with pd.ExcelWriter(os.path.join(a.out, "evaluation.xlsx")) as xw:
            pd.DataFrame([{"metric": k, "value": v} for k, v in {**{f"classification.{k}": v for k, v in cls_s.items() if k != 'per_label'},
                          **{f"recommendation.{k}": v for k, v in rec_s.items()}}.items()]).to_excel(xw, "Summary", index=False)
            align.astype(str).to_excel(xw, "PDF_vs_ground_truth", index=False); cls_df.to_excel(xw, "Classification", index=False)
            rec_df.to_excel(xw, "Recommendation", index=False)
    except Exception as exc:
        print("xlsx not written:", exc)

    print("\nPDF rules vs ground_truth.csv:", summary["pdf_rules_vs_ground_truth"])
    if len(align) and not align.aligned.all():
        print(align[~align.aligned][["statement_id", "gt_rules", "numbers_not_in_ground_truth"]].to_string(index=False))
    print("\nCLASSIFICATION  (labels:", summary["labels"] + ")")
    for k, v in cls_s.items():
        print(f"  {k}: {v}")
    print("\nRECOMMENDATION against ground_truth.csv")
    for k, v in rec_s.items():
        print(f"  {k}: {v}")
    print("\nFiles written to", a.out)


if __name__ == "__main__":
    main()
