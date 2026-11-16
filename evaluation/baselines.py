"""Baselines for the conflict-detection comparison (same rules, same labels as evaluate_detection_and_recommendation.py).

"""
import argparse, itertools, json, os, re, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_detection_and_recommendation import load_rules, load_labels, natural   # same loaders as the evaluation

NEG = "Valid Discrepancy"


def texts_of(rules):
    return {r["statement_id"]: str(r.get("formalization") or r.get("original") or "") for r in rules}


# ----------------------------------------------------------------------------- scorers
def score_tfidf(pairs, T):
    from sklearn.feature_extraction.text import TfidfVectorizer
    ids = sorted(T, key=natural)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([T[i] for i in ids])
    X = vec.transform([T[i] for i in ids]); ix = {i: k for k, i in enumerate(ids)}
    return {p: float(X[ix[p[0]]].multiply(X[ix[p[1]]]).sum()) for p in pairs}


def score_embedding(pairs, T, model="sentence-transformers/all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model); ids = sorted(T, key=natural)
    E = m.encode([T[i] for i in ids], normalize_embeddings=True, show_progress_bar=False); ix = {i: k for k, i in enumerate(ids)}
    return {p: float(np.dot(E[ix[p[0]]], E[ix[p[1]]])) for p in pairs}


def score_nli(pairs, T, model="cross-encoder/nli-deberta-v3-base", batch=32):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model); labels = [l.lower() for l in ce.model.config.id2label.values()]
    c_idx = next(i for i, l in enumerate(labels) if l.startswith("contra"))
    pl = list(pairs); inp = [(T[a], T[b]) for a, b in pl] + [(T[b], T[a]) for a, b in pl]
    out = []
    for k in range(0, len(inp), batch):
        lg = np.asarray(ce.predict(inp[k:k + batch], apply_softmax=True, show_progress_bar=False)); out.extend(lg[:, c_idx])
    n = len(pl)
    return {p: float(max(out[i], out[n + i])) for i, p in enumerate(pl)}


LLM_PROMPT = """You check two production rules of a manufacturing plant for conflicts.

Rule A: {a}
Rule B: {b}

Definitions
- Diverging Outcome: the conditions can hold at the same time (equal, nested or overlapping) but the outcomes are incompatible (e.g. required vs prohibited, activate vs deactivate, disjoint time limits).
- Diverging Precondition: the same outcome is demanded for different or mutually exclusive conditions (e.g. a threshold stated with ONLY / MUST NOT exceed contradicts another threshold for the same action; disjoint required ranges for the same state; different predecessors for the same result).
- Severity: "conflict" (the rules cannot both be obeyed), "potential conflict" (they may coexist, e.g. alternatives or maintenance vs operation), "observation" (probably intended, e.g. alternative triggers).
- Valid: no conflict, e.g. different roles, alarm levels, zones or variables, a restated duplicate, nested conditions with compatible outcomes, or actions that can all be carried out.

Answer with one JSON object only: {{"conflict": true|false, "category": "Diverging Outcome"|"Diverging Precondition"|"none", "severity": "conflict"|"potential conflict"|"observation"|"none", "reason": "<one sentence>"}}"""


def parse_llm(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    return bool(d.get("conflict", False)), str(d.get("category", "none")), str(d.get("severity", "none")), str(d.get("reason", ""))


def score_llm(pairs, R, model, cache_path, sleep=0.0):
    import anthropic
    client = anthropic.Anthropic()
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            d = json.loads(l); cache[(d["a"], d["b"])] = d
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, (a, b) in enumerate(pairs, 1):
            if (a, b) in cache: continue
            ra, rb = R[a], R[b]
            msg = LLM_PROMPT.format(a=f'{ra.get("original") or ""} [{ra.get("formalization")}]', b=f'{rb.get("original") or ""} [{rb.get("formalization")}]')
            for attempt in range(4):
                try:
                    resp = client.messages.create(model=model, max_tokens=300, temperature=0, messages=[{"role": "user", "content": msg}]); break
                except Exception as e:                                   # rate limit / transient error
                    time.sleep(2 ** attempt); resp = None
            if resp is None: raise SystemExit(f"LLM call failed for {a},{b}")
            c, cat, sev, why = parse_llm(resp.content[0].text)
            d = dict(a=a, b=b, conflict=c, category=cat, severity=sev, reason=why); cache[(a, b)] = d
            f.write(json.dumps(d) + "\n"); f.flush(); time.sleep(sleep)
            if n % 25 == 0: print(f"  llm {n}/{len(pairs)}")
    return {p: float(cache[p]["conflict"]) for p in pairs}, cache


def read_system(dir_, labelled_pairs):
    """Predictions of your detector on the labelled pairs, from classification_results.csv written by the evaluator."""
    d = pd.read_csv(os.path.join(dir_, "classification_results.csv"), dtype=str).fillna("")
    flagged = {frozenset((r.id_a, r.id_b)) for _, r in d.iterrows() if r.detected_severity}
    return {p: float(frozenset(p) in flagged) for p in labelled_pairs}, d


# ----------------------------------------------------------------------------- metrics
def prf(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    return tp, fp, fn, p, r, (2 * p * r / (p + r) if p + r else float("nan"))


def summarise(name, y, s, native_threshold=None, binary=False):
    y = np.asarray(y); s = np.asarray(s, float)
    row = dict(method=name, n_pairs=len(y), positives=int(y.sum()))
    if binary or len(set(s)) <= 2:
        tp, fp, fn, p, r, f = prf(y, (s >= 0.5).astype(int))
        row.update(operating_point="native decision", threshold=0.5, precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)
        row.update(AP=float("nan"), AUC=float("nan"))
        return [row]
    row.update(AP=float(average_precision_score(y, s)), AUC=float(roc_auc_score(y, s)))
    rows = []
    if native_threshold is not None:
        tp, fp, fn, p, r, f = prf(y, (s >= native_threshold).astype(int))
        rows.append({**row, "operating_point": f"fixed threshold {native_threshold}", "threshold": native_threshold, "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn})
    P, Rc, Th = precision_recall_curve(y, s)
    F = 2 * P * Rc / np.where(P + Rc == 0, 1, P + Rc); k = int(np.nanargmax(F[:-1]))
    tp, fp, fn, p, r, f = prf(y, (s >= Th[k]).astype(int))
    rows.append({**row, "operating_point": "best-F1 threshold (oracle, favours the baseline)", "threshold": float(Th[k]), "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--methods", nargs="+", default=["tfidf"], choices=["tfidf", "embedding", "nli", "llm"])
    ap.add_argument("--system-results", nargs="*", default=[], help="result folders of the evaluator (name=folder or folder)")
    ap.add_argument("--llm-model", default="claude-sonnet-5-5"); ap.add_argument("--llm-sleep", type=float, default=0.0)
    ap.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base"); ap.add_argument("--nli-threshold", type=float, default=0.80)
    ap.add_argument("--all-pairs", action="store_true", help="also score every pair of the rule base (tfidf / embedding / nli) for recall and flag counts")
    ap.add_argument("--out", default="baselines_out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    rules = load_rules(a.rules); R = {r["statement_id"]: r for r in rules}; T = texts_of(rules)
    lab = load_labels(a.labels)
    pairs, y = [], []
    for _, r in lab.iterrows():
        p = tuple(sorted((r.id_a, r.id_b), key=natural))
        if p in pairs: continue
        pairs.append(p); y.append(0 if r.lab_cat == NEG else 1)
    y = np.array(y); print(f"{len(pairs)} labelled pairs: {int(y.sum())} conflicting, {int((1 - y).sum())} valid")

    rows, scores = [], pd.DataFrame({"id_a": [p[0] for p in pairs], "id_b": [p[1] for p in pairs], "label_conflict": y})
    cat_rows = []
    for m in a.methods:
        print("running", m)
        if m == "tfidf": sc = score_tfidf(pairs, T); nat = None
        elif m == "embedding": sc = score_embedding(pairs, T, a.embedding_model); nat = None
        elif m == "nli": sc = score_nli(pairs, T, a.nli_model); nat = a.nli_threshold
        else:
            sc, cache = score_llm(pairs, R, a.llm_model, os.path.join(a.out, "llm_cache.jsonl"), a.llm_sleep); nat = 0.5
            tp_pairs = [p for p, yy in zip(pairs, y) if yy and cache[p]["conflict"]]
            lab_by_pair = {tuple(sorted((r.id_a, r.id_b), key=natural)): r for _, r in lab.iterrows()}
            if tp_pairs:
                cat_rows.append(dict(method="llm", category_accuracy_given_detected=float(np.mean([cache[p]["category"] == lab_by_pair[p].lab_cat for p in tp_pairs])),
                                     severity_accuracy_given_detected=float(np.mean([cache[p]["severity"] == lab_by_pair[p].lab_sev for p in tp_pairs]))))
        scores[m] = [sc[p] for p in pairs]
        rows += summarise(m if m != "nli" else "plain NLI", y, scores[m], nat, binary=(m == "llm"))
    for item in a.system_results:
        name, _, folder = item.partition("=") if "=" in item else (os.path.basename(item.rstrip("/\\")), "", item)
        sc, d = read_system(folder, pairs); scores["system:" + name] = [sc[p] for p in pairs]
        rows += summarise("system: " + name, y, scores["system:" + name], binary=True)
        det = d[d.outcome == "TP"]
        if len(det):
            cat_rows.append(dict(method="system: " + name, category_accuracy_given_detected=float((det.category_ok == "True").mean()),
                                 severity_accuracy_given_detected=float((det.severity_ok == "True").mean())))
    res = pd.DataFrame(rows); scores.to_csv(os.path.join(a.out, "pair_scores.csv"), index=False)
    cols = ["method", "operating_point", "threshold", "precision", "recall", "f1", "AP", "AUC", "tp", "fp", "fn", "n_pairs", "positives"]
    res = res[cols]; res.to_csv(os.path.join(a.out, "baseline_comparison.csv"), index=False)
    if cat_rows: pd.DataFrame(cat_rows).to_csv(os.path.join(a.out, "category_severity_accuracy.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nCOMPARISON ON THE LABELLED PAIRS\n", res.round(3).to_string(index=False))
    if cat_rows: print("\n", pd.DataFrame(cat_rows).round(3).to_string(index=False))

    if a.all_pairs:                                                          # every pair of the rule base
        ids = sorted(T, key=natural); allp = list(itertools.combinations(ids, 2)); pos = {p for p, yy in zip(pairs, y) if yy}
        out = []
        for m in a.methods:
            if m == "llm": continue
            print("all pairs:", m, len(allp))
            sc = score_tfidf(allp, T) if m == "tfidf" else score_embedding(allp, T, a.embedding_model) if m == "embedding" else score_nli(allp, T, a.nli_model)
            th = float(res[(res.method.str.contains("NLI" if m == "nli" else m)) & res.operating_point.str.startswith("best")].threshold.iloc[0]) if m != "nli" else a.nli_threshold
            flagged = {p for p in allp if sc[p] >= th}
            out.append(dict(method=m, threshold=th, flagged_pairs=len(flagged), true_positives=len(flagged & pos), recall_on_labelled_conflicts=len(flagged & pos) / len(pos),
                            precision_labelled_only=len(flagged & pos) / max(1, len(flagged & pos) + len([p for p in flagged if p in set(pairs) and p not in pos])),
                            unlabelled_flags=len([p for p in flagged if p not in set(pairs)])))
        pd.DataFrame(out).to_csv(os.path.join(a.out, "baseline_all_pairs.csv"), index=False); print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
