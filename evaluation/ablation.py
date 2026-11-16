"""Ablation of the conflict-detection approach.  Start it WITHOUT parameters:   python ablation.py

It looks for these files in the folder of this script (or the current folder):
    all_pdf_rules.jsonl, all_conflicting_rules_100.jsonl, all_labels_100.csv, ground_truth.csv
and compares your approach
    - with / without SMT         (z3; without = interval arithmetic)
    - with / without NLI         (without = heuristic backend, lexical only)
    - with / without the lexical check  (negation / prohibition / keyword comparison and exclusiveness markers)
    - with / without a TF-IDF baseline  (TF-IDF cosine similarity alone, best-F1 threshold = favours the baseline)
NLI variants run only if sentence-transformers and the NLI model are available; otherwise they are skipped with a note.
Result: ablation_results/ablation_table.csv (+ .md) and one sub-folder per variant.
"""
import glob, json, os, subprocess, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "ablation_results")


def find(name, pattern=None):
    for d in (HERE, os.getcwd()):
        p = os.path.join(d, name)
        if os.path.exists(p): return p
        if pattern:
            g = glob.glob(os.path.join(d, pattern))
            if g: return g[0]
    sys.exit(f"File not found: {name} (put it next to ablation.py)")


RULES = [find("all_pdf_rules.jsonl"), find("all_conflicting_rules_100.jsonl")]
LABELS = find("all_labels_100.csv")
GT = find("ground_truth.csv", "*ground_truth*.csv")


def nli_available():
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer       # noqa: F401
        CrossEncoder("cross-encoder/nli-deberta-v3-base"); return True
    except Exception as e:
        print("NLI variants skipped:", str(e).splitlines()[0][:120]); return False


def z3_available():
    try:
        import z3  # noqa: F401
        return True
    except Exception:
        print("Note: z3 is not installed (pip install z3-solver); 'without SMT' will equal 'with SMT'."); return False


def run_variant(name, backend, flags):
    folder = os.path.join(OUT, "".join(c if c.isalnum() else "_" for c in name).strip("_"))
    cmd = [sys.executable, os.path.join(HERE, "evaluate_detection_and_recommendation.py"), "--rules", *RULES, "--labels", LABELS,
           "--ground-truth", GT, "--backend", backend, "--out", folder, *flags]
    print("running:", name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("  FAILED:", (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["?"])[-1]); return None
    s = json.load(open(os.path.join(folder, "summary.json"))); c, rec = s["classification"], s["recommendation"]
    return {"variant": name, "backend": backend, "TP": c["tp"], "FN": c["fn"],
            "FP": c["fp_labelled_valid"] + c["fp_unlabelled_pairs"] + c.get("fp_pairs_of_pdf_rules", 0),
            "precision": c["precision"], "recall": c["recall"], "F1": c["f1"],
            "category_acc": c["category_accuracy_given_detected"], "severity_acc": c["severity_accuracy_given_detected"],
            "rec_decided": rec["decided"], "rec_correct": rec["correct"], "rec_precision_when_decided": rec["precision_when_decided"]}


def _tfidf_vectors(texts):
    """Plain-Python TF-IDF (unigrams + bigrams, sublinear tf, smooth idf, L2 norm); no scikit-learn needed."""
    import math, re
    from collections import Counter
    docs = []
    for t in texts:
        w = re.findall(r"[a-z0-9_]+", t.lower()); docs.append(Counter(w + [f"{x} {y}" for x, y in zip(w, w[1:])]))
    df = Counter(k for d in docs for k in d); n = len(docs); vecs = []
    for d in docs:
        v = {k: (1 + math.log(c)) * (math.log((1 + n) / (1 + df[k])) + 1) for k, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0; vecs.append({k: x / norm for k, x in v.items()})
    return vecs


def tfidf_baseline():
    """TF-IDF cosine on the labelled pairs; threshold = best F1 on the labels (favours the baseline)."""
    from evaluate_detection_and_recommendation import load_rules, load_labels, natural
    rules = load_rules(RULES); T = {r["statement_id"]: str(r.get("formalization") or r.get("original") or "") for r in rules}
    ids = sorted(T, key=natural); V = dict(zip(ids, _tfidf_vectors([T[i] for i in ids])))
    lab = load_labels(LABELS); pairs, y = [], []
    for _, r in lab.iterrows():
        p = tuple(sorted((r.id_a, r.id_b), key=natural))
        if p not in pairs: pairs.append(p); y.append(0 if r.lab_cat == "Valid Discrepancy" else 1)
    y = np.array(y); s = np.array([sum(x * V[b].get(k, 0.0) for k, x in V[a].items()) for a, b in pairs])
    best = None
    for th in sorted(set(s)):
        pred = s >= th; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        p_, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn); f = 2 * p_ * r_ / max(1e-9, p_ + r_)
        if best is None or f > best[0]: best = (f, tp, fp, fn, p_, r_)
    f, tp, fp, fn, p_, r_ = best
    return {"variant": "TF-IDF baseline only (no rules, no SMT, no NLI)", "backend": "tfidf", "TP": tp, "FN": fn, "FP": fp, "precision": p_, "recall": r_, "F1": f}


def main():
    os.makedirs(OUT, exist_ok=True); nli = nli_available(); z3_available()
    variants = []
    if nli:
        variants += [("full approach (NLI + SMT + lexical)", "nli", []), ("without SMT", "nli", ["--no-smt"]),
                     ("without lexical check (NLI + SMT)", "nli", ["--no-lexical"]),
                     ("without subject guard", "nli", ["--no-subject-guard"])]
    variants += [("without NLI (heuristic: lexical + SMT)", "heuristic", []), ("without NLI and without SMT", "heuristic", ["--no-smt"]),
                 ("without NLI and without lexical check (SMT only)", "heuristic", ["--no-lexical"]),
                 ("without NLI and without subject guard", "heuristic", ["--no-subject-guard"])]
    rows = [r for r in (run_variant(*v) for v in variants) if r]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation_table.csv"), index=False)       # saved first, in case the baseline fails
    try:
        rows.append(tfidf_baseline())
    except Exception as e:
        print("TF-IDF baseline failed (table without it is already saved):", repr(e))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "ablation_table.csv"), index=False)
    try: open(os.path.join(OUT, "ablation_table.md"), "w", encoding="utf-8").write(t.round(3).to_markdown(index=False))
    except Exception: pass
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 55)
    print("\nABLATION (classification on the labelled pairs; labels = manual judgement, not expert-reviewed)\n")
    print(t.round(3).to_string(index=False)); print("\nSaved to", os.path.join(OUT, "ablation_table.csv"))


if __name__ == "__main__":
    main()
