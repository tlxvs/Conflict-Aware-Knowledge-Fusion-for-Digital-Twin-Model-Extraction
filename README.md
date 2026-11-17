# Conflict Detection and Resolution Recommendations for Expert Knowledge in Digital Twin Model Extraction

This repository contains a **conflict handling module** that detects contradictions between expert knowledge statements and recommends resolutions as decision-support for human reviewers before the knowledge is used for Digital Twin (DT) model extraction. The conflict handling module is embedded in a model extraction pipeline that builds SPN models from IoT data and expert knowledge.Only resolved knowledge enters the final DT model.


## Key Features

- **Hybrid conflict detection** on formalized production rules (`IF condition THEN outcome (IN scope)`), utilizing:
  - deterministic lexical checks (identity, negation, explicit exclusiveness such as "only" or "must not exceed"),
  - numerical constraint solving with the Z3 SMT solver,
  - semantic similarity with Sentence-BERT,
  - natural language inference (NLI) with a fine-tuned DeBERTa-v3 model where deterministic checks are inconclusive.
- **Anomaly taxonomy**: Diverging Outcome, Diverging Condition, Internal Inconsistency.
- **Conflict rating**: *Conflict*, *Potential conflict*, and *Observation*, assigned per knowledge type to distinguish logical contradictions from legitimate branching and parallelism.
- **Traceable annotation** of each statement with its counterpart IDs, anomaly category, conflict rating, evidence, and NLI scores.
- **Conflict visualization in the SPN**: flagged components are drawn dashed with color-coded annotation boxes per conflict cluster and are excluded from simulation until resolved.
- **Resolution recommendations** as human-in-the-loop decision-support: conflicting rules are clustered, split into maximal compatible camps, and ranked by previous expert decisions, IoT log overlap, source corroboration, specificity, and number of findings. The reviewer accepts, rejects, or defers camps or rules.


## Getting Started

*Tested with Python 3.12.*

```bash
git clone https://github.com/tlxvs/Conflict-Aware-Knowledge-Fusion-for-Digital-Twin-Model-Extraction.git            # 1. clone the repository
pip install -r requirements.txt  # 2. install the requirements
python main.py                   # 3. run the pipeline incl. the validation layer
```

Running `main.py` executes the full pipeline on the case study data of the paper. After conflicts are detected and resolution recommendations are computed by the algorithm, a review interface opens in which the user can resolve the flagged conflict clusters before the final model is extracted.

The conflict handling module is located in the folder `statement_conflict_detection_and_resolution`. To use the conflict handling module functionality independently do:

```bash
cd statement_conflict_detection_and_resolution      # 1. navigate into the folder
call `detect_conflicts` from `conflict_detection.py` with a pandas DataFrame containing the columns `original`, `formalization`, `category`, `timestamp`, `source`, and `statement_id`.                # 2. run the detection
call `apply_recommendations` from `conflict_resolution.py` with the output of `detect_conflicts`, the IoT `event_log` and `state_log` DataFrames, and the same `detection_kwargs` as for detection.     # 3. run the clustering and recommendation              
```


The evaluation is located in the folder `evaluation` and can be executed with: 
```bash
cd evaluation      # 1. navigate into the folder
python evaluate_detection_and_recommendation.py --rules all_pdf_rules.jsonl all_conflicting_rules_100.jsonl --labels all_labels_100.csv --ground-truth ground_truth.csv --backend nli --timeseries timeseries_annotated.csv --out results_all_nli_iot   # 2. run the evaluation 
python ablation.py # 3. run the ablation             

```


## Input

The case study has two inputs located in `input_data`:

1. **Expert knowledge** rules extracted from the iMAKS dataset SOP and DS documents.
2. **IoT data** from the iMAKS dataset timeseries_annotated.

## Output

Results are saved in the `output` folder:

- conflict detection results per statement (anomaly category, severity, counterparts, evidence),
- conflict clusters, camps, recommendations, and stored expert decisions,
- the extracted SPN with visualized conflicts and the resolved data-knowledge-fused model (`.gv`, `.png`, `.pdf`),
- results of the expert knowledge processing step (classification and formalization), if run from natural language.
- extracted data-driven and data-knowledge-fused models

## Usage & Attribution

If you use this tool in a scientific project, please cite: [publication placeholder]

For questions and feedback, feel free to contact me.


