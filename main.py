"""
main.py

Main entry point for the Data-Knolwedge-Fused Digital Twin Stochastic Petri Net model extraction algorithm.

"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import json

import os

import copy

from pathlib import Path

from model_extraction.preprocess import (
    load_event_log,
    load_state_log,
    preprocess_log,
)

from model_extraction.process_discovery import (
    discover_process_model,
)

from model_extraction.structure import (
    build_spn_structure,
)

from model_extraction.inhibitor_detection import (
    detect_inhibitor_arcs,
)

from model_extraction.multiplicity import (
    detect_arc_multiplicities,
)

from model_extraction.controlflow import (
    implement_control_flow_patterns,
)

from model_extraction.distributions import (
    fit_transition_distributions,
)

from model_extraction.weights import (
    compute_immediate_transition_weights,
)

from model_extraction.reliability_ddra import (
    extract_reliability_components,
)

from model_extraction.reliability_distributions import (
    assign_reliability_distributions,
)

from model_extraction.expert_knowledge_load import load_raw_expert_knowledge, load_formalized_knowledge

from model_extraction.expert_knowledge_classification import classify_statement

from model_extraction.expert_knowledge_formalization import formalize_statement

from model_extraction.integrate_state_deviation_knowledge import integrate_state_deviation

from model_extraction.integrate_process_flow_knowledge import integrate_process_flow

from model_extraction.integrate_maintenance_knowledge import integrate_maintenance

from model_extraction.integrate_operational_knowledge import integrate_operational

from model_extraction.integrate_structural_knowledge import integrate_structural

from model_extraction.integrate_temporal_and_probabilistic_knowledge import integrate_temporal_and_probabilistic

from model_extraction.pyspn_visualization import visualize_pyspn_model


from statement_conflict_detection_and_resolution.conflict_detection import detect_conflicts_in_statement

from statement_conflict_detection_and_resolution.conflict_resolution import apply_recommendations

from statement_conflict_detection_and_resolution.decision_support import resolve_pending_clusters

from statement_conflict_detection_and_resolution.pyspn_visualization_with_conflicts import visualize_pyspn_model_with_conflicts

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def extract_spn(
    DATA_INPUT_FILE: str,
    EXPERT_KNOWLEDGE_INPUT_FILE: str,
    miner: str = "alpha",
):

    print("=" * 70)
    print(" DKDT Stochastic Petri Net Model Extraction")
    print("=" * 70)

    # ----------------------------------------------------------
    # Data-Driven Model Extraction
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Step 1: IoT Data Processing
    # ----------------------------------------------------------

    print("\n[1] Loading and preprocessing event log...")

    event_log = load_event_log(DATA_INPUT_FILE)

    #state_log = load_state_log(DATA_INPUT_FILE)

    cleaned_event_log, traces = preprocess_log(event_log)

    print(f"Cases: {len(traces)}")

    print("\n Removing previous expert knowledge statements...")

    

    # ----------------------------------------------------------
    # Step 2: Automated Expert Knowledge Processing -> As this case study starts from the rules available in the iMAKS dataset, the classification and formalization is skipped
    # ----------------------------------------------------------

    """
    
    print("\n[2.1] Load expert knowledge ... ")

    id, expert_knowledge = load_raw_expert_knowledge(
        EXPERT_KNOWLEDGE_INPUT_FILE,
    )

    print("expert knowledge", expert_knowledge)

    knowledge_df = pd.DataFrame(
        columns=["knowledge_type", "formalized_knowledge", "original", "id"]
    )

    print("\n[2.2] Classify expert knowledge statements... ")

    for index, knowledge_item in enumerate(expert_knowledge):

        knowledge_type = classify_statement(
            knowledge_item,
        )

        print(
            "\n[2.3 and 2.4] Orchestrate and formalize expert knowledge statements... "
        )

        print("knowledge type", knowledge_type)

        formalized_knowledge = formalize_statement(knowledge_type, knowledge_item)

        print("formalization", formalized_knowledge)

        knowledge_df.loc[len(knowledge_df)] = {
            "knowledge_type": knowledge_type,
            "formalized_knowledge": formalized_knowledge,
            "original": knowledge_item,
            "id": id[index],
        }


    """

    print("\n[2.5] Validate statements including conflict detection and resolution... ")

    records = read_jsonl("input_data/conflicted_expert_knowledge.jsonl")

    df = pd.DataFrame(records)

    conflict_detected_knowledge = detect_conflicts_in_statement(records=df, backend="nli", verbose=True)
    
    conflict_detected_knowledge = pd.DataFrame(conflict_detected_knowledge)

 
    

    # ----------------------------------------------------------
    # Step 3: Data-driven Stochastic Petri Net Model Extraction
    # ----------------------------------------------------------

    print("\n[3.1] Discovering process model...")

    discovered_net = discover_process_model(cleaned_event_log, algorithm=miner)

    print("\n[3.2] Building the stochastic Petri net structure...")

    spn = build_spn_structure(
        traces=traces,
        process_model=discovered_net,
    )

    print("\n[3.3] Detecting inhibitor arcs...")

    #spn = detect_inhibitor_arcs(
    #    spn=spn,
    #    traces=traces,
    #)

    print("\n[3.4] Detecting multiplicities...")

    spn = detect_arc_multiplicities(
        spn=spn,
        traces=traces,
    )

    print("\n[3.5] Detecting control-flow patterns...")

    spn = implement_control_flow_patterns(
        spn=spn,
        traces=traces,
    )

    print("\n[3.6] Fitting duration distributions...")

    spn = fit_transition_distributions(spn)

    print("\n[3.7] Computing immediate transition weights...")

    spn = compute_immediate_transition_weights(
        spn=spn,
        traces=traces,
    )


    visualize_pyspn_model(spn, "data-driven-model") 

    # ----------------------------------------------------------
    # Step 4: Visualization of Conflicts
    # ----------------------------------------------------------

    print("\n[4] Visualize conflicts  ... ")

    conflict_spn = copy.deepcopy(spn)

    for _, row in conflict_detected_knowledge.iterrows():

        conflicts = list()

        knowledge_type = str(row["category"]).strip()
        knowledge = str(row["formalization"])
        id = str(row["statement_id"]).strip()
        has_conflict = bool(row["has_conflict"])

        for finding in row["conflicts"]:
                  conflicts.append({"statement_id": finding["statement_id"], "confidence": 1.0, "method": finding["conflict_type"], "category": finding["category"]})


        for part_knowledge in knowledge.split(";"):

            
            if knowledge_type == "State_Deviation":
                integrate_state_deviation(conflict_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[4.1] Integrate state deviation knowledge ... ")

            elif knowledge_type == "Process_Flow":
                integrate_process_flow(conflict_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[4.2] Integrate process flow knowledge ... ")
                
            elif knowledge_type == "Maintenance":
                integrate_maintenance(conflict_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[4.3] Integrate maintenance knowledge ... ")

            elif knowledge_type == "Operational":
                integrate_operational(conflict_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[4.4] Integrate operational knowledge ... ")

            elif knowledge_type == "Structural":
                integrate_structural(conflict_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[4.5] Integrate structural knowledge ... ")

            elif knowledge_type == "Temporal_and_Probabilistic":
                integrate_temporal_and_probabilistic(conflict_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[4.6] Integrate temporal_and_probabilistic knowledge ... ")

    visualize_pyspn_model(conflict_spn, "DKDT-conflict-aware-model") 

    # ----------------------------------------------------------
    # Step 5: Resolution Recommendations
    # ----------------------------------------------------------


    print("\n[5] Resolution Recommendations ... ")

    resolved_spn = copy.deepcopy(spn)

    # 2. Recommendation generation
    state_log = load_state_log("input_data/timeseries_annotated.csv")

    recommended_records, pending_clusters = apply_recommendations(
        conflict_detected_knowledge.to_dict(orient="records"),
        ["confidence", "corroboration", "conservative"], 
        state_log=state_log,
        verbose = True
    )

    # ----------------------------------------------------------
    # Step 6: User Interface as Decision-Support
    # ----------------------------------------------------------


    print("\n[6] User Interface as Decision-Support ... ")

    # 3. Human UI
    result = resolve_pending_clusters(
        pending_clusters,
        recommended_records
    )

    result["kept"] = pd.DataFrame(result["kept"])


    # ----------------------------------------------------------
    # Step 7: Extraction of DT Models with Resolved Conflicts
    # ----------------------------------------------------------


    print("\n[7] Extraction of DT Models with Resolved Conflicts ... ")


        
    for _, row in result["kept"].iterrows():

        knowledge_type = str(row["category"]).strip()
        knowledge = str(row["formalization"])
        id = str(row["statement_id"]).strip()
        has_conflict = False

        conflicts = list()

        for finding in row["conflicts"]:

            conflicts.append({"statement_id": finding["statement_id"], "confidence": 1.0, "method": finding["conflict_type"], "category": finding["category"]})
        

        for part_knowledge in knowledge.split(";"):

            

            if knowledge_type == "State_Deviation":
                integrate_state_deviation(resolved_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[7.1] Integrate state deviation knowledge ... ")

                

            elif knowledge_type == "Process_Flow":
                integrate_process_flow(resolved_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[7.2] Integrate process flow knowledge ... ")

                

            elif knowledge_type == "Maintenance":
                integrate_maintenance(resolved_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[7.3] Integrate maintenance knowledge ... ")
                

            elif knowledge_type == "Operational":
                integrate_operational(resolved_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[7.4] Integrate operational knowledge ... ")
                

            elif knowledge_type == "Structural":
                integrate_structural(resolved_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[7.5] Integrate structural knowledge ... ")
                print("\n[5.6] Integrate temporal_and_probabilistic knowledge ... ")

            elif knowledge_type == "Temporal_and_Probabilistic":
                integrate_temporal_and_probabilistic(resolved_spn, part_knowledge, id, has_conflict, conflicts)
                print("\n[7.6] Integrate temporal_and_probabilistic knowledge ... ")


    # ----------------------------------------------------------
    # Step 8: Visualization of the extracted Data-Knowledge-Fused Stochastic Petri Net model
    # ----------------------------------------------------------
    
    print("\n[8] Visualize...")

    visualize_pyspn_model(resolved_spn, "DKDT-conflict-resolved-model")

    

    return resolved_spn



# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Input files
DATA_INPUT = "input_data/timeseries_annotated.csv"
EXPERT_KNOWLEDGE_INPUT = "input_data/conflicted_expert_knowledge.csv"

# Process discovery algorithm
MINER = "alpha"

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():

    DATA_INPUT_FILE = Path(DATA_INPUT)

    # Check input file exists
    if not DATA_INPUT_FILE.exists():
        raise FileNotFoundError(f"Event log not found: {DATA_INPUT_FILE}")

    print("Input event log:", DATA_INPUT_FILE)

    EXPERT_KNOWLEDGE_INPUT_FILE = Path(EXPERT_KNOWLEDGE_INPUT)

    # Check input file exists
    if not DATA_INPUT_FILE.exists():
        raise FileNotFoundError(f"Event log not found: {DATA_INPUT_FILE}")

    print("Miner:", MINER)
    
    # Clear the output file at the start of a new execution
    if os.path.exists("output/classified_and_formalized_expert_knowledge.jsonl"):
        os.remove("output/classified_and_formalized_expert_knowledge.jsonl")

    if os.path.exists("output/conflict_detected_expert_knowledge.jsonl"):
            os.remove("output/conflict_detected_expert_knowledge.jsonl")

    if os.path.exists("output/conflict_resolved_expert_knowledge.jsonl"):
            os.remove("output/conflict_resolved_expert_knowledge.jsonl")

    # Start Data-Knowledge-Fused Digital Twin Stochastic Petri Net model extraction pipeline
    extract_spn(
        DATA_INPUT_FILE=str(DATA_INPUT_FILE),
        EXPERT_KNOWLEDGE_INPUT_FILE=str(EXPERT_KNOWLEDGE_INPUT_FILE),
        miner=MINER,
    )

def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {n}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {n} must be a JSON object")
            rows.append(row)
    return rows


def load_state_log(path="../input_data/timeseries_annotated.csv"):
    """Long sensor CSV -> wide state log: one row per timestamp and station."""
    raw = pd.read_csv(path)
    raw = raw[raw["quality"].astype(str).str.upper() == "GOOD"]
    wide = raw.pivot_table(index=["timestamp", "station_id"], columns="sensor_type",
                           values="value", aggfunc="mean")
    wide.columns = [f"{c.lower()}_sensor" for c in wide.columns]   # TMP -> tmp_sensor
    wide = wide.reset_index().rename(columns={"station_id": "scope"})

    def states(g):
        words = set()
        for _, r in g.iterrows():
            normal = str(r["anomaly_label"]).upper() == "NORMAL" and str(r["alarm_flag"]).upper() == "NONE"
            words.add("operational" if normal else "alarm")
            if str(r["alarm_flag"]).upper() not in ("NONE", "NAN"):
                words.add(str(r["alarm_flag"]).lower())
            if isinstance(r["severity"], str):
                words.add(r["severity"].lower())
        return " ".join(sorted(words))

    state = (raw.groupby(["timestamp", "station_id"])
                .apply(states)
                .rename("state").reset_index()
                .rename(columns={"station_id": "scope"}))
    return wide.merge(state, on=["timestamp", "scope"]).drop(columns=["timestamp"])


# --------------------------------------------------------------------------
# Execute
# --------------------------------------------------------------------------

if __name__ == "__main__":

    main()
