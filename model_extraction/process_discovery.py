# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

from __future__ import annotations

import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

# --------------------------------------------------------------------------
# Confgurations
# --------------------------------------------------------------------------

CASE_COL = "batch_id"

ACTIVITY_COL = "station_id"

TIMESTAMP_COL = "timestamp"

# --------------------------------------------------------------------------
# DataFrame → PM4Py Event Log
# --------------------------------------------------------------------------


def dataframe_to_eventlog(df: pd.DataFrame):
    """
    Converts a pandas dataframe into a PM4Py EventLog.
    """

    df = df.rename(columns={
        CASE_COL: "case:concept:name",
        ACTIVITY_COL: "concept:name",
        TIMESTAMP_COL: "time:timestamp",
    })

    return log_converter.apply(
        df,
        variant=log_converter.Variants.TO_EVENT_LOG
    )


# --------------------------------------------------------------------------
# Alpha Miner
# --------------------------------------------------------------------------


def alpha_discovery(event_log):
    """
    Discover a Petri net using Alpha Miner.
    """

    net, initial_marking, final_marking = alpha_miner.apply(event_log)

    return {
        "net": net,
        "initial_marking": initial_marking,
        "final_marking": final_marking,
        "algorithm": "alpha",
    }


# --------------------------------------------------------------------------
# Main interface
# --------------------------------------------------------------------------


def discover_process_model(cleaned_log: pd.DataFrame, algorithm: str = "alpha"):
    """
    Discover the process model.
    """

    event_log = dataframe_to_eventlog(cleaned_log)

    algorithm = algorithm.lower()

    if algorithm == "alpha":
        return alpha_discovery(event_log)

    elif algorithm == "alpha":
        return alpha_discovery(event_log)

    else:
        raise ValueError(
            f"Unknown discovery algorithm '{algorithm}'. " "Choose 'alpha' or 'alpha'."
        )
