# ==========================================================================
# Imports
# ==========================================================================

from __future__ import annotations

import pandas as pd
import pm4py


# ==========================================================================
# Configuration
# ==========================================================================

CASE_COL = "batch_id"

ACTIVITY_COL = "station_id"

RESOURCE_COL = "sensor_id"

TIMESTAMP_COL = "timestamp"

#PROCESS_MODEL_COL = "process_model_id"

#TARGET_PROCESS_MODEL = "WF_102"


# ==========================================================================
# Activities to KEEP
# ==========================================================================

KEEP_ACTIVITIES = {
    "ST01_FILLING",
    "ST02_SEALING",
    "ST03_LABELLING",
    "ST04_PACKAGING",
    #"sm_2_transport",
    #"DM_2_lower",
}


# ==========================================================================
# Helper: normalize activity names for comparison
# ==========================================================================


def normalize_activity_name(activity):
    """
    Normalize activity names for comparison.

    Matching is:
        - case-insensitive
        - whitespace-insensitive

    Example:

        DM_2_lower
        dm_2_lower
        DM_2_LOWER

    all become:

        dm_2_lower
    """

    if pd.isna(activity):
        return ""

    return str(activity).strip().lower()


# ==========================================================================
# Helper: check whether activity should be kept
# ==========================================================================


def activity_should_be_kept(activity):
    """
    Determine whether an activity belongs to KEEP_ACTIVITIES.

    Handles the duplicated resource prefix produced by
    merge_resource_activity().

    Examples:

        hbw_2_unload
        hbw_2_hbw_2_unload

    Both are accepted.

    Likewise:

        DM_2_lower
        dm_2_dm_2_lower

    are both accepted.
    """

    activity = normalize_activity_name(activity)

    if not activity:
        return False

    # Normalize configured activities
    targets = {
        normalize_activity_name(a)
        for a in KEEP_ACTIVITIES
    }

    # --------------------------------------------------------------
    # Exact match
    # --------------------------------------------------------------

    if activity in targets:
        return True

    # --------------------------------------------------------------
    # Handle duplicated prefixes
    #
    # Example:
    #
    # hbw_2_hbw_2_unload
    #
    # ends with:
    #
    # hbw_2_unload
    # --------------------------------------------------------------

    for target in targets:

        if activity.endswith("_" + target):
            return True

    return False


# ==========================================================================
# Timestamp normalization
# ==========================================================================

def normalize_timestamp_column(df: pd.DataFrame):
    """
    Normalize timestamps.

    Supported input examples:

        23-06-2021 15:40:14.420000
        2021-06-23 15:40:14.420000+00:00
        2026-01-06T06:00:00

    Timezone information is removed afterwards.

    Invalid timestamps are removed.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------------

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="coerce",
        format="mixed",
    )

    # --------------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------------

    df = df.dropna(
        subset=[TIMESTAMP_COL]
    ).copy()

    # --------------------------------------------------------------
    # Remove timezone
    # --------------------------------------------------------------

    if hasattr(
        df[TIMESTAMP_COL].dt,
        "tz_localize"
    ):

        try:

            df[TIMESTAMP_COL] = (
                df[TIMESTAMP_COL]
                .dt
                .tz_localize(None)
            )

        except TypeError:

            # Already timezone-naive
            pass

    return df

# ==========================================================================
# Activity normalization
# ==========================================================================


def merge_resource_activity(df):
    """
    Merge resource and activity names.

    The resource is added as a prefix.

    Example:

        resource:
            hbw_2

        activity:
            unload

        result:
            hbw_2_unload

    If the activity already contains the resource prefix,
    duplication is avoided where possible.
    """

    df = df.copy()

    if (
        RESOURCE_COL not in df.columns
        or ACTIVITY_COL not in df.columns
    ):
        return df

    def normalize_activity(resource, activity):

        # ----------------------------------------------------------
        # Convert safely to strings
        # ----------------------------------------------------------

        resource = str(resource).strip()

        activity = str(activity).strip("/").strip()

        # ----------------------------------------------------------
        # Split activity path
        # ----------------------------------------------------------

        parts = activity.split("/")

        # ----------------------------------------------------------
        # Resource prefix
        #
        # Example:
        #
        # hbw_2 -> hbw
        # ----------------------------------------------------------

        resource_prefix = (
            resource
            .split("_")[0]
            .lower()
        )

        # ----------------------------------------------------------
        # Remove duplicated first path element
        # ----------------------------------------------------------

        if (
            len(parts) > 1
            and parts[0].lower() == resource_prefix
        ):

            parts = parts[1:]

        # ----------------------------------------------------------
        # Build final activity name
        # ----------------------------------------------------------

        return (
            resource
            + "_"
            + "_".join(parts)
        )

    df[ACTIVITY_COL] = [
        normalize_activity(
            resource,
            activity,
        )
        for resource, activity
        in zip(
            df[RESOURCE_COL],
            df[ACTIVITY_COL],
        )
    ]

    return df


# ==========================================================================
# Filter activities
# ==========================================================================


def filter_kept_activities(df):
    """
    Keep ONLY activities specified in KEEP_ACTIVITIES.

    All lifecycle transitions belonging to those activities
    are retained.

    For example, if hbw_2_unload has:

        scheduled
        start
        complete

    all three events remain.

    Activities not listed in KEEP_ACTIVITIES are removed.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Build boolean mask
    # --------------------------------------------------------------

    #df = df[df["event_type"].isna() | df["event_type"].str.startswith("start", na=False)]

    #
    mask = df[ACTIVITY_COL].apply(
        activity_should_be_kept
    )

    # --------------------------------------------------------------
    # Filter
    # --------------------------------------------------------------

    filtered = df.loc[mask].copy()

    # --------------------------------------------------------------
    # Sort
    # --------------------------------------------------------------

    filtered = (
        filtered
        .sort_values(
            by=[
                CASE_COL,
                TIMESTAMP_COL,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return filtered


# ==========================================================================
# Print filtering statistics
# ==========================================================================


def print_filter_statistics(
    before_df,
    after_df,
):
    """
    Print useful information to verify filtering.
    """

    print()
    print("=" * 80)
    print("FILTERING STATISTICS")
    print("=" * 80)

    print(
        f"Events before filtering: "
        f"{len(before_df)}"
    )

    print(
        f"Events after filtering:  "
        f"{len(after_df)}"
    )

    print(
        f"Cases before filtering:  "
        f"{before_df[CASE_COL].nunique()}"
    )

    print(
        f"Cases after filtering:   "
        f"{after_df[CASE_COL].nunique()}"
    )

    print()
    print("Activities retained:")
    print("-" * 80)

    if len(after_df) == 0:

        print("NO ACTIVITIES REMAIN")

    else:

        counts = (
            after_df[ACTIVITY_COL]
            .value_counts()
        )

        for activity, count in counts.items():

            print(
                f"{activity:<60} {count}"
            )

    print()
    print("=" * 80)
    print()


# ==========================================================================
# Load event log
# ==========================================================================


def load_event_log(filename: str):
    """
    Load CSV or XES event log.

    Processing:

        1. Load
        2. Filter process model
        3. Normalize case ID
        4. Normalize timestamps
        5. Merge resource/activity
        6. Sort
        7. Keep configured activities
        8. Return DataFrame
    """

    filename_lower = filename.lower()

    # ======================================================================
    # CSV
    # ======================================================================

    if filename_lower.endswith(".csv"):

        print(
            f"Loading CSV event log: {filename}"
        )

        df = pd.read_csv(
            filename,
            sep=",",
            dtype={16: str}
        )

        print(df.head(10))

        # --------------------------------------------------------------
        # Filter process model
        # --------------------------------------------------------------

        #if PROCESS_MODEL_COL in df.columns:

        #    df = df[
        #        df[PROCESS_MODEL_COL]
        #        == TARGET_PROCESS_MODEL
        #    ].copy()

        # --------------------------------------------------------------
        # Case ID
        # --------------------------------------------------------------

        if CASE_COL not in df.columns:

            raise KeyError(
                f"Required column '{CASE_COL}' "
                f"not found in CSV."
            )

        df[CASE_COL] = (
            df[CASE_COL]
            .astype(str)
        )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        df = normalize_timestamp_column(df)

        # --------------------------------------------------------------
        # Activity
        # --------------------------------------------------------------

        if ACTIVITY_COL not in df.columns:

            raise KeyError(
                f"Required column '{ACTIVITY_COL}' "
                f"not found in CSV."
            )

        # --------------------------------------------------------------
        # Resource/activity normalization
        # --------------------------------------------------------------

        #df = merge_resource_activity(df)

        # --------------------------------------------------------------
        # Sort BEFORE filtering
        # --------------------------------------------------------------

        df = (
            df
            .sort_values(
                by=[
                    CASE_COL,
                    TIMESTAMP_COL,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        # --------------------------------------------------------------
        # Filter activities
        # --------------------------------------------------------------

        before_filtering = df.copy()

        df = filter_kept_activities(df)

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        print_filter_statistics(
            before_filtering,
            df,
        )

        return df

    # ======================================================================
    # XES
    # ======================================================================

    elif filename_lower.endswith(".xes"):

        print(
            f"Loading XES event log: {filename}"
        )

        if pm4py is None:

            raise ImportError(
                "PM4Py is required for XES files."
            )

        # --------------------------------------------------------------
        # Read XES
        # --------------------------------------------------------------

        log = pm4py.read_xes(
            filename
        )

        # --------------------------------------------------------------
        # Convert to DataFrame
        # --------------------------------------------------------------

        df = pm4py.convert_to_dataframe(
            log
        )

        # --------------------------------------------------------------
        # Process model
        # --------------------------------------------------------------

        #if PROCESS_MODEL_COL in df.columns:

        #    df = df[
        #        df[PROCESS_MODEL_COL]
        #        == TARGET_PROCESS_MODEL
        #    ].copy()

        # --------------------------------------------------------------
        # Case ID
        # --------------------------------------------------------------

        if CASE_COL not in df.columns:

            raise KeyError(
                f"Required column '{CASE_COL}' "
                f"not found in XES."
            )

        df[CASE_COL] = (
            df[CASE_COL]
            .astype(str)
        )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        df = normalize_timestamp_column(df)

        # --------------------------------------------------------------
        # Resource/activity normalization
        # --------------------------------------------------------------

        #df = merge_resource_activity(df)

        # --------------------------------------------------------------
        # Sort
        # --------------------------------------------------------------

        df = (
            df
            .sort_values(
                by=[
                    CASE_COL,
                    TIMESTAMP_COL,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        # --------------------------------------------------------------
        # Filter activities
        # --------------------------------------------------------------

        before_filtering = df.copy()

        df = filter_kept_activities(df)

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        print_filter_statistics(
            before_filtering,
            df,
        )

        return df

    # ======================================================================
    # Unsupported format
    # ======================================================================

    else:

        raise ValueError(
            "Supported formats: CSV, XES"
        )


# ==========================================================================
# Load state log
# ==========================================================================


def load_state_log(filename: str):
    """
    Load CSV or XES state log.

    Uses exactly the same preprocessing as the event log.
    """

    # Reuse the same robust loader.

    filename = "input_data/state_log.csv"

    filename_lower = filename.lower()
    
        # ======================================================================
        # CSV
        # ======================================================================
    
    if filename_lower.endswith(".csv"):
    
        print(
            f"Loading CSV event log: {filename}"
        )

        df = pd.read_csv(
            filename,
            sep=";",
        )

        # --------------------------------------------------------------
        # Filter process model
        # --------------------------------------------------------------

        #if PROCESS_MODEL_COL in df.columns:

        #    df = df[
        #        df[PROCESS_MODEL_COL]
        #        == TARGET_PROCESS_MODEL
        #    ].copy()

        # --------------------------------------------------------------
        # Case ID
        # --------------------------------------------------------------

        if RESOURCE_COL not in df.columns:

            raise KeyError(
                f"Required column '{RESOURCE_COL}' "
                f"not found in CSV."
            )

        df[RESOURCE_COL] = (
            df[RESOURCE_COL]
            .astype(str)
        )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        df = normalize_timestamp_column(df)

        # --------------------------------------------------------------
        # Activity
        # --------------------------------------------------------------

        #if ACTIVITY_COL not in df.columns:

        #    raise KeyError(
        #        f"Required column '{ACTIVITY_COL}' "
        #        f"not found in CSV."
        #    )

        # --------------------------------------------------------------
        # Resource/activity normalization
        # --------------------------------------------------------------

        #df = merge_resource_activity(df)

        # --------------------------------------------------------------
        # Sort BEFORE filtering
        # --------------------------------------------------------------

        #df = df["state"].str.startswith("failure", na=False)

        #df = (
        #    df
        #    .sort_values(
        #        by=[
        #            RESOURCE_COL,
        #            TIMESTAMP_COL,
        #        ],
        #        kind="stable",
        #    )
        #    .reset_index(drop=True)
        #)

        # --------------------------------------------------------------
        # Filter activities
        # --------------------------------------------------------------

        #before_filtering = df.copy()

        #df = filter_kept_activities(df)

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        #print_filter_statistics(
        #    before_filtering,
        #    df,
        #)

        return df


# ==========================================================================
# Sort log
# ==========================================================================


def sort_log(df):
    """
    Sort event log by case and timestamp.
    """

    return (
        df
        .sort_values(
            by=[
                CASE_COL,
                TIMESTAMP_COL,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


# ==========================================================================
# Extract traces
# ==========================================================================


def extract_traces(df):
    """
    Extract traces from the already-filtered DataFrame.

    Important:
        Filtering happens BEFORE this function.

    Therefore every trace contains only the activities defined
    in KEEP_ACTIVITIES.
    """

    traces = {}

    # --------------------------------------------------------------
    # Make sure data is sorted
    # --------------------------------------------------------------

    df = sort_log(df)

    # --------------------------------------------------------------
    # Group by case
    # --------------------------------------------------------------

    grouped = df.groupby(
        CASE_COL,
        sort=False,
    )

    for case_id, trace in grouped:

        trace = (
            trace
            .sort_values(
                TIMESTAMP_COL,
                kind="stable",
            )
            .reset_index(drop=True)
        )

        events = []

        for _, row in trace.iterrows():

            events.append(
                {
                    "activity": row.get(
                        ACTIVITY_COL,
                        None,
                    ),

                    "timestamp": row.get(
                        TIMESTAMP_COL,
                        None,
                    ),

                    "lifecycle_state": row.get(
                        "lifecycle:state",
                        None,
                    ),

                    "lifecycle_transition": row.get(
                        "lifecycle:transition",
                        None,
                    ),

                    "resource": row.get(
                        RESOURCE_COL,
                        None,
                    ),

                    "row": row,
                }
            )

        traces[str(case_id)] = events

    return traces


# ==========================================================================
# Complete preprocessing
# ==========================================================================


def preprocess_log(
    log: pd.DataFrame,
):
    """
    Complete preprocessing pipeline for an already-loaded DataFrame.

    Processing:

        1. Copy
        2. Normalize case IDs
        3. Normalize timestamps
        4. Merge resource/activity
        5. Normalize lifecycle fields
        6. Sort chronologically
        7. Keep only configured activities
        8. Extract traces
    """

    log = log.copy()

    # ======================================================================
    # Case ID
    # ======================================================================

    if CASE_COL not in log.columns:

        raise KeyError(
            f"Required column '{CASE_COL}' "
            f"not found."
        )

    log[CASE_COL] = (
        log[CASE_COL]
        .astype(str)
    )

    # ======================================================================
    # Process model filtering
    # ======================================================================

    #if PROCESS_MODEL_COL in log.columns:

    #    log = log[
    #        log[PROCESS_MODEL_COL]
    #        == TARGET_PROCESS_MODEL
    #    ].copy()

    # ======================================================================
    # Timestamp normalization
    # ======================================================================

    log = normalize_timestamp_column(
        log
    )

    # ======================================================================
    # Activity normalization
    # ======================================================================

    if ACTIVITY_COL not in log.columns:

        raise KeyError(
            f"Required column '{ACTIVITY_COL}' "
            f"not found."
        )

    #log = merge_resource_activity(
    #    log
    #)

    # ======================================================================
    # Lifecycle normalization
    # ======================================================================

    if "lifecycle:state" in log.columns:

        log["lifecycle:state"] = (
            log["lifecycle:state"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "lifecycle:transition" in log.columns:

        log["lifecycle:transition"] = (
            log["lifecycle:transition"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # ======================================================================
    # Sort BEFORE filtering
    # ======================================================================

    log = (
        log
        .sort_values(
            by=[
                CASE_COL,
                TIMESTAMP_COL,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ======================================================================
    # Filter activities
    # ======================================================================

    before_filtering = log.copy()

    log = filter_kept_activities(
        log
    )

    # ======================================================================
    # Statistics
    # ======================================================================

    print_filter_statistics(
        before_filtering,
        log,
    )

    # ======================================================================
    # Extract traces
    # ======================================================================

    traces = extract_traces(
        log
    )

    return log, traces


# ==========================================================================
# Debug helper
# ==========================================================================


def print_trace(
    traces,
    case_id,
):
    """
    Print one trace in chronological order.

    Useful for verifying that filtering works correctly.
    """

    case_id = str(case_id)

    if case_id not in traces:

        print(
            f"Case '{case_id}' not found."
        )

        return

    print()
    print("=" * 100)
    print(f"TRACE: {case_id}")
    print("=" * 100)

    for number, event in enumerate(
        traces[case_id],
        start=1,
    ):

        print(
            f"{number:4d} | "
            f"{event['timestamp']} | "
            f"{event['activity']} | "
            f"{event['lifecycle_transition']} | "
            f"{event['lifecycle_state']}"
        )

    print("=" * 100)