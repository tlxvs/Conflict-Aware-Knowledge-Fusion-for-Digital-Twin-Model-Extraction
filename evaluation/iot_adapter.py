"""Converts the long-format sensor CSV (timeseries_annotated.csv) into the state log used by the IoT overlap criterion."""
import pandas as pd


def load_state_log(path="../input_data/timeseries_annotated.csv"):
    raw = pd.read_csv(path)
    raw = raw[raw["quality"].astype(str).str.upper() == "GOOD"]
    wide = raw.pivot_table(index=["timestamp", "station_id"], columns="sensor_type", values="value", aggfunc="mean")
    wide.columns = [f"{c.lower()}_sensor" for c in wide.columns]            # TMP -> tmp_sensor (variable names of the rules)
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

    state = pd.DataFrame([(ts, st, states(g)) for (ts, st), g in raw.groupby(["timestamp", "station_id"])], columns=["timestamp", "scope", "state"])
    return wide.merge(state, on=["timestamp", "scope"]).drop(columns=["timestamp"])
