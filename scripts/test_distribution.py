import sys

import pandas as pd

from apps.api.schemas import ScoreWindowRequest, SensorReading
from apps.api.scoring import ScoringService
from apps.api.synthetic import generate_sensor_window


def build_request(engine_id: int, degradation: str) -> ScoreWindowRequest:
    raw_window = generate_sensor_window(engine_id, degradation, window_size=30)
    readings = [SensorReading(**r) for r in raw_window]
    return ScoreWindowRequest(engine_id=engine_id, window=readings)


def main() -> None:
    service = ScoringService()
    service.load_models()

    if not service.is_loaded:
        print("ERROR: Models not loaded. Run training first.")
        sys.exit(1)

    degradation_levels = ["healthy", "mid", "critical"]
    engines = list(range(1, 11))  # engines 1-10
    repeats = 7  # 7 repeats x 10 engines x 3 levels = 210 samples

    rows = []
    total = len(degradation_levels) * len(engines) * repeats
    count = 0

    for deg in degradation_levels:
        for engine_id in engines:
            for _ in range(repeats):
                count += 1
                request = build_request(engine_id, deg)
                try:
                    result = service.score(request)
                    rows.append(
                        {
                            "degradation": deg,
                            "engine_id": engine_id,
                            "forecast_rul": result["forecast"],
                            "failure_prob": result["failure_probability_next_30_cycles"],
                            "anomaly_score": result["anomaly_score"],
                            "anomaly_level": result["anomaly_level"],
                            "alert_level": result["alert_level"],
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "degradation": deg,
                            "engine_id": engine_id,
                            "forecast_rul": None,
                            "failure_prob": None,
                            "anomaly_score": None,
                            "anomaly_level": "error",
                            "alert_level": f"ERROR: {e}",
                        }
                    )
                if count % 30 == 0:
                    print(f"  Progress: {count}/{total}", file=sys.stderr)

    df = pd.DataFrame(rows)

    # ---- Summary by degradation level ----
    print("\n" + "=" * 80)
    print("DISTRIBUTION SUMMARY")
    print("=" * 80)

    for deg in degradation_levels:
        subset = df[df["degradation"] == deg]
        valid = subset.dropna(subset=["forecast_rul"])
        print(f"\n--- {deg.upper()} ({len(valid)} samples) ---")

        if valid.empty:
            print("  No valid results")
            continue

        print(
            f"  Forecast RUL     : mean={valid['forecast_rul'].mean():.1f}  "
            f"min={valid['forecast_rul'].min():.1f}  "
            f"max={valid['forecast_rul'].max():.1f}  "
            f"std={valid['forecast_rul'].std():.1f}"
        )

        print(
            f"  Failure Prob     : mean={valid['failure_prob'].mean():.3f}  "
            f"min={valid['failure_prob'].min():.3f}  "
            f"max={valid['failure_prob'].max():.3f}  "
            f"std={valid['failure_prob'].std():.3f}"
        )

        print(
            f"  Anomaly Score    : mean={valid['anomaly_score'].mean():.3f}  "
            f"min={valid['anomaly_score'].min():.3f}  "
            f"max={valid['anomaly_score'].max():.3f}  "
            f"std={valid['anomaly_score'].std():.3f}"
        )

        print(f"  Anomaly Level    : {dict(valid['anomaly_level'].value_counts())}")
        print(f"  Alert Level      : {dict(valid['alert_level'].value_counts())}")

    # ---- Cross-tab: degradation vs alert level ----
    print("\n" + "=" * 80)
    print("CROSS-TAB: Degradation vs Alert Level")
    print("=" * 80)
    valid_all = df.dropna(subset=["forecast_rul"])
    if not valid_all.empty:
        ct = pd.crosstab(valid_all["degradation"], valid_all["alert_level"])
        ct = ct.reindex(
            index=degradation_levels,
            columns=["NORMAL", "WARNING", "CRITICAL"],
            fill_value=0,
        )
        print(ct.to_string())

    # ---- Cross-tab: degradation vs anomaly level ----
    print("\n" + "=" * 80)
    print("CROSS-TAB: Degradation vs Anomaly Level")
    print("=" * 80)
    if not valid_all.empty:
        ct2 = pd.crosstab(valid_all["degradation"], valid_all["anomaly_level"])
        ct2 = ct2.reindex(index=degradation_levels, columns=["low", "medium", "high"], fill_value=0)
        print(ct2.to_string())

    print("\n" + "=" * 80)
    print("Use these numbers to decide if alert logic thresholds need adjustment.")
    print("=" * 80)


if __name__ == "__main__":
    main()
