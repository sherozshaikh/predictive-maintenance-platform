"""Synthetic sensor window generator calibrated against real CMAPSS FD001 data.

Each degradation profile uses baselines and noise levels derived from the
actual training data at the corresponding RUL stage:
  - healthy : RUL > 100 cycles  (early life, no degradation)
  - mid     : RUL 30-60 cycles  (mid-life, moderate wear)
  - critical: RUL < 15 cycles   (near failure)

Drift-per-cycle values simulate the sensor trends observed in CMAPSS
across a 30-cycle window at each stage.
"""

import random

# Baselines per degradation level: {sensor: (mean, noise_std, drift_per_cycle)}
# Derived from real CMAPSS FD001 statistics at each RUL stage.
SENSOR_PROFILES: dict[str, dict[str, tuple[float, float, float]]] = {
    "healthy": {
        "s2": (642.42, 0.37, 0.0),
        "s3": (1587.50, 4.67, 0.0),
        "s4": (1403.78, 6.01, 0.0),
        "s7": (553.86, 0.62, 0.0),
        "s8": (2388.06, 0.05, 0.0),
        "s9": (9057.83, 8.73, 0.0),
        "s11": (47.38, 0.17, 0.0),
        "s12": (521.83, 0.50, 0.0),
        "s13": (2388.06, 0.05, 0.0),
        "s14": (8138.71, 8.18, 0.0),
        "s15": (8.4217, 0.027, 0.0),
        "s17": (392.41, 1.15, 0.0),
        "s20": (38.91, 0.13, 0.0),
        "s21": (23.35, 0.08, 0.0),
    },
    "mid": {
        "s2": (642.92, 0.36, 0.008),
        "s3": (1593.45, 4.57, 0.10),
        "s4": (1413.79, 5.74, -0.20),
        "s7": (552.90, 0.58, -0.015),
        "s8": (2388.13, 0.05, 0.001),
        "s9": (9071.52, 20.39, 0.30),
        "s11": (47.69, 0.16, 0.005),
        "s12": (521.02, 0.47, -0.015),
        "s13": (2388.13, 0.06, 0.001),
        "s14": (8147.94, 18.53, 0.25),
        "s15": (8.4613, 0.025, -0.0005),
        "s17": (393.97, 1.12, 0.03),
        "s20": (38.72, 0.13, -0.003),
        "s21": (23.24, 0.07, -0.002),
    },
    "critical": {
        "s2": (643.53, 0.31, 0.020),
        "s3": (1600.58, 4.32, 0.30),
        "s4": (1425.89, 4.86, -0.50),
        "s7": (551.76, 0.52, -0.040),
        "s8": (2388.21, 0.07, 0.002),
        "s9": (9092.82, 47.48, 1.50),
        "s11": (48.06, 0.14, 0.012),
        "s12": (520.05, 0.43, -0.030),
        "s13": (2388.21, 0.07, 0.002),
        "s14": (8163.23, 42.89, 1.20),
        "s15": (8.5098, 0.022, -0.0010),
        "s17": (395.82, 1.06, 0.08),
        "s20": (38.50, 0.11, -0.008),
        "s21": (23.10, 0.07, -0.005),
    },
}

# Near-constant sensors (not selected for modeling, but required by schema)
CONSTANT_SENSORS: dict[str, tuple[float, float]] = {
    "s1": (518.67, 0.5),
    "s5": (14.62, 0.01),
    "s6": (21.61, 0.01),
    "s10": (1.3, 0.001),
    "s16": (0.03, 0.001),
    "s18": (2388.0, 0.1),
    "s19": (100.0, 0.01),
}


# Cycle offset per degradation level: simulates where in the engine's
# lifetime this window sits.  CMAPSS engines run 128-362 cycles.
# The model uses cycle_norm (cycle / max_cycle) as a feature, so
# the absolute cycle number affects the prediction.
CYCLE_OFFSETS: dict[str, int] = {
    "healthy": 1,  # early life: cycles 1-30
    "mid": 100,  # mid life: cycles 100-130
    "critical": 200,  # late life: cycles 200-230
}


def generate_sensor_window(engine_id: int, degradation: str, window_size: int = 30) -> list[dict]:
    profile = SENSOR_PROFILES.get(degradation, SENSOR_PROFILES["mid"])
    cycle_offset = CYCLE_OFFSETS.get(degradation, 1)
    readings = []
    for i in range(window_size):
        cycle = cycle_offset + i
        reading: dict = {
            "cycle": cycle,
            "op1": round(random.gauss(0.0, 0.002), 4),
            "op2": round(random.gauss(0.0, 0.0003), 4),
            "op3": 100.0,
        }
        # Selected sensors with degradation-specific baselines + drift
        for sensor, (base, noise_std, drift) in profile.items():
            value = base + drift * (i + 1) + random.gauss(0, noise_std)
            reading[sensor] = round(value, 4)
        # Constant sensors (near-zero variance in CMAPSS FD001)
        for sensor, (base, noise_std) in CONSTANT_SENSORS.items():
            reading[sensor] = round(base + random.gauss(0, noise_std), 4)
        readings.append(reading)
    return readings
