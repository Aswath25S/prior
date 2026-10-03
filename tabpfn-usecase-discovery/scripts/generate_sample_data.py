from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def manufacturing(seed: int = 42, rows: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.5, 18, rows)
    hours = np.clip(age * rng.normal(1500, 250, rows), 300, None)
    temperature = rng.normal(67 + age * 0.35, 7, rows)
    vibration = np.clip(rng.gamma(2.2, 0.8, rows) + age * 0.045, 0, None)
    maintenance = rng.poisson(np.clip(1.5 + age * 0.05, 0.2, None), rows)
    load = np.clip(rng.normal(72, 16, rows), 20, 110)
    logits = -5.2 + 0.045 * (temperature - 65) + 0.42 * vibration + 0.018 * (load - 65) + 0.045 * age - 0.16 * maintenance
    probability = 1 / (1 + np.exp(-logits))
    failure = rng.binomial(1, probability)
    resolution_days = np.where(failure == 1, rng.integers(1, 20, rows).astype(float), np.nan)
    frame = pd.DataFrame({
        "machine_id": [f"M-{1000 + i:05d}" for i in range(rows)],
        "equipment_age_years": age.round(1),
        "operating_hours": hours.round(0),
        "temperature_c": temperature.round(1),
        "vibration_level": vibration.round(2),
        "maintenance_count_90d": maintenance,
        "operating_load": load.round(1),
        "failure_resolution_days": resolution_days,
        "failure_next_30_days": failure,
    })
    for col, rate in {"temperature_c": 0.035, "vibration_level": 0.025, "maintenance_count_90d": 0.015}.items():
        frame.loc[rng.random(rows) < rate, col] = np.nan
    return frame


def retail(seed: int = 84, days: int = 240, stores: int = 4, products: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    product_base = rng.uniform(18, 55, products)
    product_price = rng.uniform(8, 40, products)
    store_effect = rng.normal(0, 4, stores)
    for day_idx, date in enumerate(dates):
        seasonal = 6 * np.sin(2 * np.pi * day_idx / 7) + 5 * np.sin(2 * np.pi * day_idx / 365)
        holiday = int((date.month == 12 and date.day >= 20) or (date.month == 7 and date.day == 4))
        for store in range(stores):
            for product in range(products):
                promo = int(rng.random() < 0.13)
                price = product_price[product] * (0.82 if promo else 1.0) * rng.normal(1, 0.025)
                expected = product_base[product] + store_effect[store] + seasonal + 13 * promo + 8 * holiday - 0.75 * (price - product_price[product])
                sold = max(0, int(rng.normal(max(2, expected), 6)))
                inventory = max(sold, int(rng.normal(85, 22)))
                records.append({"store_id": f"S-{store + 1:02d}", "product_id": f"P-{product + 1:03d}", "date": date.date().isoformat(),
                                "price": round(price, 2), "promotion_flag": promo, "holiday_flag": holiday,
                                "inventory_level": inventory, "units_sold": sold})
    frame = pd.DataFrame(records)
    frame.loc[rng.random(len(frame)) < 0.01, "price"] = np.nan
    return frame


def main() -> None:
    output = ROOT / "data" / "samples"
    output.mkdir(parents=True, exist_ok=True)
    manufacturing().to_csv(output / "synthetic_manufacturing.csv", index=False)
    retail().to_csv(output / "synthetic_retail.csv", index=False)
    print(f"Generated sample data in {output}")


if __name__ == "__main__":
    sys.exit(main())
