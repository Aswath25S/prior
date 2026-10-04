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
    """Opening-time inputs to predict actual sales over one full trading day.

    History is simulated for 28 days before the exported period. Lagged fields
    use completed prior days only; opening stock is set before today's sales.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    regular_price = rng.uniform(3, 25, products).round(2)
    product_base = rng.uniform(15, 55, products)
    price_elasticity = rng.uniform(1.0, 2.2, products)
    shelf_space = rng.integers(2, 7, products)
    store_area = rng.integers(450, 1800, stores)
    catchment = rng.integers(4000, 25000, stores)
    usual_visitors = 180 + .23 * store_area + .006 * catchment
    history = {(s, p): [] for s in range(stores) for p in range(products)}
    demand_state = np.zeros((stores, products))
    # Warm-up supplies real simulated preceding observations for every exported lag.
    all_dates = pd.date_range("2024-12-04", periods=days + 28, freq="D")
    for day_idx, date in enumerate(all_dates):
        weekend = int(date.dayofweek >= 5)
        holiday = int((date.month, date.day) in {(1, 1), (7, 4), (12, 25)})
        payday = int(date.day <= 3 or date.day >= 25)
        seasonal = 1 + .08 * np.sin(2 * np.pi * date.dayofyear / 365)
        for store in range(stores):
            opening_hours = 8 if holiday else 10 if date.dayofweek == 6 else 12
            # A planning estimate available before opening, rather than actual footfall.
            expected_visitors = int(round(usual_visitors[store] * (1 + .25 * weekend + .08 * payday)
                                         * seasonal * opening_hours / 12))
            for product in range(products):
                past = history[store, product]
                mean_7d = float(np.mean(past[-7:])) if past else product_base[product]
                weekday_mean = float(np.mean(past[-28::7])) if len(past) >= 28 else mean_7d
                promo = int(rng.random() < .18)
                discount = int(rng.choice([10, 15, 20, 25])) if promo else 0
                price = round(regular_price[product] * (1 - discount / 100), 2)
                competitor_price = round(regular_price[product] * rng.uniform(.88, 1.12), 2)
                display = int(promo and rng.random() < .65)
                campaign = int(rng.random() < (.45 if promo else .06))
                shelf_facings = int(shelf_space[product] + 2 * display)
                # Stock is chosen using past sales and a rough planned uplift, not today's outcome.
                stock_plan = max(5, mean_7d * (1.35 + discount / 100 + .2 * display))
                opening_stock = max(0, int(rng.normal(stock_plan, max(3, stock_plan * .18))))
                if rng.random() < .06:
                    opening_stock = int(opening_stock * rng.uniform(.15, .6))
                expected = product_base[product] * expected_visitors / 450
                expected *= (regular_price[product] / price) ** price_elasticity[product]
                expected *= 1 + .45 * (competitor_price / price - 1)
                expected *= 1 + .18 * display + .12 * campaign
                expected *= 1 + .035 * (shelf_facings - 3)
                # Temporary local demand and ordinary customer variability remain unobserved.
                demand_state[store, product] = .7 * demand_state[store, product] + rng.normal(0, .07)
                expected *= np.exp(demand_state[store, product])
                demand = int(rng.poisson(rng.gamma(35, max(.1, expected) / 35)))
                sold = min(opening_stock, demand)
                if day_idx >= 28:
                    records.append({
                        "store_id": f"S-{store + 1:02d}", "product_id": f"P-{product + 1:03d}",
                        "date": date.date().isoformat(), "price": price, "discount_pct": discount,
                        "expected_store_visitors": expected_visitors,
                        "opening_stock_units": opening_stock,
                        "avg_units_sold_previous_7_days": round(mean_7d, 1),
                        "avg_units_sold_same_weekday_previous_4_weeks": round(weekday_mean, 1),
                        "units_sold": sold,
                    })
                past.append(sold)
    return pd.DataFrame(records)


def main() -> None:
    output = ROOT / "data" / "samples"
    output.mkdir(parents=True, exist_ok=True)
    manufacturing().to_csv(output / "synthetic_manufacturing.csv", index=False)
    retail().to_csv(output / "synthetic_retail.csv", index=False)
    print(f"Generated sample data in {output}")


if __name__ == "__main__":
    sys.exit(main())
