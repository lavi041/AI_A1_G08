import csv
import random
from pathlib import Path

random.seed(2026)

out = Path("data/AI_A1_G08.csv")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []

for i in range(1, 101):
    plot_area = round(random.uniform(0.5, 5.0), 2)
    rainfall = round(random.uniform(650, 1400), 1)
    soil_ph = round(random.uniform(5.2, 7.4), 2)
    seed_kg = round(plot_area * random.uniform(18, 28), 1)
    distance = round(random.uniform(1, 45), 1)
    arrival = round(random.uniform(5, 22), 1)

    # Synthetic harvest-yield relationship
    yield_kg = (
        plot_area * 820
        + rainfall * 0.65
        + soil_ph * 115
        + seed_kg * 10
        - distance * 7
        - abs(arrival - 10) * 12
        + random.gauss(0, 180)
    )
    yield_kg = max(250, round(yield_kg, 1))

    # Synthetic dispatch-attention label
    attention_score = (
        (distance > 30)
        + (arrival < 7 or arrival > 19)
        + (rainfall > 1250)
        + (yield_kg > 4000)
    )

    dispatch_attention = 1 if attention_score >= 2 else 0

    rows.append([
        f"HL-{i:04d}",
        plot_area,
        rainfall,
        soil_ph,
        seed_kg,
        distance,
        arrival,
        yield_kg,
        dispatch_attention,
    ])

with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "record_id",
        "plot_area_ha",
        "rainfall_mm",
        "soil_ph",
        "seed_kg",
        "distance_km",
        "arrival_hour",
        "actual_yield_kg",
        "dispatch_attention",
    ])
    writer.writerows(rows)

print(f"Created {out} with {len(rows)} records.")
