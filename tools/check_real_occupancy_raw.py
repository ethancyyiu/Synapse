"""Write-once raw-sensor feasibility check for docs/real-occupancy-protocol.md."""

import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/occupancy_sensor_anonymized.csv"
PROTOCOL = ROOT / "docs/real-occupancy-protocol.md"
RESULT = ROOT / "data/real_occupancy_raw_feasibility.json"
SOURCE_HASH = "1bc10d1654f2856b18e034fbf4d7d97b6d0dfd6b04d5863d07045379d9702e3c"
TRAIN_PARTITION = "training"
HELDOUT_PARTITION = "heldout"
EXCLUDED_PARTITION = "excluded"
CLASS_NAMES = ("empty", "partly occupied", "full")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def category(row):
    count = int(row["Room_Occupancy_Count"])
    if count not in (0, 1, 2, 3):
        raise ValueError(f"unexpected occupant count {count}")
    return 0 if count == 0 else 1 if count < 3 else 2


def readings(row):
    values = (
        float(row["S5_CO2"]),
        float(row["mean_S1_to_S4_Sound"]),
        float(row["S6_PIR_plus_S7_PIR"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite sensor reading")
    return values


def load_rows():
    csv_bytes = SOURCE.read_bytes()
    if sha256(csv_bytes) != SOURCE_HASH:
        raise ValueError("local sensor CSV differs from its locked SHA-256")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    required = {"partition", "Room_Occupancy_Count", "S5_CO2",
                "mean_S1_to_S4_Sound", "S6_PIR_plus_S7_PIR"}
    if set(reader.fieldnames or []) != required:
        raise ValueError("unexpected local sensor columns")
    rows = [{"csv_line": line, "partition": row["partition"],
             "class": category(row),
             "readings": readings(row)} for line, row in enumerate(reader, start=2)]
    if len(rows) != 10129:
        raise ValueError(f"expected 10129 sensor rows, got {len(rows)}")
    partitions = {row["partition"] for row in rows}
    expected = {TRAIN_PARTITION, HELDOUT_PARTITION, EXCLUDED_PARTITION}
    if partitions != expected:
        raise ValueError(f"unexpected sensor partitions: {sorted(partitions)}")
    return rows, sha256(csv_bytes)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("preflight", "run"):
        raise SystemExit("usage: python tools/check_real_occupancy_raw.py preflight|run")
    rows, csv_hash = load_rows()
    training = [row for row in rows if row["partition"] == TRAIN_PARTITION]
    heldout = [row for row in rows if row["partition"] == HELDOUT_PARTITION]
    counts = {partition: dict(Counter(row["class"] for row in rows
                                      if row["partition"] == partition))
              for partition in (TRAIN_PARTITION, HELDOUT_PARTITION,
                                EXCLUDED_PARTITION)}
    by_class = [[row for row in training if row["class"] == label] for label in range(3)]
    if any(len(group) < 96 for group in by_class):
        raise ValueError("fewer than 96 training rows in a category")
    selected = [group[math.floor((i + .5) * len(group) / 96)]
                for group in by_class for i in range(96)]
    if len({row["csv_line"] for row in selected}) != 288:
        raise ValueError("training selection was not unique")
    minima = [min(row["readings"][j] for row in training) for j in range(3)]
    maxima = [max(row["readings"][j] for row in training) for j in range(3)]
    if any(hi <= lo for lo, hi in zip(minima, maxima)):
        raise ValueError("zero or negative sensor scaling range")
    if sys.argv[1] == "preflight":
        print(json.dumps({"source_sha256": SOURCE_HASH,
                          "partition_class_counts": counts,
                          "selected_training": 288,
                          "heldout_rows": len(heldout), "sensor_ranges": list(zip(minima, maxima))},
                         indent=2))
        return
    if RESULT.exists():
        raise FileExistsError(f"write-once result already exists: {RESULT}")

    def scale(values):
        return tuple(max(0.0, min(1.0, (value - lo) / (hi - lo)))
                     for value, lo, hi in zip(values, minima, maxima))

    selected_x = [(row, scale(row["readings"])) for row in selected]
    confusion = [[0] * 3 for _ in range(3)]
    predictions = []
    for row in heldout:
        x = scale(row["readings"])
        neighbors = sorted(((sum((a - b) ** 2 for a, b in zip(x, train_x)), order, train_row)
                            for order, (train_row, train_x) in enumerate(selected_x)),
                           key=lambda item: (item[0], item[1]))[:5]
        votes = Counter(neighbor[2]["class"] for neighbor in neighbors)
        predicted = max(range(3), key=lambda label: (votes[label], -label))
        confusion[row["class"]][predicted] += 1
        predictions.append({"csv_line": row["csv_line"], "actual": row["class"],
                            "predicted": predicted, "scaled_sensors": x,
                            "neighbor_lines": [item[2]["csv_line"] for item in neighbors]})
    support = [sum(matrix_row) for matrix_row in confusion]
    recall = [confusion[label][label] / support[label] if support[label] else 0
              for label in range(3)]
    balanced = sum(recall) / 3
    gates = {"heldout_support": all(n >= 50 for n in support),
             "balanced_accuracy": balanced >= .70,
             "every_class_recall": all(value >= .60 for value in recall)}
    result = {"protocol_sha256": sha256(PROTOCOL.read_bytes()),
              "source_csv_sha256": csv_hash,
              "task": "one-room real-sensor three-category occupancy classification",
              "features": ["S5_CO2", "mean_S1_to_S4_Sound", "S6_PIR_plus_S7_PIR"],
              "classes": CLASS_NAMES, "training_partition": TRAIN_PARTITION,
              "heldout_partition": HELDOUT_PARTITION,
              "partition_class_counts": counts,
              "training_csv_lines": [row["csv_line"] for row in selected],
              "training_sensor_min": minima, "training_sensor_max": maxima,
              "classifier": "5-nearest-neighbor, squared Euclidean, fixed tie rules",
              "confusion_actual_by_predicted": confusion, "heldout_support": support,
              "recall": recall, "balanced_accuracy": balanced,
              "overall_accuracy": sum(confusion[i][i] for i in range(3)) / len(heldout),
              "balanced_constant_baseline": 1 / 3, "gates": gates,
              "pass": all(gates.values()), "predictions": predictions}
    with RESULT.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("confusion_actual_by_predicted", "heldout_support", "recall",
                       "balanced_accuracy", "overall_accuracy", "gates", "pass")}, indent=2))


if __name__ == "__main__":
    main()
