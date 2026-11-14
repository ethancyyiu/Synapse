# Real room-sensor occupancy: raw feasibility failed

The [protocol](real-occupancy-protocol.md) was fixed before evaluation. The [UCI Room Occupancy Estimation dataset](https://archive.ics.uci.edu/dataset/864/room%2Boccupancy%2Bestimation) contains measurements from one instrumented room. The three fixed classes were empty (0 people), partly occupied (1–2), and full (3). The reference received only CO2, mean sound, and summed PIR readings. It used 96 calibration examples per class and a fixed five-nearest-neighbor rule, then predicted every row in the held-out partition.

| Actual class | Predicted empty | Predicted partly occupied | Predicted full | Recall |
| --- | ---: | ---: | ---: | ---: |
| Empty, 703 rows | 659 | 42 | 2 | **93.7%** |
| Partly occupied, 93 rows | 10 | 3 | 80 | **3.2%** |
| Full, 201 rows | 12 | 164 | 25 | **12.4%** |

Balanced accuracy was **36.5%**, above the 33.3% balanced constant-class baseline but far below the locked **70%** floor. Partly occupied and full also missed the locked **60% per-class** floor. Overall accuracy was **68.9%**, dominated by empty-room rows, and is not evidence of three-class learnability. The held-out class-support gate passed. The joint raw-information gate **failed**.

The confusion demonstrates that these three readings and this fixed raw classifier did not reliably distinguish occupancy in the held-out partition. It does not isolate whether the cause was sensor drift, an insufficient feature set, a class boundary, or this room's occupancy schedule. The held-out partly occupied rows contain two people; one-person generalization was not tested. The measurements were gathered in one controlled room, with no HVAC action outcomes, so this is neither deployment evidence nor a recovery decision study.

**Stop decision:** No VPN/KC/MBON simulation, Fly A/B reward training, DN messages, or teaching comparison was run. The previous synthetic recovery failures and validated RPS results remain unchanged. A different real-sensor task or representation would need a separate prospective protocol; changing this run's features or threshold would invalidate its held-out result.

The [write-once artifact](../data/real_occupancy_raw_feasibility.json) preserves all 997 held-out predictions, confusion counts, source and protocol hashes, selected calibration rows, and gate outcomes. The local adapted CSV is derived from the UCI dataset under CC BY 4.0; its attribution is in the [protocol](real-occupancy-protocol.md). Reproduce in a copy **without** the result artifact with `python tools/check_real_occupancy_raw.py preflight` and `python tools/check_real_occupancy_raw.py run`; the runner refuses overwrite.
