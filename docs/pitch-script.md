# Synapse: two-minute judge script

## Before judges arrive

From the repository root, run `python -m fraud.server` and open <http://localhost:8767/>. Keep **Simulation**, **Statistics**, and **Application** within reach. The 200-purchase banking page reads saved results and a saved Nessie sandbox ledger; it makes no live Nessie call. The footer's **Mock review queue** link opens the preserved six-purchase fraud prototype.

## Spoken script and clicks

**0:00-0:20 - Question.** (Simulation tab.) "When conditions change, can one AI help another adapt without handing over its data or weights? We tested that with two simulated fly brains."

**0:20-1:10 - Science.** (Press **Fast replay (4×)**. Open **Round-by-round details** in Step 3; use the timeline's Step 3 marker if needed.) "Fly A learns the new rule from its own wins and losses. It sends its preferred move as measured spikes on two simulated neurons. Fly B decodes those spikes and updates its own learner, using its own rewards too. Then we disconnect Fly A and test what Fly B retained."

**1:10-1:30 - Controlled result.** (Open **Statistics**.) "Across 15 fixed runs, taught Fly Bs averaged 68 wins out of 72 after 48 rounds, versus 33 with shuffled messages and 10 alone. The average lead over each run's strongest control was 24.9 wins. The gain was faster recovery; a plain digital message worked as well as spikes."

**1:30-1:58 - Banking application.** (Open **Application**. Point to the two comparison cards, the source label, and **Open all 200 saved purchases**.) "Bank A has a newer tree model; Bank B starts with an older logistic model. A flag-or-allow bit crosses between them, then Bank B retrains and scores on its own. On this one saved, over-sampled 200-purchase ledger, it flagged 8 of 20 frauds alone and 13 with advice, with zero false alarms in either case. The ledger is marked Nessie-sourced, but our current key cannot access its original account. Across the full locked study, average precision rose from 0.756 to 0.789, yet the preset success test failed."

**1:58-2:00 - Boundary.** "The bank model does not use the fly brain. Neither demo establishes production fraud accuracy."

## Keep these boundaries clear

- The Science study used simulated fly brains and an engineered spike-message channel. It does not demonstrate biological fly-to-fly teaching.
- The saved 200-purchase snapshot is marked as Nessie **mock** data, with transaction IDs and row references that match the saved agent run. The current credential lists only the separate six-purchase account; the original account returned HTTP 404, so its readback cannot be independently confirmed now. The 28 anonymized features and fraud labels stayed in the local public dataset; no bank or credit union participated.
- The ledger over-samples frauds for display. The agent's ask and trust rules were not tested against controls. The v2 15-seed study's mean average-precision lead was 0.033, but its 95% lower bound was 0.010, below the required 0.020, so the locked gate **failed**.
- The separate six-purchase mock review dashboard remains available through **Mock review queue**. Its local classifier and Gemini prototype are separate from this banking application.
