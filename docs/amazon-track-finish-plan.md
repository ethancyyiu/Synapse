# Synapse: HackPrinceton finish plan

Status checked November 14, 2026, about 3:45 p.m. Toronto time. This is a working checklist, not a claim that the Amazon prize is currently available.

## Decision in one sentence

Keep the scientific result. Build and pitch one practical, simulated fault-recovery workflow around it: after an alarm-to-response rule changes, a trained agent sends a small action recommendation to a new agent, which learns and is then tested alone.

**Target user:** a person supervising a fleet of machines with different controllers. **What exists today:** a browser research demo centered on Rock-Paper-Scissors, a live RPS Challenge, and saved 15-run fault-response results. **What is not established:** recovery on actual machines, sensor data, or a trustworthy way to reject stale advice.

## First: eligibility and track check (P0)

- [ ] Ask the organizer or sponsor desk whether **Best Practical AI Innovation by Amazon** is an active Fall 2026 opt-in prize and where to select it. Amazon is a sponsor, but this named prize is absent from the current public Devpost prize list. Do not rely on the Fall 2025 listing.
- [ ] Check the team's eligibility and check-in status. Select **Science & Discovery** as the one main track if that remains the best fit; add the Amazon sponsor prize only if organizers confirm it is active. Do not enter Best Hardware Hack without actual hardware.
- [ ] Audit provenance before submission. The repository was created during the event, but parts of the research and Fly Hero base predate it. Clearly identify pre-event work, open-source components, and code written during the event in the README and Devpost description. The published rule says at least **85% of submitted code** must have been written during the event. Do not claim the existing code became new because it was imported into a new repository.
- [ ] If this repository cannot meet that threshold, submit a small, genuinely new event-built application repository and link this research repository as credited background. Do not copy the old implementation into the new repository and count it as new code. Ask organizers how they treat a linked prior project or dependency before relying on that approach.
- [ ] Treat **Sunday, November 15, 8:00 a.m. ET** as the submission deadline. The rules call it a hard stop even though other Devpost pages show 8:30 a.m. Submit well before 8:00.

## Product slice to finish (P0)

Build one separate **Fault Recovery** view. Preserve the research page for detailed evidence.

- [ ] Opening screen states the user's problem in one sentence: a machine's alarm-response mapping changed and a replacement controller must recover quickly.
- [ ] A **Change conditions** control applies the existing three-alarm, three-response simulated task. Show the changed mapping to the audience, but keep it hidden from the agents if the experiment requires that.
- [ ] A **Run recovery** control runs the taught student and at least the alone and shuffled-message controls on matched schedules. Prefer live execution of the existing task; if time or compute prevents it, label saved runs as recorded and keep a separate live interaction visible. Never imply a replay is live.
- [ ] Show each step in plain language: alarm observed, local choice, reward, teacher advice, next choice. Show that the teacher sends a preference rather than weights or raw sensor history.
- [ ] A **Disconnect teacher / Test independently** control freezes learning and measures the student's own 72-trial result. Make the control comparison visible on the same screen.
- [ ] Include a plain-language result card: "In this simulated fault task, taught students led the strongest control by 19 of 72 correct responses at 48 rounds on average across 15 runs." Link the protocol and saved result. Do not present this as a real-machine result.

**Cut line:** if the fault task cannot be made reliable tonight, ship the existing RPS live Challenge with a short, honest fault-response evidence panel. A fragile new demo is worse than a working one. Do not present RPS as a deployed fault-response application.

## Time budget (Toronto / Eastern time)

- **Saturday, 4:00–5:00 p.m.:** confirm sponsor prize, decide the compliant submission repository, freeze the one-screen scope.
- **5:00–10:00 p.m.:** build the fault-recovery workflow and connect it to the actual simulator or clearly marked saved runs.
- **10:00 p.m.–midnight:** verify numbers, run checks, and rehearse on the judge laptop. Cut features that fail a fresh-load run.
- **Sunday, before 6:30 a.m.:** finish README, Devpost text, attribution, and two-minute pitch.
- **By 7:30 a.m.:** upload all code, submit, and test public links. Keep the remaining time for submission problems.

## Evidence and technical checks (P0)

- [ ] Run `npm test` and the relevant live-demo check after UI changes. Check the result text against `data/fault_response_cohort_summary.json`.
- [ ] Rehearse the exact path on the presentation laptop, served locally and offline: open page, change condition, run, disconnect, compare. Test after a fresh browser reload.
- [ ] Keep the caveats visible: synthetic three-cue task; speed of recovery, not higher final performance; digital advice works as well as spikes; a stale teacher can harm learning. Do not call the present system a reliable agentic planner.
- [ ] If adding a multi-step decision loop, show actual actions and tests in the UI. Treat trust calibration as experimental because its locked study missed its stated success bar.

## Pitch and submission (P0)

- [ ] Write a 2-minute script around one user and one demo path. Suggested timing: 20 seconds problem, 65 seconds workflow, 25 seconds measured comparison, 10 seconds limits.
- [ ] Prepare a 30-second fallback using the recorded demo and the 15-run chart in case live training is slow. Clearly label recordings.
- [ ] Update the README opening with: target user, what was built during the event, how to run, exact measured result, and limitations. Keep credits to Fly Hero, connectome data, fonts, and any other reused assets.
- [ ] Create the Devpost entry, add all team members and Discord usernames, select one main track and any confirmed sponsor opt-ins, link the public GitHub repository, and upload all code **before** the cutoff. An optional video must be at most 2 minutes.
- [ ] Submit by 7:30 a.m. ET at the latest. Reopen the public GitHub and Devpost links in a private browser window to check they work.

## If time remains (P1)

- [ ] Add a single simple schematic showing two different controllers, a small advice message, and independent student testing.
- [ ] Add a small "wrong teacher" scenario to demonstrate the limitation, but do not claim the current trust mechanism fully solves it.
- [ ] Ask one person unfamiliar with Synapse to use the demo without explanation; fix only the first point where they get stuck.

## Avoid tonight (P2)

- [ ] Do not start another broad real-data benchmark. The room-occupancy feasibility test failed its predefined floor, and the warehouse navigation attempt stopped at teacher feasibility.
- [ ] Do not bolt on a chatbot, AWS service, or hardware solely for sponsor optics. The working fault-recovery story and clear provenance matter more.
- [ ] Do not re-run large cohorts unless a concrete bug invalidates the saved evidence.

## Key references

- [Current event rules](https://hackprinceton-fall-26.devpost.com/rules)
- [Current event prizes and requirements](https://hackprinceton-fall-26.devpost.com/)
- [Fault-response protocol](fault-response-cohort-protocol.md) and [saved summary](../data/fault_response_cohort_summary.json)
- [Existing pitch script](pitch-script.md)
- [Real occupancy result](real-occupancy-report.md) and [warehouse result](warehouse-navigation-report.md)
