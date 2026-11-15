// Bank view tab: the locked real-data fraud study, and one run of the credit-union agent around it.
// Every number shown here is read from a saved result file; nothing is typed into the page.
const STUDY_URL = 'data/real_data_advice/fraud_v2_summary.json';
const RUN_URL = 'data/bank_demo/agent_run.json';

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter(child => child != null));
  return node;
};
const fixed = (value, digits = 3) => Number(value).toFixed(digits);
const signed = (value, digits = 3) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
const load = async url => {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
};

function renderStudy(study) {
  const body = document.getElementById('bank-study-body');
  if (!study?.meanAP) {
    body.replaceChildren(el('p', { className: 'legend', textContent: 'The study summary file is missing.' }));
    return;
  }
  const ap = study.meanAP;
  const [low, high] = study.mainTest.bootstrap95;
  const share = (ap.taught - ap.alone) / (ap.fullLabels - ap.alone);
  const arms = [
    ['Credit union alone', ap.alone],
    ['With the bank\'s advice', ap.taught, 'taught'],
    ['With every true label (ceiling)', ap.fullLabels],
    ['Out-of-date bank model', ap.stale],
    ['Random advice', ap.random],
    ['Shuffled advice', ap.shuffled],
  ];
  const recall = study.meanRecallTop1;
  body.replaceChildren(
    el('p', {}, el('strong', { textContent: study.mainTest.passed ? 'Passed. ' : 'Failed its pass rule, narrowly. ' }),
      `Advice of ${study.adviceBytes.toLocaleString()} bytes raised the credit union's average precision from ${fixed(ap.alone)} to ${fixed(ap.taught)}, ` +
      `${Math.round(share * 100)}% of the way to what every true label would give (${fixed(ap.fullLabels)}). ` +
      `The lead over the best control was ${signed(study.mainTest.meanAdvantage)} (95% CI ${signed(low)} to ${signed(high)}), ` +
      `and the rule fixed in advance needed the lower end above +0.02.`),
    el('ul', { className: 'bank-arms' }, ...arms.map(([name, value, cls]) =>
      el('li', { className: cls ?? '' }, el('span', { textContent: name }),
        Object.assign(el('b'), { style: `--v:${value}` }), el('em', { textContent: fixed(value) })))),
    el('p', { className: 'legend', textContent:
      `Mean average precision on held-out purchases over the ${study.seedsPassingTeacherGate} of ${study.seedsRun} runs where the bank's model passed its own check ` +
      `(the other ${study.seedsRun - study.seedsPassingTeacherGate} stopped there and are not replaced). Advice was worse than alone or no better in ${study.seedsAdvantageAtMostZero} of those runs. ` +
      `On a second measure the advice did not help: the share of frauds in the top 1% of scores was ${fixed(recall.taught)} with advice and ${fixed(recall.alone)} alone.` }),
  );
}

function renderRun(run) {
  if (!run?.steps) return;
  const source = run.ledgerSource === 'nessie' ? 'Capital One Nessie API' : 'offline ledger (Nessie was not used for this run)';
  const steps = run.steps.map(step => el('li', {},
    el('h3', { textContent: step.title }),
    el('p', { textContent: step.text }),
    step.checks && el('ul', { className: 'bank-checks' }, ...step.checks.map(check => el('li', {},
      el('span', { className: check.passed ? 'bank-pass' : 'bank-fail', textContent: check.passed ? 'Pass' : 'Fail' }),
      `${check.name}: ${check.value}`))),
    step.decision && el('p', { className: 'bank-decision', textContent: `→ ${step.decision}` }),
    step.staleTeacher && el('p', { className: step.staleTeacher.rejected ? 'legend' : 'bank-warn', textContent: step.staleTeacher.text }),
  ));
  const match = run.matchesLockedStudy;
  document.getElementById('bank-agent-body').replaceChildren(
    el('p', { className: 'legend', textContent: `Ledger: ${source}. Study seed ${run.seed}, chosen by rule: ${run.seedRule}.` }),
    el('ol', { className: 'bank-steps' }, ...steps),
    match === false ? el('p', { className: 'bank-warn', textContent: 'This run does not reproduce the saved study result for its seed, so do not present it.' }) : null,
  );

  const { ledger } = run;
  document.getElementById('bank-ledger').hidden = false;

  // Conclusion first: what each model would have caught on this ledger.
  const flagAt = run.settings.flagAt;
  const money = value => value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fraudAmount = run.purchases.filter(p => p.fraud).reduce((sum, p) => sum + p.amount, 0);
  const caughtAmount = arm => run.purchases.filter(p => p.fraud && p[arm] >= flagAt).reduce((sum, p) => sum + p.amount, 0);
  const a = ledger.alone, t = ledger.taught;
  const ap = run.testMetrics;
  const rows = [
    [`Frauds caught (of ${ledger.frauds})`, a.fraudsFlagged, t.fraudsFlagged, 'high'],
    ['Frauds missed', ledger.frauds - a.fraudsFlagged, ledger.frauds - t.fraudsFlagged, 'low'],
    ['False alarms', a.falseAlarms, t.falseAlarms, 'low'],
    [`Fraud amount stopped (of ${money(fraudAmount)})`, caughtAmount('alone'), caughtAmount('taught'), 'high', money],
    ['Average precision, all held-out purchases', ap.alone.ap, ap.taught.ap, 'high', v => fixed(v)],
  ];
  document.getElementById('bank-summary-rows').replaceChildren(...rows.map(([label, alone, taught, goal, show = String]) => {
    const better = goal === 'high' ? taught > alone : taught < alone;
    return el('tr', {}, el('th', { scope: 'row', textContent: label }),
      el('td', { textContent: show(alone) }),
      el('td', { className: better ? 'better' : '', textContent: show(taught) }));
  }));
  const extra = t.fraudsFlagged - a.fraudsFlagged, alarms = t.falseAlarms - a.falseAlarms;
  document.getElementById('bank-ledger-headline').textContent = extra > 0
    ? `With the bank's advice, the credit union caught ${extra} more of the ${ledger.frauds} frauds, ` +
      `${alarms > 0 ? `with ${alarms} more false alarm${alarms === 1 ? '' : 's'}` : 'with no extra false alarms'}, ` +
      `and stopped ${money(caughtAmount('taught'))} of fraud instead of ${money(caughtAmount('alone'))}.`
    : `On this ledger, the bank's advice did not help the credit union catch more fraud.`;

  document.getElementById('bank-ledger-note').textContent =
    `One run, seed ${run.seed}: ${ledger.purchases} held-out purchases from the ${source} that neither model trained on, flagged at a fraud score of ${flagAt}. ` +
    `Frauds are over-sampled for display (${ledger.frauds} here; the real rate is about 1 in 600), so the amounts show the direction, not real-world losses. ` +
    `The locked study above is the tested result.`;
  document.getElementById('bank-ledger-rows').replaceChildren(...run.purchases.map(p => el('tr', { className: p.fraud ? 'fraud' : '' },
    el('td', { textContent: String(p.id ?? p.row) }),
    el('td', { textContent: p.amount.toFixed(2) }),
    el('td', { textContent: fixed(p.alone) }),
    el('td', { textContent: fixed(p.taught) }),
    el('td', { className: p.flagged ? 'flag' : '', textContent: p.flagged ? 'Flag' : 'Allow' }),
    el('td', { textContent: p.fraud ? 'Fraud' : 'Legitimate' }),
  )));
}

renderStudy(await load(STUDY_URL));
renderRun(await load(RUN_URL));
