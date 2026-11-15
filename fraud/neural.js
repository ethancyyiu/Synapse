// Fraud-response study UI. Full teaching is a recorded, measured simulation;
// the selected-purchase check runs a frozen fly in a browser worker.
const $ = id => document.getElementById(id);
const ACTIONS = ['BLOCK', 'APPROVE', 'REVIEW'];
let artifact, selected, traceIndex = -1, worker, requestId = 0;

function cell(text) { const node = document.createElement('td'); node.textContent = String(text); return node; }

export function selectNeuralTransaction(row, source) {
  selected = row ? { row, source } : null;
  $('neural-probe').disabled = !selected || !artifact;
  $('neural-probe-result').textContent = '';
  if (!selected) {
    $('neural-purchase').textContent = 'Select an account and transaction above.';
    $('neural-cue').textContent = 'Waiting for a purchase cue.';
    return;
  }
  const { transaction: tx, neural_cue: cue } = row;
  $('neural-purchase').textContent = `${source === 'nessie' ? 'Live Nessie mock' : 'Offline fixture'} purchase: ${tx.merchant_name}, $${Number(tx.amount).toFixed(2)}. Local NumPy assessment: ${row.bands[$('model-select').value]}.`;
  $('neural-cue').textContent = `Cue ${cue.index} · ${cue.name}. ${cue.basis}`;
}

function showTrace() {
  traceIndex = (traceIndex + 1) % artifact.teachingTrace.length;
  const t = artifact.teachingTrace[traceIndex];
  $('neural-trace').textContent = `Recorded message ${t.round}/96, cue ${t.cue}: Fly A preferred ${ACTIONS[t.sentAction]}; DNge175 spike counts [${t.dnCounts.join(', ')}]; Fly B decoded ${ACTIONS[t.decodedAction]} and made an imitation update (${t.imitationUpdate}). Fly B's own training action was ${ACTIONS[t.flyBAction]}.`;
}

function renderSummary(data) {
  const body = $('neural-results');
  body.replaceChildren();
  for (const seed of data.seeds) {
    const tr = document.createElement('tr');
    const p = seed.points?.find(x => x.rounds === 48);
    tr.append(cell(seed.seed), cell(p?.alone ?? '—'), cell(p?.trained ?? '—'),
      cell(p?.random ?? '—'), cell(p?.shuffled ?? '—'), cell(seed.advantage48 ?? '—'));
    body.append(tr);
  }
  $('neural-summary').textContent = data.successGatePassed
    ? `Predeclared synthetic teaching gate: PASSED. Mean 48-round lead ${data.meanAdvantage48.toFixed(1)}/72 over each seed's strongest control; ${data.positiveAdvantageSeeds}/${data.seeds.length} seeds positive. ${data.totalDecoded}/${data.totalMessages} DN messages decoded. Recovery times and all five seeds are in the summary.`
    : `Predeclared synthetic teaching gate: FAILED. Mean 48-round lead ${data.meanAdvantage48?.toFixed(1) ?? 'unavailable'}/72; ${data.positiveAdvantageSeeds}/${data.seeds.length} seeds positive. Full outcomes are in the summary.`;
  $('neural-recovery').textContent = data.seeds.map(s => {
    if (!s.recoveryAt) return `Seed ${s.seed}: teacher gate not reached`;
    const time = arm => s.recoveryAt[arm] === null ? '>96' : String(s.recoveryAt[arm]);
    return `Seed ${s.seed} recovery rounds — trained ${time('trained')}, alone ${time('alone')}, random ${time('random')}, shuffled ${time('shuffled')}`;
  }).join(' · ');
}

export async function initNeural() {
  try {
    const [run, summary] = await Promise.all([
      fetch('data/fraud_neural/seed_401.json').then(r => { if (!r.ok) throw Error('Focused neural run unavailable'); return r.json(); }),
      fetch('data/fraud_neural/summary.json').then(r => { if (!r.ok) throw Error('Final neural summary unavailable'); return r.json(); }),
    ]);
    if (run.status !== 'completed' || !run.teacherFrozen || !run.synapsesUnchanged)
      throw Error('Saved neural run did not pass frozen validation');
    artifact = run;
    $('neural-teacher').textContent = `Focused recorded run, seed ${run.seed}: Fly A reached ${run.teacherWins}/72 on its frozen updated-rule check (gate ${run.teacherGate}/72). Its 12 training episodes, 24 trials each, scored ${run.teacherEpisodeWins.join(', ')} correct. Fly B then received ${run.decoded.trained}/${run.messages.trained} action messages through measured DN spikes.`;
    $('neural-next').disabled = false;
    $('neural-probe').disabled = !selected;
    renderSummary(summary);
  } catch (error) {
    $('neural-teacher').textContent = `Neural study unavailable: ${error.message}`;
    $('neural-summary').textContent = `Saved comparison unavailable: ${error.message}`;
  }
}

$('neural-next').addEventListener('click', showTrace);
$('neural-probe').addEventListener('click', () => {
  if (!artifact || !selected) return;
  const cue = selected.row.neural_cue.index;
  $('neural-probe').disabled = true;
  $('neural-probe-result').textContent = `Running frozen Fly B on cue ${cue} with Fly A disconnected…`;
  if (!worker) worker = new Worker(new URL('./neural_worker.js', import.meta.url), { type: 'module' });
  const id = ++requestId;
  worker.onmessage = ({ data }) => {
    if (data.requestId !== id) return;
    $('neural-probe').disabled = !selected;
    $('neural-probe-result').textContent = data.error ? `Probe failed: ${data.error}` :
      `Frozen Fly B chose ${data.action} for cue ${data.cue}. ${data.activeKcs} Kenyon cells were active in its ${data.neurons}-neuron simulation. Teacher disconnected; weights unchanged. This is a synthetic response policy, not a fraud verdict.`;
  };
  worker.onerror = event => {
    $('neural-probe').disabled = !selected;
    $('neural-probe-result').textContent = `Probe failed: ${event.message || 'worker error'}`;
  };
  worker.postMessage({ cue, requestId: id });
});
