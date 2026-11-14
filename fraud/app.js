const $ = id => document.getElementById(id);
const state = { mode: 'fixture', account: '', rows: [], selected: null, model: 'independent', frontier: false };
const money = value => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
const pct = value => `${Math.round(value * 100)}%`;

async function getJson(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
  return data;
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function reviewKey(id) { return `synapse-fraud-review:${state.account}:${id}`; }
function reviewStatus(id) { return sessionStorage.getItem(reviewKey(id)) || 'Not reviewed'; }

function renderQueue() {
  const list = $('tx-list');
  list.replaceChildren();
  const name = state.model;
  const sorted = [...state.rows].sort((a, b) => b.scores[name] - a.scores[name]);
  for (const row of sorted) {
    const txn = row.transaction;
    const button = el('button', undefined, 'tx');
    button.type = 'button';
    button.classList.toggle('selected', state.selected === txn.id);
    button.append(el('strong', txn.merchant_name), el('span', money(txn.amount), 'amount'),
      el('small', `${txn.category} · ${txn.date.slice(0, 10)} · ${reviewStatus(txn.id)}`),
      el('span', row.bands[name], `risk ${row.bands[name]}`));
    button.addEventListener('click', () => { state.selected = txn.id; renderQueue(); renderDetail(); });
    list.append(button);
  }
  if (!sorted.length) list.append(el('p', 'No purchases were returned for this account.', 'muted'));
  for (const band of ['HIGH', 'REVIEW', 'LOW']) {
    $(band.toLowerCase() + '-count').textContent = String(sorted.filter(row => row.bands[name] === band).length);
  }
}

function detailMetric(container, label, value) {
  const box = el('div');
  box.append(el('span', label), el('strong', value));
  container.append(box);
}

function renderDetail() {
  const root = $('detail');
  root.replaceChildren();
  const row = state.rows.find(item => item.transaction.id === state.selected);
  if (!row) { root.append(el('h3', 'Transaction evidence'), el('p', 'Select a transaction.', 'detail-placeholder')); return; }
  const txn = row.transaction;
  const top = el('div', undefined, 'detail-head');
  top.append(el('h3', txn.merchant_name), el('span', row.bands[state.model], `risk ${row.bands[state.model]}`));
  root.append(top, el('div', money(txn.amount), 'money'),
    el('p', `${txn.category} · ${txn.date.slice(0, 10)} · ${txn.id}`, 'muted'));
  const metrics = el('div', undefined, 'detail-grid');
  detailMetric(metrics, 'Selected local model score', pct(row.scores[state.model]));
  detailMetric(metrics, 'Historical median purchase', money(row.context.historical_median));
  detailMetric(metrics, 'Prior purchases', String(row.context.history_purchases));
  detailMetric(metrics, 'Purchases in prior 24h', String(row.context.purchases_last_24h));
  root.append(metrics, el('h4', 'Why it was ranked here'));
  const list = el('ul', undefined, 'evidence');
  row.evidence.forEach(item => list.append(el('li', item)));
  root.append(list);
  const teacher = el('div', undefined, 'callout');
  teacher.append(el('strong', `Deterministic teacher fixture: ${row.teacher_fixture.risk}`),
    el('p', row.teacher_fixture.explanation),
    el('p', `Uncertainty: ${row.teacher_fixture.uncertainty}`));
  root.append(teacher);
  const comparison = el('p', `Local scores · independent ${pct(row.scores.independent)} · guided ${pct(row.scores.guided)} · shuffled ${pct(row.scores.shuffled)}`, 'muted');
  root.append(comparison);
  if (state.mode === 'fixture') root.append(el('p', `Synthetic scenario label: ${row.synthetic_label ? 'injected suspicious case' : 'benign case'}. This is not a verified bank fraud label.`, 'muted'));
  root.append(el('p', 'The score is an uncalibrated ranking value. Review the transaction with a person before any action.', 'muted'));
  const actions = el('div', undefined, 'actions');
  const review = el('button', 'Mark for review'); review.type = 'button';
  const dismiss = el('button', 'Dismiss alert'); dismiss.type = 'button';
  for (const [button, value] of [[review, 'Review queued'], [dismiss, 'Dismissed']]) {
    button.addEventListener('click', () => {
      sessionStorage.setItem(reviewKey(txn.id), value);
      $('workspace-status').textContent = `${txn.merchant_name}: ${value.toLowerCase()} in this browser session. No financial action was taken.`;
      renderQueue(); renderDetail();
    });
  }
  actions.append(review, dismiss);
  if (state.frontier) {
    const ask = el('button', 'Ask configured frontier teacher'); ask.type = 'button';
    ask.addEventListener('click', async () => {
      ask.disabled = true; ask.textContent = 'Waiting for teacher…';
      try {
        const result = await getJson('/api/fraud/frontier', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ account_id: state.account, transaction_id: txn.id }) });
        const box = el('div', undefined, 'callout');
        box.append(el('strong', `${result.source}: ${result.risk}`), el('p', result.explanation),
          el('p', `Indicators: ${result.indicators.join('; ') || 'none'}. Uncertainty: ${result.uncertainty || 'not supplied'}`));
        actions.after(box);
      } catch (error) { $('workspace-status').textContent = error.message; }
      finally { ask.disabled = false; ask.textContent = 'Ask configured frontier teacher'; }
    });
    actions.append(ask);
  }
  root.append(actions, el('p', reviewStatus(txn.id), 'status'));
}

async function scan() {
  state.account = $('account-select').value;
  if (!state.account) return;
  $('scan-btn').disabled = true;
  $('workspace-status').textContent = 'Retrieving transactions, building history context, scoring locally, and ranking review candidates…';
  try {
    const data = await getJson(`/api/fraud/account?id=${encodeURIComponent(state.account)}`);
    state.rows = data.transactions;
    state.selected = state.rows.length ? [...state.rows].sort((a, b) => b.scores[state.model] - a.scores[state.model])[0].transaction.id : null;
    renderQueue(); renderDetail();
    $('workspace-status').textContent = `Analyzed ${state.rows.length} ${data.mode === 'fixture' ? 'generated mock' : 'Nessie mock'} purchases. Teacher disconnected for local scoring.`;
  } catch (error) {
    $('workspace-status').textContent = error.message;
    state.rows = []; state.selected = null; renderQueue(); renderDetail();
  } finally { $('scan-btn').disabled = false; }
}

async function init() {
  try {
    const [status, accounts, research] = await Promise.all([
      getJson('/api/fraud/status'), getJson('/api/fraud/accounts'), getJson('/api/fraud/research')]);
    state.mode = status.mode; state.frontier = status.frontier_configured;
    if (status.frontier_student_trained) {
      const option = el('option', `OpenAI-guided local student (${status.frontier_student_messages} messages)`);
      option.value = 'frontier_guided';
      $('model-select').append(option);
    }
    $('source-mode').textContent = status.mode === 'fixture'
      ? 'Offline fixture mode · generated mock purchases' : 'Nessie mock-data mode · read-only';
    $('account-select').replaceChildren(...accounts.accounts.map(account => {
      const option = el('option', account.name); option.value = account.id; return option;
    }));
    const labels = { independent: 'Independent', guided: 'Teacher-guided', shuffled: 'Shuffled advice' };
    $('metric-body').replaceChildren(...['independent', 'guided', 'shuffled'].map(name => {
      const result = research.arms[name]; const tr = el('tr');
      tr.append(el('td', labels[name]),
        el('td', `${result.confusion.tp} / ${result.confusion.tp + result.confusion.fn}`),
        el('td', String(result.confusion.fp)), el('td', result.f1.toFixed(3)));
      return tr;
    }));
    $('gate-result').textContent = `Predefined gate: ${research.success_gate_passed ? 'passed' : 'failed'}.`;
    await scan();
  } catch (error) {
    $('workspace-status').textContent = `Could not start Synapse Fraud: ${error.message}. Run python -m fraud.server from the repository root.`;
    $('metric-body').replaceChildren(el('tr', `Could not load saved result: ${error.message}`));
  }
}

$('scan-btn').addEventListener('click', scan);
$('model-select').addEventListener('change', event => {
  state.model = event.target.value;
  if (state.rows.length) {
    state.selected = [...state.rows].sort((a, b) => b.scores[state.model] - a.scores[state.model])[0].transaction.id;
    renderQueue(); renderDetail();
    $('workspace-status').textContent = `Showing ${state.model} local student scores on the same loaded purchases.`;
  }
});
init();
