// Charts and the measured comparison table used by the RPS page.
const $ = id => document.getElementById(id);
const pct = value => `${(value * 100).toFixed(1)}%`;

export function drawChart(canvas, lines, xmax, colors) {
  const context = canvas.getContext('2d');
  const { width, height } = canvas;
  const left = 46, right = 15, top = 12, bottom = 32;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const y = value => top + (1 - value) * plotHeight;

  context.clearRect(0, 0, width, height);
  context.fillStyle = colors.paper;
  context.fillRect(0, 0, width, height);
  context.strokeStyle = '#2e2d2a';
  context.lineWidth = 1;
  for (const value of [0, 1 / 3, 0.6, 1]) {
    context.beginPath();
    context.moveTo(left, y(value));
    context.lineTo(width - right, y(value));
    context.stroke();
    context.fillStyle = colors.graphite;
    context.font = '13px Atkinson, sans-serif';
    context.textAlign = 'right';
    context.fillText(`${Math.round(value * 100)}%`, left - 5, y(value) + 4);
  }
  context.textAlign = 'left';
  context.fillText('episode', left, height - 8);
  context.textAlign = 'right';
  context.fillText(String(xmax), width - right, height - 8);

  for (const { values, color } of lines) {
    if (!values.length) continue;
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    values.forEach((value, index) => {
      const x = left + ((index + 1) / xmax) * plotWidth;
      if (index) context.lineTo(x, y(value));
      else context.moveTo(x, y(value));
    });
    context.stroke();
  }
}

export function showResults(results, colors) {
  if (!results) {
    $('experiment-intro').textContent = 'No measured experiment file. Run node tools/run_rps_experiments.mjs.';
    return;
  }

  const meta = results.meta, seedCount = meta.seeds.length;
  $('experiment-intro').textContent = `${seedCount} seeds · ${meta.episodes} episodes per student · ${meta.roundsPerEpisode} rounds per episode · ${meta.opponent.noise * 100}% opponent noise. Teacher held-out mean: ${pct(results.teacher.heldoutMean)}.`;
  const alone = results.conditions.alone.summary;
  const trained = results.conditions.trained.summary;
  const gap = trained.heldoutMean - alone.heldoutMean;
  $('experiment-conclusion').textContent = gap <= 0
    ? `No teaching benefit detected: trained-signal and alone students finished at ${pct(trained.heldoutMean)} and ${pct(alone.heldoutMean)} on disconnected held-out play.`
    : `Trained-signal students finished ${pct(gap)} above alone students on disconnected held-out play. This ${seedCount}-seed comparison is exploratory.`;

  const teacherMean = Array.from({ length: meta.episodes }, (_, episode) =>
    results.teacher.runs.reduce((sum, run) => sum + run.curve[episode], 0) / seedCount);
  drawChart($('teacher-chart'), [{ values: teacherMean, color: colors.gfp }], meta.episodes, colors);

  const entries = [
    ['alone', 'Alone', '#e6e8eb'],
    ['trained', 'Trained teacher', '#5cc17a'],
    ['untrained', 'Untrained teacher', '#5b9bff'],
    ['shuffled_signal', 'Shuffled teacher signal', '#e466b8'],
    ['shuffled_wiring_alone', 'Shuffled wiring, alone', '#8d95a0'],
  ];
  const lines = entries.map(([key, , color]) =>
    ({ values: results.conditions[key].summary.curveMean, color }));
  drawChart($('experiment-chart'), lines, meta.episodes, colors);

  const table = document.createElement('table');
  const caption = document.createElement('caption');
  caption.textContent = 'Actual measurements. Colored chart lines follow the rows below.';
  table.append(caption);
  const head = table.createTHead().insertRow();
  for (const label of ['Condition', 'Last 3 episodes', 'Held out, disconnected', 'Reached 60%', 'Rounds to 60% by seed']) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    head.append(cell);
  }
  const body = table.createTBody();
  for (const [key, label, color] of entries) {
    const summary = results.conditions[key].summary;
    const row = body.insertRow();
    const name = row.insertCell();
    name.textContent = label;
    name.style.borderLeft = `5px solid ${color}`;
    row.insertCell().textContent = `${pct(summary.last3Mean)} ± ${pct(summary.last3Sd)}`;
    row.insertCell().textContent = `${pct(summary.heldoutMean)} ± ${pct(summary.heldoutSd)}`;
    row.insertCell().textContent = `${summary.reachedThreshold}/${seedCount} seeds`;
    row.insertCell().textContent = summary.roundsToThreshold.map(value => value ?? '—').join(', ');
  }
  $('experiment-table').replaceChildren(table);
}
