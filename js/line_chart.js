// Interactive line chart in the same style as the 15-run result chart on the Stats tab:
// white main line with a green label, grey dashed controls, names at the line ends, gridlines,
// a hover/keyboard read-out of every line, and a legend that highlights a line and explains it.
//
// drawLineChart(container, {
//   title, ticks: ['0', '6', ...], xLabel, max: 72, yTicks: [0, 24, 48, 72],
//   series: [{ name, values: [...], about, main?: true }],
//   band?: { from, to, label },   // shaded span of checkpoint indices
//   bar?: { value, label },       // dashed horizontal threshold
//   unit: 'wins',
// })
const NS = 'http://www.w3.org/2000/svg';
const GREEN = '#4ade80';
const W = 740, H = 330, X0 = 56, X1 = 584, Y0 = 290, Y1 = 40;
// Control styles, in the order controls appear: colour and dash pattern.
const CONTROL_STYLES = [
  { color: '#8a929b', dash: '' },
  { color: '#6f7882', dash: '12 4' },
  { color: '#a3abb4', dash: '2 4' },
  { color: '#a3abb4', dash: '7 4' },
  { color: '#6f7882', dash: '4 3 1 3' },
];
const reduceMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function keySwatch(s) {
  const key = el('svg', { width: 30, height: 8, 'aria-hidden': 'true' });
  key.append(el('line', { x1: 1, x2: 29, y1: 4, y2: 4, stroke: s.color, 'stroke-width': s.main ? 3 : 2, ...(s.dash ? { 'stroke-dasharray': s.dash } : {}) }));
  return key;
}

export function drawLineChart(container, opts) {
  const { series: input, ticks, max, unit = 'wins' } = opts;
  const yTicks = opts.yTicks ?? [0, max / 3, (2 * max) / 3, max].map(v => Math.round(v));
  const n = ticks.length;
  const xs = ticks.map((_, i) => X0 + ((X1 - X0) * i) / Math.max(1, n - 1));
  const y = v => Y0 - ((Y0 - Y1) * v) / max;
  let control = 0;
  const series = input.map(s => ({ ...s, ...(s.main ? { color: '#ffffff', dash: '' } : CONTROL_STYLES[control++ % CONTROL_STYLES.length]) }));

  container.replaceChildren();
  container.classList.add('result-chart', 'line-chart');
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', tabindex: '0' });
  svg.append(el('title', {}, opts.title ?? ''));
  const desc = series.map(s => `${s.name} ${s.values.at(-1)}`).join(', ');
  svg.append(el('desc', {}, `Final ${unit} out of ${max}: ${desc}.`));

  if (opts.band) {
    const a = xs[opts.band.from], b = xs[opts.band.to];
    svg.append(el('rect', { x: a, y: Y1 - 6, width: b - a, height: Y0 - Y1 + 6, fill: '#ffffff', 'fill-opacity': '.035' }));
    svg.append(el('text', { x: (a + b) / 2, y: Y1 - 12, 'text-anchor': 'middle', fill: '#6b6960', 'font-size': 12 }, opts.band.label));
  }
  for (const v of yTicks) {
    svg.append(el('line', { x1: X0, x2: X1, y1: y(v), y2: y(v), stroke: '#2e2d2a', 'stroke-width': 1 }));
    svg.append(el('text', { x: X0 - 8, y: y(v) + 4, 'text-anchor': 'end', fill: '#6b6960', 'font-size': 13 }, String(v)));
  }
  ticks.forEach((t, i) => svg.append(el('text', { x: xs[i], y: Y0 + 20, 'text-anchor': 'middle', fill: '#6b6960', 'font-size': 13 }, t)));
  if (opts.xLabel) svg.append(el('text', { x: (X0 + X1) / 2, y: H - 3, 'text-anchor': 'middle', fill: '#6b6960', 'font-size': 13 }, opts.xLabel));
  if (opts.bar) {
    svg.append(el('line', { x1: X0, x2: X1, y1: y(opts.bar.value), y2: y(opts.bar.value), stroke: '#5f5d58', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
    svg.append(el('text', { x: X0 + 6, y: y(opts.bar.value) - 6, fill: '#a3abb4', 'font-size': 12 }, opts.bar.label));
  }

  // Controls first so the main line is drawn on top.
  const order = [...series.filter(s => !s.main), ...series.filter(s => s.main)];
  for (const s of order) {
    const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${xs[i].toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
    s.path = el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.main ? 3 : 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', ...(s.dash ? { 'stroke-dasharray': s.dash } : {}) });
    svg.append(s.path);
    if (s.main) s.marks = s.values.map((v, i) => { const c = el('circle', { cx: xs[i], cy: y(v), r: 4.5, fill: '#ffffff', stroke: '#191918', 'stroke-width': 2 }); svg.append(c); return c; });
  }

  // Names at the line ends, nudged apart so equal or close lines stay readable.
  const GAP = 17;
  const placed = [...series].sort((a, b) => y(a.values.at(-1)) - y(b.values.at(-1))).map(s => ({ s, y: y(s.values.at(-1)) + 4 }));
  for (let i = 1; i < placed.length; i++) placed[i].y = Math.max(placed[i].y, placed[i - 1].y + GAP);
  const overflow = placed.at(-1).y - (Y0 + 4);
  if (overflow > 0) placed.forEach(p => { p.y -= overflow; });
  for (let i = placed.length - 2; i >= 0; i--) placed[i].y = Math.min(placed[i].y, placed[i + 1].y - GAP);
  for (const { s, y: ly } of placed) {
    s.label = el('text', { x: X1 + 10, y: ly.toFixed(1), fill: s.main ? GREEN : '#8a929b', 'font-size': s.main ? 15 : 14, ...(s.main ? { 'font-weight': 700 } : {}) }, s.name);
    svg.append(s.label);
  }

  // Hover layer.
  const layer = el('g', { class: 'rc-hover' });
  layer.style.display = 'none';
  const hair = el('line', { y1: Y1 - 6, y2: Y0, stroke: '#ffffff', 'stroke-opacity': '.35' });
  layer.append(hair);
  const dots = series.map(s => { const c = el('circle', { r: 5, fill: s.color, stroke: '#191918', 'stroke-width': 2 }); layer.append(c); return c; });
  const hit = el('rect', { x: X0 - 16, y: Y1 - 10, width: X1 - X0 + 32, height: Y0 - Y1 + 20, fill: 'transparent' });
  svg.append(layer, hit);

  const side = document.createElement('div');
  side.className = 'result-chart-side';
  const explain = document.createElement('p');
  explain.className = 'rc-explain';
  explain.setAttribute('role', 'status');
  const key = document.createElement('ul');
  key.className = 'chart-key';
  for (const s of series) {
    const li = document.createElement('li');
    li.append(keySwatch(s), s.name);
    li.dataset.name = s.name;
    key.append(li);
  }
  side.append(explain, key);
  const tip = document.createElement('div');
  tip.className = 'rc-tip'; tip.hidden = true;
  tip.setAttribute('role', 'status');
  const hint = document.createElement('p');
  hint.className = 'legend rc-hint';
  hint.textContent = 'Hover or tab onto the chart to read every line at each checkpoint (← → to move). Touch a name to turn its line green and see what it means; click to keep it.';
  container.append(svg, side, tip, hint);

  let focused = null, current = null;
  const emphasize = name => {
    const picked = series.find(s => s.name === name);
    explain.classList.toggle('on', !!picked);
    if (picked) {
      const strong = document.createElement('strong');
      strong.textContent = picked.name;
      explain.replaceChildren(strong, ` ${picked.about ?? ''}`);
    } else explain.textContent = 'Touch a name to see what that line means.';
    for (const s of series) {
      const on = s.name === name, dim = !!name && !on;
      s.path.style.stroke = on ? GREEN : '';
      s.path.style.opacity = dim ? '.15' : '';
      s.label.style.fill = on ? GREEN : '';
      s.label.style.opacity = dim ? '.35' : '';
      (s.marks || []).forEach(m => { m.style.fill = on ? GREEN : ''; m.style.opacity = dim ? '.15' : ''; });
      if (on) s.path.parentNode.insertBefore(s.path, s.marks?.[0] ?? s.label); // bring to front
    }
    dots.forEach((d, i) => { d.style.opacity = name && series[i].name !== name ? '.2' : ''; });
    tip.querySelectorAll('.rc-row').forEach(r => r.classList.toggle('dim', !!name && r.dataset.name !== name));
  };

  const show = k => {
    current = k;
    layer.style.display = '';
    hair.setAttribute('x1', xs[k]); hair.setAttribute('x2', xs[k]);
    series.forEach((s, i) => { dots[i].setAttribute('cx', xs[k]); dots[i].setAttribute('cy', y(s.values[k])); });
    const head = document.createElement('p');
    head.className = 'rc-head';
    head.textContent = opts.tickTitle ? opts.tickTitle(k) : ticks[k];
    tip.replaceChildren(head);
    [...series].sort((a, b) => b.values[k] - a.values[k]).forEach(s => {
      const row = document.createElement('div');
      row.className = 'rc-row'; row.dataset.name = s.name;
      const v = document.createElement('strong');
      v.textContent = String(s.values[k]);
      const label = document.createElement('span');
      label.textContent = s.name;
      const sw = keySwatch(s); sw.setAttribute('width', 18);
      sw.firstChild.setAttribute('x2', 17);
      row.append(sw, v, label);
      tip.append(row);
    });
    tip.hidden = false;
    const box = svg.getBoundingClientRect(), fig = container.getBoundingClientRect();
    const px = box.left - fig.left + (xs[k] * box.width) / W;
    const left = px + 14 + tip.offsetWidth > fig.width ? px - 14 - tip.offsetWidth : px + 14;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${box.top - fig.top + (Y1 * box.height) / H}px`;
    emphasize(focused);
  };
  const hide = () => { current = null; layer.style.display = 'none'; tip.hidden = true; };

  hit.addEventListener('pointermove', e => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const sx = p.matrixTransform(svg.getScreenCTM().inverse()).x;
    let k = 0;
    xs.forEach((x, i) => { if (Math.abs(x - sx) < Math.abs(xs[k] - sx)) k = i; });
    if (k !== current) show(k);
  });
  hit.addEventListener('pointerleave', hide);
  svg.addEventListener('focus', () => show(current ?? n - 1));
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      show(Math.min(n - 1, Math.max(0, (current ?? n - 1) + (e.key === 'ArrowRight' ? 1 : -1))));
    } else if (e.key === 'Escape') hide();
  });

  // Legend items and line-end names: hover previews a line, click/Enter keeps it.
  const items = [...key.children, ...series.map(s => s.label)];
  const nameOf = item => item.dataset?.name ?? item.textContent;
  const sync = () => items.forEach(item => item.setAttribute('aria-pressed', String(nameOf(item) === focused)));
  for (const item of items) {
    const name = nameOf(item);
    item.setAttribute('tabindex', '0'); item.setAttribute('role', 'button');
    item.addEventListener('pointerenter', () => emphasize(name));
    item.addEventListener('pointerleave', () => emphasize(focused));
    item.addEventListener('focus', () => emphasize(name));
    item.addEventListener('blur', () => emphasize(focused));
    const toggle = () => { focused = focused === name ? null : name; sync(); emphasize(focused); };
    item.addEventListener('click', toggle);
    item.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  }
  sync(); emphasize(null);

  // Draw the lines in, controls first, then the main line.
  if (!reduceMotion()) {
    order.forEach((s, i) => {
      const len = s.path.getTotalLength();
      const dash = s.dash;
      s.path.style.strokeDasharray = `${len}`;
      s.path.style.strokeDashoffset = `${len}`;
      const anim = s.path.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: 700, delay: 80 * i, easing: 'ease-out', fill: 'forwards' });
      anim.onfinish = () => { anim.cancel(); s.path.style.strokeDasharray = dash || ''; s.path.style.strokeDashoffset = ''; };
    });
  }
}
