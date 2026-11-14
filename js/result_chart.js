// Hover/keyboard layer for the static 15-run result chart on the Stats tab.
// Data mirrors the "Show as a table" view; the SVG itself is unchanged.
const chart = document.querySelector('.result-chart');
const svg = chart && chart.querySelector('svg');
if (svg) {
  const NS = 'http://www.w3.org/2000/svg';
  const rounds = [0, 24, 48, 96], xs = [56, 188, 320, 584];
  const y = wins => 290 - wins * 250 / 72;
  // Order matches the <path> elements in the SVG.
  const series = [
    { name: 'Untrained teacher', wins: [0.7, 8.1, 22.6, 23.9],
      about: 'A never-trained Fly A. It always suggested one move, so it is a weak control.' },
    { name: 'Alone', wins: [0.7, 4.3, 10.1, 35.1],
      about: 'No messages. Fly B learns only from its own wins and losses.' },
    { name: 'Random messages', wins: [0.7, 11.8, 32.4, 51.3],
      about: 'A random move on the same rounds. Tests whether any signal helps.' },
    { name: 'Shuffled messages', wins: [0.7, 9.6, 32.6, 53],
      about: "Fly A's real suggestions, on the wrong rounds. Tests whether timing matters." },
    { name: 'Taught by Fly A', wins: [0.7, 51.3, 67.7, 69.9],
      about: 'The trained Fly A sends its real suggested move each round.' },
  ];
  const paths = [...svg.querySelectorAll('path')];
  const texts = [...svg.querySelectorAll('text')];
  series.forEach((s, i) => {
    s.path = paths[i];
    s.color = paths[i].getAttribute('stroke');
    s.dash = paths[i].getAttribute('stroke-dasharray') || '';
    s.label = texts.find(t => t.textContent === s.name);
  });
  // Native <title> tooltips would compete with the custom one.
  svg.querySelectorAll('circle[fill="transparent"]').forEach(c => c.remove());
  series[4].marks = [...svg.querySelectorAll('circle')];

  const layer = document.createElementNS(NS, 'g');
  layer.setAttribute('class', 'rc-hover');
  layer.style.display = 'none';
  const hair = document.createElementNS(NS, 'line');
  hair.setAttribute('y1', 34); hair.setAttribute('y2', 290);
  hair.setAttribute('stroke', '#ffffff'); hair.setAttribute('stroke-opacity', '.35');
  layer.append(hair);
  const dots = series.map(s => {
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('r', 5); c.setAttribute('fill', s.color);
    c.setAttribute('stroke', '#191918'); c.setAttribute('stroke-width', 2);
    layer.append(c); return c;
  });
  const hit = document.createElementNS(NS, 'rect');
  hit.setAttribute('x', 40); hit.setAttribute('y', 30);
  hit.setAttribute('width', 548); hit.setAttribute('height', 270);
  hit.setAttribute('fill', 'transparent');
  svg.append(layer, hit);

  const tip = document.createElement('div');
  tip.className = 'rc-tip'; tip.hidden = true;
  tip.setAttribute('role', 'status');
  chart.append(tip);

  const GREEN = '#4ade80';
  // Explain the touched line beside the chart on wide screens.
  const explain = document.createElement('p');
  explain.className = 'rc-explain';
  explain.setAttribute('role', 'status');
  chart.querySelector('.result-chart-side').prepend(explain);
  let focused = null, current = null;
  const emphasize = name => {
    const picked = series.find(s => s.name === name);
    explain.classList.toggle('on', !!picked);
    if (picked) {
      const strong = document.createElement('strong');
      strong.textContent = picked.name;
      explain.replaceChildren(strong, ` ${picked.about}`);
    } else explain.textContent = 'Touch a name to see what that line means.';
    series.forEach(s => {
      const dim = name && s.name !== name, on = s.name === name;
      // The touched line turns green; clearing the inline style restores its own colour.
      s.path.style.stroke = on ? GREEN : '';
      if (s.label) s.label.style.fill = on ? GREEN : '';
      (s.marks || []).forEach(m => { m.style.fill = on ? GREEN : ''; });
      s.path.style.opacity = dim ? '.18' : '';
      if (s.label) s.label.style.opacity = dim ? '.35' : '';
      (s.marks || []).forEach(m => { m.style.opacity = dim ? '.18' : ''; });
    });
    dots.forEach((d, i) => { d.style.opacity = name && series[i].name !== name ? '.2' : ''; });
    [...tip.querySelectorAll('.rc-row')].forEach(r => r.classList.toggle('dim', !!name && r.dataset.name !== name));
  };

  const show = k => {
    current = k;
    layer.style.display = '';
    hair.setAttribute('x1', xs[k]); hair.setAttribute('x2', xs[k]);
    series.forEach((s, i) => { dots[i].setAttribute('cx', xs[k]); dots[i].setAttribute('cy', y(s.wins[k])); });
    tip.replaceChildren();
    const head = document.createElement('p');
    head.className = 'rc-head';
    head.textContent = k === 0 ? 'At the rule change' : `${rounds[k]} rounds after the change${k === 2 ? ' (test point)' : ''}`;
    tip.append(head);
    [...series].sort((a, b) => b.wins[k] - a.wins[k]).forEach(s => {
      const row = document.createElement('div');
      row.className = 'rc-row'; row.dataset.name = s.name;
      const key = document.createElementNS(NS, 'svg');
      key.setAttribute('width', 18); key.setAttribute('height', 6); key.setAttribute('aria-hidden', 'true');
      const l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', 1); l.setAttribute('x2', 17); l.setAttribute('y1', 3); l.setAttribute('y2', 3);
      l.setAttribute('stroke', s.color); l.setAttribute('stroke-width', 2);
      if (s.dash) l.setAttribute('stroke-dasharray', s.dash);
      key.append(l);
      const v = document.createElement('strong');
      v.textContent = String(s.wins[k]);
      const n = document.createElement('span');
      n.textContent = s.name;
      row.append(key, v, n);
      tip.append(row);
    });
    tip.hidden = false;
    // Place beside the hairline, flipping left near the right edge.
    const box = svg.getBoundingClientRect(), fig = chart.getBoundingClientRect();
    const px = box.left - fig.left + xs[k] * box.width / 740;
    const left = px + 14 + tip.offsetWidth > fig.width ? px - 14 - tip.offsetWidth : px + 14;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${box.top - fig.top + 30 * box.height / 330}px`;
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

  svg.setAttribute('tabindex', '0');
  svg.addEventListener('focus', () => show(current ?? 2));
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      show(Math.min(3, Math.max(0, (current ?? 2) + (e.key === 'ArrowRight' ? 1 : -1))));
    } else if (e.key === 'Escape') hide();
  });

  // Legend: hover previews a line, click/Enter pins it.
  // The names beside the lines in the graph work the same way.
  const nameOf = li => li.textContent.trim();
  const items = [...chart.querySelectorAll('.chart-key li'), ...series.map(s => s.label).filter(Boolean)];
  const sync = () => items.forEach(li => li.setAttribute('aria-pressed', String(nameOf(li) === focused)));
  items.forEach(li => {
    const name = nameOf(li);
    li.tabIndex = 0; li.setAttribute('role', 'button');
    li.addEventListener('pointerenter', () => emphasize(name));
    li.addEventListener('pointerleave', () => emphasize(focused));
    li.addEventListener('focus', () => emphasize(name));
    li.addEventListener('blur', () => emphasize(focused));
    const toggle = () => { focused = focused === name ? null : name; sync(); emphasize(focused); };
    li.addEventListener('click', toggle);
    li.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  });
  sync(); emphasize(null);
  const hint = document.createElement('p');
  hint.className = 'legend rc-hint';
  hint.textContent = 'Hover or tab onto the chart to read every line at each checkpoint (← → to move). Touch a name to turn its line green and see what it means; click to keep it.';
  chart.append(hint);
}
