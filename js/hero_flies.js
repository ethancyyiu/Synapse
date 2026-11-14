// Decorative pair of line-drawn flies in the hero's empty space.
// Fly A (larger, steady) periodically sends a two-dot "message" to Fly B
// (smaller, erratic): a nod to the two message neurons in the experiment.
// Purely visual; aria-hidden. Sits still under prefers-reduced-motion.
import { BRAIN_POINTS, BRAIN_ASPECT } from './hero_brain_points.js';

const canvas = document.querySelector('.hero-flies');
const hero = canvas && canvas.parentElement;
if (canvas && hero) {
  const ctx = canvas.getContext('2d');
  const css = getComputedStyle(document.documentElement);
  const INK = css.getPropertyValue('--ink').trim() || '#edebe6';
  const GREY = css.getPropertyValue('--graphite').trim() || '#a9a69f';
  const ACCENT = css.getPropertyValue('--gfp').trim() || '#5cc17a';
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  let W = 0, H = 0, region = null;

  const fly = (size, speed, jitter, seed) => ({
    x: 0, y: 0, a: seed, size, speed, jitter, seed,
    rest: 0, flee: 0, fx: 0, fy: 0, glint: 0, trail: [], lastDot: 0, dead: null, spin: 0,
  });
  const A = fly(42, 38, 0.6, 1.3), B = fly(31, 62, 1.8, 4.1);
  const flies = [A, B];
  let message = null, nextMessage = 2500, pointer = null, poofs = [];

  const measure = () => {
    const dpr = window.devicePixelRatio || 1, box = hero.getBoundingClientRect();
    W = box.width; H = box.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Empty area: right of the text column, above the stat row.
    const text = [...hero.querySelectorAll('h1, .hook, .hook-why, .kicker')];
    const right = Math.max(...text.map(el => {
      const r = el.getBoundingClientRect(), range = document.createRange();
      range.selectNodeContents(el);
      return Math.min(r.right, range.getBoundingClientRect().right) - box.left;
    }));
    const stats = hero.querySelector('.hero-stats');
    const bottom = stats ? stats.getBoundingClientRect().top - box.top - 12 : H - 12;
    const x0 = right + 48, x1 = W - 24, y0 = 16, y1 = bottom;
    region = x1 - x0 > 180 && y1 - y0 > 120 ? { x0, x1, y0, y1 } : null;
    if (region) flies.forEach((f, i) => {
      if (!f.x || f.x < x0 || f.x > x1 || f.y > y1) {
        f.x = x0 + (x1 - x0) * (i ? 0.72 : 0.3); f.y = y0 + (y1 - y0) * (i ? 0.62 : 0.4);
      }
    });
  };

  const drawFly = (f, t) => {
    const s = f.size / 26, o = f.dead === null ? 1 : Math.max(0, 1 - f.dead / 800);
    ctx.save();
    ctx.translate(f.x, f.y); ctx.rotate(f.a + Math.PI / 2); ctx.scale(s, s);
    ctx.lineWidth = 1.6 / s; ctx.strokeStyle = ACCENT; ctx.lineCap = 'round';
    // Wings: translucent white, a slight flutter while flying.
    const flap = f.rest > 0 || f.dead !== null || still.matches ? 0 : Math.sin(t / 22 + f.seed) * 0.18;
    for (const side of [-1, 1]) {
      ctx.save();
      ctx.rotate(side * (0.5 + flap));
      ctx.beginPath(); ctx.ellipse(side * 5, 2, 4.2, 8.5, side * 0.25, 0, Math.PI * 2);
      ctx.fillStyle = INK; ctx.globalAlpha = o * (0.35); ctx.fill();
      // A received message briefly flashes the wings green.
      if (f.glint > 0) { ctx.fillStyle = ACCENT; ctx.globalAlpha = o * (f.glint * 0.7); ctx.fill(); }
      ctx.strokeStyle = INK; ctx.globalAlpha = o * (0.8); ctx.stroke();
      ctx.restore();
    }
    ctx.strokeStyle = ACCENT;
    ctx.globalAlpha = o * (1);
    // Legs.
    ctx.beginPath();
    for (const side of [-1, 1]) for (const k of [-1, 0, 1]) {
      ctx.moveTo(side * 2.2, 1 + k * 2.6); ctx.lineTo(side * 6, 1 + k * 3.6 + 1.5);
    }
    ctx.globalAlpha = o * (0.7); ctx.stroke(); ctx.globalAlpha = o * (1);
    // Body and head.
    ctx.globalAlpha = o;
    ctx.strokeStyle = INK;
    ctx.fillStyle = canvas.dataset.bg || '#121212';
    ctx.beginPath(); ctx.ellipse(0, 2.5, 3, 6.2, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, -5.6, 2.6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = INK;
    if (f.dead === null) for (const side of [-1, 1]) { ctx.beginPath(); ctx.arc(side * 1.3, -6.4, 0.75, 0, Math.PI * 2); ctx.fill(); }
    else { // x-ed out eyes
      ctx.lineWidth = 0.9 / s; ctx.beginPath();
      for (const side of [-1, 1]) { const ex = side * 1.3; ctx.moveTo(ex - 0.8, -7.2); ctx.lineTo(ex + 0.8, -5.6); ctx.moveTo(ex + 0.8, -7.2); ctx.lineTo(ex - 0.8, -5.6); }
      ctx.stroke();
    }
    ctx.restore();
  };

  // The fly becomes its brain: a sample of real soma positions from the
  // connectome (see tools/make_hero_brain_sprite.mjs), a few twinkling green.
  const raw = atob(BRAIN_POINTS), brainPts = [];
  for (let i = 0; i < raw.length; i += 2)
    brainPts.push([raw.charCodeAt(i) / 255 - 0.5, (raw.charCodeAt(i + 1) / 255 - 0.5) * BRAIN_ASPECT, i % 17 === 0]);
  const drawBrain = (f, t) => {
    const d = f.dead, alpha = Math.min(1, Math.max(0, (d - 300) / 400)) * Math.min(1, Math.max(0, (3300 - d) / 700));
    if (alpha <= 0) return;
    const grow = 1 - (1 - Math.min(1, Math.max(0, (d - 300) / 700))) ** 3; // ease-out spread from the fly
    const w = f.size * 2.2 * grow, rise = Math.max(0, d - 300) * 0.012;
    const cx = f.bx, cy = f.by - rise;
    ctx.fillStyle = INK; ctx.globalAlpha = alpha * 0.75;
    for (const [x, y, lit] of brainPts) if (!lit) ctx.fillRect(cx + x * w - 0.6, cy + y * w - 0.6, 1.2, 1.2);
    ctx.fillStyle = ACCENT;
    brainPts.forEach(([x, y, lit], i) => {
      if (!lit) return;
      ctx.globalAlpha = alpha * (0.3 + 0.7 * Math.max(0, Math.sin(t / 160 + i * 0.37 + f.seed)));
      ctx.fillRect(cx + x * w - 0.9, cy + y * w - 0.9, 1.8, 1.8);
    });
    ctx.globalAlpha = 1;
  };

  const dots = (x0, y0, x1, y1, color, alpha) => {
    const len = Math.hypot(x1 - x0, y1 - y0), n = Math.floor(len / 7);
    ctx.fillStyle = color; ctx.globalAlpha = alpha;
    for (let i = 1; i < n; i++) ctx.fillRect(x0 + (x1 - x0) * i / n - 0.5, y0 + (y1 - y0) * i / n - 0.5, 1, 1);
    ctx.globalAlpha = 1;
  };

  const step = (f, dt, t) => {
    const r = region;
    if (f.dead !== null) {
      f.dead += dt;
      if (f.dead < 800) {
        f.a += f.spin * dt / 1000;
        f.y = Math.min(r.y1 + 20, f.y + (30 + f.dead * 0.08) * dt / 1000);
        f.bx = f.x; f.by = f.y;
      }
      if (f.dead > 3500) { // a new fly buzzes in from the right edge
        f.dead = null; f.trail = []; f.x = r.x1; f.y = r.y0 + (r.y1 - r.y0) * Math.random(); f.a = Math.PI;
      }
      return;
    }
    if (f.rest > 0) { f.rest -= dt; return; }
    if (pointer && Math.hypot(pointer.x - f.x, pointer.y - f.y) < 60 && f.flee <= 0) {
      f.flee = 650; f.a = Math.atan2(f.y - pointer.y, f.x - pointer.x);
    }
    const fleeing = f.flee > 0;
    if (fleeing) f.flee -= dt;
    // Smooth wander: a sum of slow sines, plus a pull back to the region's centre near edges.
    let turn = (Math.sin(t / 900 + f.seed) + Math.sin(t / 370 + f.seed * 2.7) * f.jitter * 0.5) * 1.1;
    const cx = (r.x0 + r.x1) / 2, cy = (r.y0 + r.y1) / 2;
    const edge = Math.min(f.x - r.x0, r.x1 - f.x, f.y - r.y0, r.y1 - f.y);
    if (edge < 60 && !fleeing) {
      let d = Math.atan2(cy - f.y, cx - f.x) - f.a;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      turn += d * 4;
    }
    f.a += turn * dt / 1000;
    const v = f.speed * (fleeing ? 2.6 : 1) * dt / 1000;
    f.x = Math.min(r.x1, Math.max(r.x0, f.x + Math.cos(f.a) * v));
    f.y = Math.min(r.y1, Math.max(r.y0, f.y + Math.sin(f.a) * v));
    if (t - f.lastDot > 45) { f.trail.push({ x: f.x, y: f.y, t }); f.lastDot = t; }
    // Occasionally land for a moment (the steady fly more often).
    if (!fleeing && Math.random() < dt / (f === A ? 9000 : 16000)) f.rest = 1800 + Math.random() * 1600;
  };

  const drawStill = () => {
    ctx.clearRect(0, 0, W, H);
    if (!region) return;
    const r = region;
    A.x = r.x0 + (r.x1 - r.x0) * 0.32; A.y = r.y0 + (r.y1 - r.y0) * 0.42; A.a = -0.3;
    B.x = r.x0 + (r.x1 - r.x0) * 0.7; B.y = r.y0 + (r.y1 - r.y0) * 0.6; B.a = 2.6;
    dots(A.x + 26, A.y + 6, B.x - 20, B.y - 3, GREY, 0.6);
    flies.forEach(f => drawFly(f, 0));
  };

  let last = 0, raf = 0, visible = true;
  const frame = t => {
    raf = 0;
    const dt = Math.min(50, t - (last || t)); last = t;
    ctx.clearRect(0, 0, W, H);
    if (region) {
      flies.forEach(f => step(f, dt, t));
      // Faint dotted trails that fade over ~2s.
      ctx.fillStyle = GREY;
      flies.forEach(f => {
        f.trail = f.trail.filter(p => t - p.t < 2000);
        f.trail.forEach(p => { ctx.globalAlpha = 0.55 * (1 - (t - p.t) / 2000); ctx.fillRect(p.x - 0.5, p.y - 0.5, 1, 1); });
      });
      ctx.globalAlpha = 1;
      // Fly A -> Fly B message: two accent dots along a dotted line.
      nextMessage -= dt;
      if (!message && nextMessage <= 0 && A.dead === null && B.dead === null) { message = { t0: t }; nextMessage = 5200 + Math.random() * 2000; }
      if (message) {
        const k = (t - message.t0) / 1300;
        dots(A.x, A.y, B.x, B.y, GREY, 0.35 * Math.min(1, 3 * (1 - k)));
        if (A.dead !== null || B.dead !== null) message = null;
        else if (k >= 1) { message = null; B.glint = 1; }
        else for (const lag of [0, 0.09]) {
          const q = Math.max(0, k - lag);
          ctx.fillStyle = ACCENT; ctx.beginPath();
          ctx.arc(A.x + (B.x - A.x) * q, A.y + (B.y - A.y) * q, 2.6, 0, Math.PI * 2); ctx.fill();
        }
      }
      B.glint = Math.max(0, B.glint - dt / 600);
      flies.forEach(f => { if (f.dead !== null) drawBrain(f, t); if (f.dead === null || f.dead < 800) drawFly(f, t); });
      poofs = poofs.filter(p => t - p.t < 450);
      poofs.forEach(p => {
        const k = (t - p.t) / 450;
        ctx.strokeStyle = p.hit ? ACCENT : GREY; ctx.lineWidth = 1.5; ctx.globalAlpha = (p.hit ? 1 : 0.5) * (1 - k);
        ctx.beginPath(); ctx.arc(p.x, p.y, 8 + k * 26, 0, Math.PI * 2); ctx.stroke();
        for (let i = 0; i < 6; i++) {
          const a = i * Math.PI / 3, r0 = 10 + k * 22, r1 = r0 + 6;
          ctx.beginPath(); ctx.moveTo(p.x + Math.cos(a) * r0, p.y + Math.sin(a) * r0);
          ctx.lineTo(p.x + Math.cos(a) * r1, p.y + Math.sin(a) * r1); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      });
    }
    if (visible && !document.hidden) raf = requestAnimationFrame(frame);
  };
  const start = () => {
    if (still.matches) { cancelAnimationFrame(raf); raf = 0; drawStill(); return; }
    if (!raf && visible && !document.hidden) { last = 0; raf = requestAnimationFrame(frame); }
  };

  canvas.dataset.bg = getComputedStyle(document.body).backgroundColor;
  measure(); start();
  new ResizeObserver(() => { measure(); if (still.matches) drawStill(); }).observe(hero);
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; start(); }).observe(hero);
  document.addEventListener('visibilitychange', start);
  still.addEventListener('change', start);
  document.fonts?.ready.then(() => { measure(); start(); });
  window.addEventListener('pointermove', e => {
    const box = canvas.getBoundingClientRect();
    pointer = { x: e.clientX - box.left, y: e.clientY - box.top };
  }, { passive: true });
  hero.addEventListener('pointerleave', () => { pointer = null; });
  hero.addEventListener('pointerdown', e => {
    if (still.matches || !region || e.target.closest('a, button, input, select, label')) return;
    const box = canvas.getBoundingClientRect(), x = e.clientX - box.left, y = e.clientY - box.top;
    let hitAny = false;
    flies.forEach(f => {
      if (f.dead === null && Math.hypot(x - f.x, y - f.y) < 18 + f.size * 0.45) {
        f.dead = 0; f.bx = f.x; f.by = f.y; f.rest = 0; f.flee = 0; f.spin = (Math.random() < 0.5 ? -1 : 1) * 5; hitAny = true;
      }
    });
    poofs.push({ x, y, t: performance.now(), hit: hitAny });
    if (hitAny) message = null;
  });
}
