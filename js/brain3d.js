import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const roleColors = {
  vpn: '#6db7ff', kc: '#63c9ad', mbon: '#e466b8', pam: '#5cc17a',
  ppl1: '#f07878', mid: '#edc46a', dn: '#a694ff', vnc: '#87919d', apl: '#d8dfe6',
};
const flashColors = { pam: '#8dff9e', ppl1: '#ff9b9b' };
const regionNames = { vpn: 'visual input (VPN)', kc: 'mushroom body (KC)', mbon: 'mushroom body output (MBON)',
  pam: 'reward dopamine (PAM)', ppl1: 'punishment dopamine (PPL1)', mid: 'central brain', dn: 'descending neurons (DN)',
  vnc: 'nerve cord (VNC)', apl: 'APL' };
const color = value => new THREE.Color(value);

function dotTexture(ring) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const context = canvas.getContext('2d');
  context.beginPath();
  context.arc(32, 32, ring ? 24 : 30, 0, Math.PI * 2);
  context.fillStyle = context.strokeStyle = '#ffffff';
  context.lineWidth = 9;
  if (ring) context.stroke(); else context.fill();
  return new THREE.CanvasTexture(canvas);
}

// Display-only shading: surface direction is estimated from point density; positions are untouched.
function shadeFromDensity(positions, ink) {
  const size = 48, bounds = [1.25, 1.71, 0.88], light = new THREE.Vector3(0.25, 0.9, 0.35).normalize();
  const clamp = value => Math.min(size - 1, Math.max(0, value));
  const key = (x, y, z) => x + size * (y + size * z);
  const grid = (value, axis) => clamp((value / bounds[axis] + 1) / 2 * (size - 1));
  let density = new Float32Array(size ** 3);
  for (let i = 0; i < positions.length; i += 3)
    density[key(Math.round(grid(positions[i], 0)), Math.round(grid(positions[i + 1], 1)), Math.round(grid(positions[i + 2], 2)))]++;
  for (const step of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    const blurred = new Float32Array(density.length);
    for (let z = 0; z < size; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
      for (let d = -2; d <= 2; d++)
        blurred[key(x, y, z)] += density[key(clamp(x + d * step[0]), clamp(y + d * step[1]), clamp(z + d * step[2]))] / 5;
    density = blurred;
  }
  const sample = (px, py, pz) => {
    const gx = grid(px, 0), gy = grid(py, 1), gz = grid(pz, 2);
    const x = Math.min(size - 2, Math.floor(gx)), y = Math.min(size - 2, Math.floor(gy)), z = Math.min(size - 2, Math.floor(gz));
    const u = gx - x, v = gy - y, w = gz - z;
    let total = 0;
    for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++)
      total += density[key(x + i, y + j, z + k)] * (i ? u : 1 - u) * (j ? v : 1 - v) * (k ? w : 1 - w);
    return total;
  };
  const [hx, hy, hz] = bounds.map(bound => bound * 4 / size);
  const shaded = new Float32Array(positions.length), normal = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    normal.set((sample(x - hx, y, z) - sample(x + hx, y, z)) / hx,
      (sample(x, y - hy, z) - sample(x, y + hy, z)) / hy, (sample(x, y, z - hz) - sample(x, y, z + hz)) / hz);
    if (normal.lengthSq()) normal.normalize();
    const brightness = 0.18 + 0.82 * ((normal.dot(light) + 1) / 2) ** 1.6;
    shaded[i] = ink.r * brightness; shaded[i + 1] = ink.g * brightness; shaded[i + 2] = ink.b * brightness;
  }
  return shaded;
}

export class Brain3D {
  constructor(canvas, header, backgroundCoordinates, neuronCoordinates, sub, colors) {
    this.canvas = canvas;
    this.header = header;
    this.sub = sub;
    canvas.style.aspectRatio = `1 / ${sub.meta.aspect}`;
    if (backgroundCoordinates.length !== sub.meta.n_points * 3 ||
        neuronCoordinates.length !== sub.meta.n_neurons * 3)
      throw Error('3D soma coordinate sidecars do not match the connectome');
    this.role = sub.neurons.role;
    this.placed = [];
    this.baseColors = [];
    this.spikeColor = color('#ffffff');
    this.flash = new Map(Object.entries(flashColors).map(([role, hex]) => [role, color(hex)]));
    this.base = this.role.map(role => color(roleColors[role] ?? colors.ink));
    this.scene = new THREE.Scene();
    this.scene.background = color(colors.paper);
    this.scene.fog = new THREE.Fog(colors.paper, 1, 10);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 40);
    this.camera.position.set(3.6, 2.6, 4.6);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 4.1;
    this.controls.maxDistance = 10;
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.spin = !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    this.controls.autoRotate = this.spin;
    this.controls.autoRotateSpeed = 3; // updates run at 30 per second
    this.controls.addEventListener('start', () => { this.controls.autoRotate = false; });

    this.group = new THREE.Group();
    this.scene.add(this.group);
    const floor = new THREE.GridHelper(4, 12, colors.ink, colors.ink); // x-z reference plane
    floor.position.y = -1.85;
    floor.material.transparent = true;
    floor.material.opacity = 0.28;
    this.group.add(floor);
    this.addAnatomy(backgroundCoordinates, colors);
    this.addNeurons(sub, neuronCoordinates);
    this.addOverlays();
    this.hiddenRoles = new Set();
    this.net = null;
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 0.04;
    let down = null;
    canvas.addEventListener('pointerdown', event => { down = [event.clientX, event.clientY]; });
    canvas.addEventListener('click', event => {
      if (down && Math.hypot(event.clientX - down[0], event.clientY - down[1]) < 5)
        this.pickAt(event.clientX, event.clientY);
    });
    this.onScreen = true;
    this.lastDraw = 0;
    new IntersectionObserver(([entry]) => { this.onScreen = entry.isIntersecting; }).observe(canvas);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    this.render();
    this.animate();
  }

  addAnatomy(positions, colors) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(shadeFromDensity(positions, color(colors.ink)), 3));
    const material = new THREE.PointsMaterial({ size: 0.05, vertexColors: true });
    this.group.add(new THREE.Points(geometry, material));
  }

  addNeurons(sub, coordinates) {
    const positions = [];
    const colors = [];
    for (let index = 0; index < sub.neurons.x.length; index++) {
      const offset = index * 3;
      if (!Number.isFinite(coordinates[offset])) continue;
      const role = this.role[index];
      const base = this.base[index];
      this.placed.push(index);
      this.baseColors.push(base);
      positions.push(coordinates[offset], coordinates[offset + 1], coordinates[offset + 2]);
      colors.push(base.r, base.g, base.b);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    this.neuronColors = geometry.getAttribute('color');
    this.neuronPoints = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.075, vertexColors: true, sizeAttenuation: true, transparent: true,
      opacity: 0.92, depthWrite: false,
    }));
    this.group.add(this.neuronPoints);
  }

  // Recent spikes are redrawn in a layer that ignores the solid surface, so none are hidden.
  addOverlays() {
    const count = this.placed.length;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    geometry.setDrawRange(0, 0);
    this.spikePoints = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.1, map: dotTexture(false), alphaTest: 0.5, vertexColors: true, transparent: true,
      depthTest: false, depthWrite: false, fog: false,
    }));
    this.spikePoints.frustumCulled = false;
    this.spikePoints.renderOrder = 2;
    this.group.add(this.spikePoints);
    // Soft additive glow on the same real spikes: neurons firing together merge into a bright area.
    const halo = document.createElement('canvas');
    halo.width = halo.height = 64;
    const context = halo.getContext('2d');
    const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 64);
    this.spikeGlow = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.42, map: new THREE.CanvasTexture(halo), vertexColors: true, transparent: true, opacity: 0.16,
      blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, fog: false,
    }));
    this.spikeGlow.frustumCulled = false;
    this.spikeGlow.renderOrder = 1;
    this.group.add(this.spikeGlow);
    this.roleTotals = new Map();
    for (const index of this.placed)
      this.roleTotals.set(this.role[index], (this.roleTotals.get(this.role[index]) ?? 0) + 1);
    const marker = new THREE.BufferGeometry();
    marker.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    this.marker = new THREE.Points(marker, new THREE.PointsMaterial({
      size: 0.34, map: dotTexture(true), color: '#ffe066', depthTest: false, depthWrite: false, transparent: true, fog: false,
    }));
    this.marker.frustumCulled = false;
    this.marker.renderOrder = 3;
    this.marker.visible = false;
    this.group.add(this.marker);
    const message = new THREE.BufferGeometry();
    message.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    this.messageRing = new THREE.Points(message, new THREE.PointsMaterial({
      size: 0.42, map: dotTexture(true), color: '#ffffff', depthTest: false, depthWrite: false, transparent: true, fog: false,
    }));
    this.messageRing.frustumCulled = false;
    this.messageRing.renderOrder = 3;
    this.messageRing.visible = false;
    this.group.add(this.messageRing);
    this.messageNeurons = [];
  }

  // Rings the teaching neurons on the student's brain while they carry a message.
  setMessageNeurons(neurons) {
    this.messageNeurons = neurons;
    const source = this.neuronPoints.geometry.getAttribute('position');
    const target = this.messageRing.geometry.getAttribute('position');
    neurons.forEach((index, i) => {
      const point = this.placed.indexOf(index);
      target.setXYZ(i, source.getX(point), source.getY(point), source.getZ(point));
    });
    target.needsUpdate = true;
  }

  setHiddenRoles(roles) {
    this.hiddenRoles = roles;
    const visible = [];
    this.placed.forEach((index, point) => { if (!roles.has(this.role[index])) visible.push(point); });
    this.neuronPoints.geometry.setIndex(visible);
    if (this.net) this.frame(this.net);
  }

  pickAt(clientX, clientY) {
    const box = this.canvas.getBoundingClientRect();
    this.raycaster.setFromCamera(new THREE.Vector2(
      (clientX - box.left) / box.width * 2 - 1, 1 - (clientY - box.top) / box.height * 2), this.camera);
    const hit = this.raycaster.intersectObject(this.neuronPoints)[0];
    if (!hit) {
      this.marker.visible = false;
      this.onPick?.(null);
      return null;
    }
    const index = this.placed[hit.index];
    const position = this.neuronPoints.geometry.getAttribute('position');
    this.marker.geometry.getAttribute('position').setXYZ(0,
      position.getX(hit.index), position.getY(hit.index), position.getZ(hit.index));
    this.marker.geometry.getAttribute('position').needsUpdate = true;
    this.marker.visible = true;
    const age = this.net ? this.net.t - this.net.lastSpike[index] : null;
    const activity = age === null ? 'no live simulation yet' :
      age >= 0 && age < 600 ? `last spike ${Math.round(age)} ms ago` : 'no spike in the last 600 ms';
    const info = { index, type: this.sub.neurons.type[index], role: this.role[index], activity };
    this.onPick?.(info);
    return info;
  }

  frame(net) {
    this.net = net;
    let spiking = 0, shown = 0;
    const active = new Map();
    const t = net.t;
    const lastSpike = net.lastSpike;
    const positions = this.neuronPoints.geometry.getAttribute('position');
    const spikePositions = this.spikePoints.geometry.getAttribute('position');
    const spikeColors = this.spikePoints.geometry.getAttribute('color');
    for (let point = 0; point < this.placed.length; point++) {
      const index = this.placed[point];
      const age = t - lastSpike[index];
      const role = this.role[index];
      if (age < 10) spiking++;
      if (age >= 0 && age < 40 && !this.hiddenRoles.has(role)) active.set(role, (active.get(role) ?? 0) + 1);
      const flash = this.flash.get(role) ?? this.spikeColor;
      const weight = age >= 0 && age < 600 ? Math.exp(-age / 180) : 0;
      const base = this.baseColors[point];
      this.neuronColors.setXYZ(point, base.r + (flash.r - base.r) * weight,
        base.g + (flash.g - base.g) * weight, base.b + (flash.b - base.b) * weight);
      if (weight > 0.33 && !this.hiddenRoles.has(role)) {
        spikePositions.setXYZ(shown, positions.getX(point), positions.getY(point), positions.getZ(point));
        const tint = this.flash.get(role) ?? this.baseColors[point];
        spikeColors.setXYZ(shown, tint.r, tint.g, tint.b);
        shown++;
      }
    }
    this.neuronColors.needsUpdate = true;
    spikePositions.needsUpdate = spikeColors.needsUpdate = true;
    this.spikePoints.geometry.setDrawRange(0, shown);
    this.messageRing.visible = this.messageNeurons.some(index => {
      const age = t - lastSpike[index];
      return age >= 0 && age < 150;
    });
    let busiest = '', top = 0;
    for (const [role, count] of active) {
      if (this.roleTotals.get(role) < 5) continue; // one lone neuron firing is not an area
      const share = count / this.roleTotals.get(role);
      if (share > top) { top = share; busiest = `${regionNames[role] ?? role} ${Math.round(share * 100)}%`; }
    }
    this.header.textContent = `${spiking} firing${busiest ? ` · busiest: ${busiest}` : ''}`;
  }

  clear(label) {
    this.net = null;
    this.spikePoints.geometry.setDrawRange(0, 0);
    this.messageRing.visible = false;
    for (let point = 0; point < this.baseColors.length; point++) {
      const value = this.baseColors[point];
      this.neuronColors.setXYZ(point, value.r, value.g, value.b);
    }
    this.neuronColors.needsUpdate = true;
    this.header.textContent = label;
  }

  resetView() {
    this.controls.reset();
    this.controls.autoRotate = this.spin;
    this.group.rotation.set(0, 0, 0);
  }

  resize() {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, width * this.sub.meta.aspect);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  animate = (now = performance.now()) => {
    requestAnimationFrame(this.animate);
    if (!this.onScreen || now - this.lastDraw < 33) return;
    this.lastDraw = now;
    this.controls.update();
    const distance = this.camera.position.length();
    this.scene.fog.near = distance - 1.2;
    this.scene.fog.far = distance + 2.6;
    this.render();
  };
}
