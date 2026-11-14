// Background thread for Fly B training; frozen checks run on their own pool of threads.
import { runStudents } from './rps_student_engine.js';

const cores = self.navigator?.hardwareConcurrency ?? 4;
const poolSize = Math.max(1, Math.min(4, cores - 2));
const control = { paused: false, slow: false, speed: 1, step: 0 };
const pool = [], queue = [], waiting = new Map();
let jobs = 0;

function dispatch(worker) {
  const next = queue.shift();
  if (!next) { worker.busy = false; return; }
  worker.busy = true;
  worker.postMessage(next);
}
function runCheckpoint(job) {
  return new Promise((resolve, reject) => {
    const id = ++jobs;
    waiting.set(id, { resolve, reject });
    queue.push({ id, ...job });
    const idle = pool.find(worker => !worker.busy);
    if (idle) dispatch(idle);
  });
}
function startPool() {
  for (let i = 0; i < poolSize; i++) {
    const worker = new Worker(new URL('./rps_checkpoint_worker.js', import.meta.url), { type: 'module' });
    worker.busy = false;
    worker.onmessage = ({ data }) => {
      const job = waiting.get(data.id);
      waiting.delete(data.id);
      if (data.error) job.reject(Error(data.error)); else job.resolve(data.result);
      dispatch(worker);
    };
    worker.onerror = event => {
      self.postMessage({ type: 'error', message: event.message || 'Checkpoint thread failed' });
    };
    pool.push(worker);
  }
}

self.onmessage = async ({ data }) => {
  if (data.type === 'control') { Object.assign(control, data.control); return; }
  Object.assign(control, data.control);
  if (!data.params.preview) startPool();
  try {
    for await (const event of runStudents({ ...data.params, control, runCheckpoint })) {
      const transfer = event.type === 'frame' ? event.nets.map(net => net.lastSpike.buffer) : [];
      self.postMessage(event, transfer);
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message });
  }
  pool.forEach(worker => worker.terminate());
  self.postMessage({ type: 'closed' });
};
