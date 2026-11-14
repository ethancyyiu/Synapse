// One isolated, teacher-disconnected frozen check on its own thread.
import { inlineCheckpoint } from './rps_student_engine.js';

self.onmessage = async ({ data: { id, ...job } }) => {
  try {
    self.postMessage({ id, result: await inlineCheckpoint(job) });
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
