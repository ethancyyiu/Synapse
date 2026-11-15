// Fraud-response labels and rewards are synthetic. The connectome learners use
// the existing fault-response interface: observe() supplies one of three cues,
// play() returns a deterministic reward for one of three actions.
import { rng } from './sim.js';
import { result } from './rps.js';

// Internal action order preserves the fault-response reward mapping exactly.
// UI order is APPROVE, REVIEW, BLOCK; internal indices are BLOCK, APPROVE, REVIEW.
export const ACTIONS = Object.freeze(['BLOCK', 'APPROVE', 'REVIEW']);
export const CUES = Object.freeze([
  'Isolated or established activity',
  'Recent purchase cluster',
  'Large deviation with novelty or a cluster',
]);
export const OLD_NEED = Object.freeze([0, 0, 1]);
export const NEW_NEED = Object.freeze([1, 1, 2]);
export const correctAction = (need, cue) => (need[cue] + 1) % 3;

export function balancedCues(seed, rounds = 72) {
  if (rounds % 3) throw Error('Balanced cue count must be divisible by three');
  const values = Array.from({ length: rounds }, (_, i) => i % 3), rand = rng(seed);
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}

export class FraudResponseGame {
  constructor(need, seed, sequence = null) {
    if (!Array.isArray(need) || need.length !== 3 || need.some(x => !Number.isInteger(x) || x < 0 || x > 2))
      throw Error('Invalid synthetic response mapping');
    this.need = [...need]; this.rand = rng(seed); this.sequence = sequence;
    this.strategy = sequence ? 'sequence' : 'fraud_response';
    this.round = 0; this.draw();
  }
  draw() {
    this.cue = this.sequence ? this.sequence[this.round] : Math.floor(this.rand() * 3);
    if (!Number.isInteger(this.cue) || this.cue < 0 || this.cue > 2)
      throw Error('Cue schedule exhausted or invalid');
  }
  observe() { return { previousPlayer: null, previousOpponent: this.cue, previousResult: null }; }
  play(action) {
    const opponent = this.need[this.cue], reward = result(action, opponent);
    this.round++;
    if (!this.sequence || this.round < this.sequence.length) this.draw();
    return { opponent, reward };
  }
}
