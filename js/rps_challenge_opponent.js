// Opponent construction shared by the page and background threads.
import { Opponent } from './rps.js';

export function analyzePattern(sequence) {
  if (!Array.isArray(sequence) || sequence.length < 1 || sequence.length > 5 ||
    sequence.some(action => !Number.isInteger(action) || action < 0 || action > 2))
    throw Error('Custom pattern must contain 1–5 Rock, Paper, or Scissors moves.');
  const successors = [0, 1, 2].map(() => new Set());
  sequence.forEach((action, index) => successors[action].add(sequence[(index + 1) % sequence.length]));
  const ambiguous = successors.flatMap((next, action) => next.size > 1 ? [action] : []);
  return { ambiguous, successors: successors.map(next => [...next]),
    distinguishable: ambiguous.length === 0,
    coverage: [0, 1, 2].filter(action => successors[action].size > 0) };
}

export function challengeOpponent(config, seed) {
  if (config.mode === 'preset' && [1, 2].includes(config.offset) && [0, 0.3].includes(config.noise ?? 0))
    return new Opponent({ strategy: 'previous_opponent', offset: config.offset, noise: config.noise ?? 0, seed });
  if (config.mode === 'custom') {
    analyzePattern(config.sequence);
    const opponent = new Opponent({ strategy: 'sequence', sequence: [...config.sequence], noise: 0, seed });
    opponent.initial = 0; // Existing sequence mode otherwise rotates its first move by seed.
    return opponent;
  }
  throw Error('Choose a supported opponent mode.');
}
