"""Observable Rock-Paper-Scissors environment. Moves are rock=0, paper=1, scissors=2.

The opponent's rule is private. A fly sees only completed-round history; the
current opponent move is revealed after the fly has acted.
"""
from dataclasses import dataclass

from .sim import Rng

MOVES = (0, 1, 2)
NAMES = ('rock', 'paper', 'scissors')


def result(player, opponent):
    """Win=1, tie=0, loss=-1."""
    if player not in MOVES or opponent not in MOVES:
        raise ValueError('moves must be 0, 1, or 2')
    return (0, 1, -1)[(player - opponent) % 3]


@dataclass(frozen=True)
class Observation:
    previous_player: int | None
    previous_opponent: int | None
    previous_result: int | None


class Opponent:
    """Strategies use private configuration and their own seeded random stream."""

    def __init__(self, strategy='previous_opponent', seed=1, sequence=(0, 1, 2),
                 offset=1, switch_after=12, second_offset=2, noise=0.0):
        if strategy not in ('sequence', 'previous_opponent', 'previous_player', 'switch'):
            raise ValueError(f'unknown strategy: {strategy}')
        if not sequence or any(x not in MOVES for x in sequence):
            raise ValueError('sequence must contain RPS moves')
        if switch_after < 1:
            raise ValueError('switch_after must be positive')
        if not 0 <= noise <= 1:
            raise ValueError('noise must be between 0 and 1')
        self.strategy, self.sequence = strategy, tuple(sequence)
        self.offset, self.switch_after, self.second_offset = offset, switch_after, second_offset
        self.noise = noise
        self.rand = Rng(seed)
        self.previous_player = self.previous_opponent = self.previous_result = None
        self.round = 0
        self.initial = int(self.rand() * 3)

    def observe(self):
        return Observation(self.previous_player, self.previous_opponent, self.previous_result)

    def _move(self):
        if self.strategy == 'sequence':
            return self.sequence[(self.round + self.initial) % len(self.sequence)]
        if self.strategy == 'previous_player':
            previous = self.previous_player
        else:
            previous = self.previous_opponent
        if previous is None:
            return self.initial
        offset = self.second_offset if self.strategy == 'switch' and self.round >= self.switch_after else self.offset
        return (previous + offset) % 3

    def play(self, player):
        if player not in MOVES:
            raise ValueError('invalid player move')
        opponent = self._move()
        if self.noise and self.rand() < self.noise:
            opponent = int(self.rand() * 3)
        score = result(player, opponent)
        self.previous_player, self.previous_opponent, self.previous_result = player, opponent, score
        self.round += 1
        return opponent, score
