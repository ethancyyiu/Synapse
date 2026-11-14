"""RPS decisions made by the original connectome and dopamine plasticity.

The three visual-projection-neuron groups encode the last publicly observed
opponent move. They were arbitrary rhythm-game lane groups, not RPS neurons.
No current opponent move or strategy label enters the network.
"""
import math

from .agents import BioRL
from .rps import Opponent

ROUND_MS = 400
T_COUNT, T_SIGNAL, T_DECIDE, T_LEARN = 50, 180, 240, 340
INPUT_HZ = 150


class RpsBioRL(BioRL):
    """The existing KC->MBON plasticity with a mandatory three-move readout."""

    def decide(self, net):
        rates = [int(net.count[ids].sum()) / len(ids) for ids in net.group_mbon]
        best = max(rates)
        winners = [k for k in range(3) if rates[k] == best]
        self.rates = rates
        self.action = winners[int(self.rand() * len(winners))]
        self.V = math.tanh(best / self.o['c_ref'])
        self.elig = net.trace[net.plastic_pre[self.action]]
        return self.action


class RpsEpisode:
    """A run of RPS rounds; weights survive between episodes, neural state resets."""

    def __init__(self, net, agent, opponent, rounds=24, learn=True,
                 channel=None, incoming_signals=None, capture_signals=False):
        self.net, self.agent, self.opponent = net, agent, opponent
        self.rounds, self.learn = rounds, learn
        self.t, self.events = 0, []
        self.channel, self.incoming_signals = channel, incoming_signals
        self.capture_signals, self.signals = capture_signals, []
        if incoming_signals is not None and (channel is None or len(incoming_signals) != rounds):
            raise ValueError('incoming signals require a channel and one signal per round')
        net.reset_state()

    @property
    def done(self):
        return self.t >= self.rounds * ROUND_MS

    @property
    def win_rate(self):
        return sum(e['reward'] == 1 for e in self.events) / len(self.events) if self.events else 0

    def step(self):
        round_no, phase = divmod(self.t, ROUND_MS)
        if phase == 0:
            self._start_round()
        elif phase == T_COUNT:
            self.net.reset_counts()
        elif phase == T_SIGNAL:
            self._exchange_signal(round_no)
        elif phase == T_DECIDE:
            self._play_round(round_no)
        elif phase == T_LEARN:
            self.agent.after_dopamine(self.net, self.learn)
        self.net.step()
        self.t += 1

    def _start_round(self):
        self.obs = self.opponent.observe()
        if self.obs.previous_opponent is not None:
            self.net.set_drive(self.net.lane_vpn[self.obs.previous_opponent][2:], INPUT_HZ)

    def _exchange_signal(self, round_no):
        if self.capture_signals:
            self.signals.append(self.channel.sample(self.net))
        if self.incoming_signals is not None:
            self.channel.inject(self.net, self.incoming_signals[round_no])

    def _play_round(self, round_no):
        if self.obs.previous_opponent is not None:
            self.net.set_drive(self.net.lane_vpn[self.obs.previous_opponent][2:], 0)
        if self.incoming_signals is not None:
            self.channel.clear(self.net)

        action = self.agent.decide(self.net)
        opponent_move, reward = self.opponent.play(action)
        self.events.append({'round': round_no, 'observation': self.obs,
                            'action': action, 'opponent': opponent_move,
                            'reward': reward, 't': self.t,
                            'rates': tuple(self.agent.rates),
                            'signal': self.incoming_signals[round_no] if self.incoming_signals is not None else None})
        self.agent.reward(self.net, reward)
        self.net.reset_counts()

    def run(self):
        while not self.done:
            self.step()
        return self


def run_training(net, agent, strategy='previous_opponent', seed=1,
                 episodes=10, rounds=24, **opponent_options):
    curve = []
    for i in range(episodes):
        opponent = Opponent(strategy, seed=seed * 1009 + i, **opponent_options)
        curve.append(RpsEpisode(net, agent, opponent, rounds).run().win_rate)
    return curve
