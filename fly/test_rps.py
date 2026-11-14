"""Focused tests for the RPS experiment. Run: python -m unittest flyhero.test_rps"""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from .rps import MOVES, Observation, Opponent, result
from .rps_channel import ActivityChannel
from .rps_learning import RpsBioRL, RpsEpisode
from .sim import Network, wiring

ROOT = Path(__file__).resolve().parents[1]


class RpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sub = json.loads((ROOT / 'data/subgraph.json').read_text())
        cls.edges = wiring(cls.sub)

    def fly(self, seed=7):
        net = Network(self.sub, self.edges, seed=seed)
        return net, RpsBioRL(net, seed=seed)

    def test_rules_observations_and_strategies(self):
        self.assertEqual([result(p, 0) for p in MOVES], [0, 1, -1])
        for strategy in ('sequence', 'previous_opponent', 'previous_player', 'switch'):
            a = Opponent(strategy, seed=44, noise=0.1)
            b = Opponent(strategy, seed=44, noise=0.1)
            self.assertEqual(a.observe(), Observation(None, None, None))
            self.assertEqual([a.play(i % 3) for i in range(30)],
                             [b.play(i % 3) for i in range(30)])
            self.assertEqual(set(vars(a.observe())),
                             {'previous_player', 'previous_opponent', 'previous_result'})

    def test_independent_brains_and_learning(self):
        a, aa = self.fly()
        b, ba = self.fly()
        original = b.m.copy()
        RpsEpisode(a, aa, Opponent(seed=91), rounds=6).run()
        self.assertTrue(np.any(a.m != 1), 'dopamine learning did not update KC->MBON weights')
        self.assertTrue(np.array_equal(b.m, original), 'one fly changed the other fly')
        self.assertFalse(np.shares_memory(a.m, b.m))
        self.assertFalse(np.shares_memory(a.v, b.v))
        self.assertIsNot(aa.rand, ba.rand)

    def test_channel_restrictions_and_log(self):
        teacher, agent = self.fly()
        student, _ = self.fly(8)
        c = ActivityChannel(self.sub, student, bandwidth=2)
        roles = self.sub['neurons']['role']
        self.assertTrue(all(roles[i] == 'mid' for i in c.source))
        self.assertTrue(all(roles[i] == 'vpn' for ids in c.target for i in ids))
        self.assertTrue(all(set(ids).isdisjoint(set(student.lane_vpn[k][2:]))
                            for k, ids in enumerate(c.target)))
        ep = RpsEpisode(teacher, agent, Opponent(seed=91), rounds=3,
                        channel=c, capture_signals=True).run()
        before = student.m.copy()
        for signal in ep.signals:
            c.inject(student, signal)
            c.clear(student)
        self.assertEqual(c.log, ep.signals)
        self.assertTrue(all(len(x) == 2 and all(v in (0, 1, 2) for v in x) for x in c.log))
        self.assertTrue(np.array_equal(before, student.m))
        with self.assertRaises(ValueError):
            c.inject(student, [0, 3])

    def test_reproducible_run(self):
        def run():
            net, agent = self.fly(12)
            ep = RpsEpisode(net, agent, Opponent(seed=119, noise=0.1), rounds=3).run()
            return [(e['action'], e['opponent'], e['reward']) for e in ep.events], net.m.copy()
        events1, m1 = run()
        events2, m2 = run()
        self.assertEqual(events1, events2)
        self.assertTrue(np.array_equal(m1, m2))

    def test_student_plays_disconnected_without_learning(self):
        net, agent = self.fly(13)
        RpsEpisode(net, agent, Opponent(seed=21), rounds=3).run()
        learned = net.m.copy()
        independent = RpsEpisode(net, agent, Opponent(seed=99, noise=0.1),
                                 rounds=3, learn=False).run()
        self.assertEqual(len(independent.events), 3)
        self.assertTrue(np.array_equal(net.m, learned))
        self.assertTrue(all(e['signal'] is None for e in independent.events))

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for JS/Python parity')
    def test_browser_port(self):
        net, agent = self.fly(7)
        channel = ActivityChannel(self.sub, net)
        ep = RpsEpisode(net, agent, Opponent(seed=91), rounds=6,
                        channel=channel, capture_signals=True).run()
        js = json.loads(subprocess.check_output(['node', 'tools/rps_trace.mjs'], cwd=ROOT, text=True))
        self.assertEqual(js['events'], [[e['action'], e['opponent'], e['reward']] for e in ep.events])
        self.assertEqual(js['signals'], [list(x) for x in ep.signals])
        self.assertAlmostEqual(js['mSum'], float(net.m.sum()), places=6)


if __name__ == '__main__':
    unittest.main()
