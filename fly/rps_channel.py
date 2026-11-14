"""Restricted, quantized activity channel between independently simulated flies.

Three intermediate neurons downstream of MBONs were selected by strong MBON
input and nonzero activity on an untrained sensory calibration. This selection
does not use RPS rewards or the hidden opponent rule. They are not known
fly-to-fly communication neurons.
"""

MID_SOURCES = (2709, 2360, 2325)


class ActivityChannel:
    def __init__(self, sub, net, bandwidth=3, target_width=2):
        if not 1 <= bandwidth <= 3 or not 1 <= target_width <= 2:
            raise ValueError('bandwidth must be 1..3 and target_width 1..2')
        if any(sub['neurons']['role'][i] != 'mid' for i in MID_SOURCES):
            raise ValueError('connectome does not match the documented channel sources')
        self.source = MID_SOURCES[:bandwidth]
        self.target = tuple(tuple(net.lane_vpn[k][:target_width]) for k in range(bandwidth))
        self.bandwidth = bandwidth
        self.log = []

    def sample(self, teacher_net):
        # At t=180 ms, before the teacher chooses or sees the current opponent move.
        return tuple(min(2, int(teacher_net.count[i])) for i in self.source)

    def inject(self, student_net, signal):
        if len(signal) != self.bandwidth or any(x not in (0, 1, 2) for x in signal):
            raise ValueError('invalid neural signal')
        for cells, value in zip(self.target, signal):
            student_net.set_drive(list(cells), value * 75)
        self.log.append(tuple(signal))

    def clear(self, student_net):
        for cells in self.target:
            student_net.set_drive(list(cells), 0)
