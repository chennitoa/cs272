import timeit
import mypa1_testing
import mypa2_dp
import matplotlib.pyplot as plt


MDP = mypa1_testing.MDP('grid16.json')


def time_PI(conv_thresh=0.000001):
    PIAgent = mypa2_dp.PIAgent(MDP, conv_thresh=conv_thresh)
    return timeit.timeit(PIAgent.policy_iteration, number=1)


def time_VI(conv_thresh=0.000001):
    VIAgent = mypa2_dp.VIAgent(MDP, conv_thresh=conv_thresh)
    return timeit.timeit(VIAgent.value_iteration, number=1)


time_PI_normal = time_PI()
time_VI_normal = time_VI()

thresholds = [0.0001, 0.001, 0.003, 0.006, 0.01]
times_PI = []
times_VI = []

for thresh in thresholds:
    times_PI.append(time_PI(thresh))
    times_VI.append(time_VI(thresh))

plt.bar(['PIAgent Time', 'VIAgent Time'], [time_PI_normal, time_VI_normal])
plt.show()

plt.xlabel('Convergence Threshold')
plt.ylabel('Execution Time')

plt.scatter(thresholds, times_PI)
plt.show()

plt.scatter(thresholds, times_VI)
plt.show()
