import gymnasium as gym
from mypa2_mf_control import QLAgent
import json
import timeit

agent = QLAgent(gym.make("Blackjack-v1"), eps=0.4)

time = timeit.timeit(agent.learn, number=1)

print(f'Learning complete, took {time} seconds')

episode, done = agent.best_run()
print('---------- BEST RUN ----------')
for experience in episode:
    print(f'Visited {experience[0]}, took action {experience[1]}, received reward {experience[2]}')
print('---------- TOTAL RETURN ----------')
print(agent.calc_return(episode, done))

# Dump agent q_table information
with open('agent_q.json', 'w') as infile:
    json.dump(agent.q, infile, indent=4)
