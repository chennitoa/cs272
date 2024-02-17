import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()

G = 0

for _ in range(1000):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(a)

    # Train DQN

    G += reward

    if terminated or truncated:
        print(G)
        G = 0
        obs, info = env.reset()

env.close()
