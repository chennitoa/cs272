import numpy as np
import matplotlib.pyplot as plt


def ucb1_bandit(arms, num_steps):
    """
    UCB1 algorithm for the multi-armed bandit problem.

    Args:
    arms (list): List of arms (bandit machines) with their true reward probabilities.
    num_steps (int): Number of steps (iterations) to run the algorithm.

    Returns:
    selected_arms (list): List of arms selected at each step.
    rewards (list): List of rewards obtained at each step.
    """
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled.
    total_rewards = np.zeros(num_arms)  # Total rewards obtained from each arm.
    selected_arms = []  # List to store the selected arms.
    rewards = []  # List to store the rewards obtained.

    for step in range(num_steps):
        ucb_values = total_rewards / (num_pulls + 1e-6) + \
            np.sqrt(2 * np.log(step + 1) / (num_pulls + 1e-6))

        chosen_arm = np.argmax(ucb_values)

        # Simulate a Bernoulli reward
        reward = np.random.binomial(1, arms[chosen_arm])

        # Update the records
        num_pulls[chosen_arm] += 1
        total_rewards[chosen_arm] += reward

        selected_arms.append(chosen_arm)
        rewards.append(reward)

    return selected_arms, rewards


def epsilon_greedy_bandit(arms, num_steps, epsilon):
    """
    Epsilon-Greedy algorithm for the multi-armed bandit problem.

    Args:
    arms (list): List of arms (bandit machines) with their true reward probabilities.
    num_steps (int): Number of steps (iterations) to run the algorithm.
    epsilon (float): The exploration-exploitation trade-off parameter (0 <= epsilon <= 1).

    Returns:
    selected_arms (list): List of arms selected at each step.
    rewards (list): List of rewards obtained at each step.
    """
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled.
    total_rewards = np.zeros(num_arms)  # Total rewards obtained from each arm.
    selected_arms = []  # List to store the selected arms.
    rewards = []  # List to store the rewards obtained.

    for step in range(num_steps):
        if np.random.rand() < epsilon:
            # Explore
            chosen_arm = np.random.choice(num_arms)
        else:
            # Exploit
            # Average performance total / #trials
            # Argmax of the averages
            qs = total_rewards / (num_pulls + 1e-6)
            chosen_arm = np.argmax(qs)

        # Simulate a Bernoulli reward
        reward = np.random.binomial(1, arms[chosen_arm])

        # Update the records
        num_pulls[chosen_arm] += 1
        total_rewards[chosen_arm] += reward

        selected_arms.append(chosen_arm)
        rewards.append(reward)

    return selected_arms, rewards


true_probs = [0.1, 0.4, 0.45]
num_steps = 100000

selected_arms, rewards = epsilon_greedy_bandit(true_probs, num_steps, 0.01)

print('---------- Epsilon Greedy Bandit ----------')
print(selected_arms.count(0), selected_arms.count(1), selected_arms.count(2))
print(f' Expected reward = {np.average(rewards)}')


selected_arms, rewards = ucb1_bandit(true_probs, num_steps)


print('---------- UCB Bandit ----------')
print(selected_arms.count(0), selected_arms.count(1), selected_arms.count(2))
print(f' Expected reward = {np.average(rewards)}')
