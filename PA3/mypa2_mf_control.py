from typing import Any
import random
import math
import gymnasium as gym


def argmax_action(d: dict[Any, float]) -> Any:
    """return a key of the maximum value in a given dictionary

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    max_actions = []
    max_value = -math.inf / 2.0  # Avoid overflowing
    for action in d:
        if abs(d[action] - max_value) < 0.00001:
            # Action has same action value
            max_actions.append(action)
        elif d[action] > max_value:
            # Action is superior
            max_actions.clear()
            max_actions.append(action)
            max_value = d[action]

    # Return random action from optimal
    return random.choice(max_actions)


class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma: float = 0.98, eps: float = 0.2,
                 alpha: float = 0.02, total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not
                decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi

    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int, dict[int, float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        q_table = dict()
        for state in range(n_states):
            # Create table for every state
            q_table.update({state: dict()})
            state_table = q_table[state]
            for action in range(n_actions):
                state_table.update({action: init_val})

        # Return the first dictionary
        return q_table

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True;
                take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        if exploration:
            if random.random() < self.eps:
                # Greedy
                action = argmax_action(self.q[state])
                return action
            else:
                # Explore
                action = random.choice(list(self.q[state].keys()))
                return action
        else:
            # Greedy
            action = argmax_action(self.q[state])
            return action

    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy

        Args:
            ss (int): state

        Returns:
            int: action
        """
        return self.eps_greedy(ss, True)

    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int, int, float]], bool]:
        """After the learning, an optimal episode (based on the latest self.q) needs to be generated for evaluation.
            From the initial state, always take the greedily best action until it reaches a goal.

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent
                cannot reach the goal after max_steps. One step is (s,a,r) Defaults to 100.

        Returns:
            tuple[
                list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
                bool: done - True if the episode reaches a goal, False if it hits max_steps.
            ]
        """
        # Get initial state
        state = self.env.reset()[0]
        steps = 0
        episode = []
        while steps < max_steps:
            action = self.eps_greedy(state, False)
            next_state, reward, goal, info, done = self.env.step(action)
            episode.append((state, action, reward))

            # If reached terminal state, return values
            if goal:
                return episode, True

            # Increment steps, next step
            steps += 1
            state = next_state

        # Reached maximum steps
        return episode, False

    def calc_return(self, episode: list[tuple[int, int, float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """
        if not done:
            return None

        total_return = 0
        for index, experience in enumerate(episode):
            # Later experiences receive less reward due to discounting
            total_return *= self.gamma
            total_return += experience[2]

        return total_return


class MCCAgent(ValueRLAgent):
    def learn(self) -> None:
        """Monte Carlo Control algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        Note: When an episode is too long (> 500 for CliffWalking), you can stop the episode
            and update the table using the partial episode.

        The results should be reflected to its q table.
        """
        returns = dict()
        for i in range(self.total_epi):
            # Generate episode
            episode = self.generate_episode(500)
            visited = [(experience[0], experience[1]) for experience in episode]
            q_value = 0
            for index, experience in reversed(list(enumerate(episode))):
                # Unpack for convenience
                state, action, reward = experience

                # Update q value
                q_value *= self.gamma
                q_value += reward

                # If first visit
                if (state, action) not in visited[0:index - 1]:
                    # Add key to returns table
                    if (state, action) not in returns:
                        returns.update({(state, action): []})

                    # Update returns table and q
                    returns[state, action].append(q_value)
                    self.q[state][action] = sum(returns[state, action]) / len(returns[state, action])

    def generate_episode(self, max_steps: int = 500) -> list[tuple[int, int, float]]:
        """Generate an episode with the maximum number of steps

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent
                cannot reach the goal after max_steps. One step is (s,a,r) Defaults to 500.

        Returns:
            list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
        """
        # Get initial state
        state = self.env.reset()[0]
        steps = 0
        episode = []
        while steps < max_steps:
            action = self.choose_action(state)
            next_state, reward, goal, info, done = self.env.step(action)
            episode.append((state, action, reward))

            # If reached terminal state, return values
            if goal:
                return episode

            # Increment steps, next step
            steps += 1
            state = next_state

        # Reached maximum steps
        return episode


class SARSAAgent(ValueRLAgent):
    def learn(self) -> None:
        """SARSA algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        for i in range(self.total_epi):
            # Generate episode
            state = self.env.reset()[0]
            action = self.choose_action(state)
            steps = 0
            max_steps = 500
            while steps < max_steps:
                # Take action
                next_state, reward, goal, info, done = self.env.step(action)

                # Calculate next action early
                next_action = self.choose_action(state)
                self.q[state][action] += self.alpha * \
                    (reward + self.gamma * self.q[next_state][next_action] - self.q[state][action])

                # Next step
                state = next_state
                action = next_action

                if goal:
                    # Reached terminal state, next episode
                    break


class QLAgent(SARSAAgent):
    def learn(self):
        """Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        for i in range(self.total_epi):
            # Generate episode
            state = self.env.reset()[0]
            steps = 0
            max_steps = 500
            while steps < max_steps:
                # Calculate and take action
                action = self.choose_action(state)
                next_state, reward, goal, info, done = self.env.step(action)

                # Update q table
                self.q[state][action] += self.alpha * \
                    (reward + self.gamma * self.q[next_state][argmax_action(self.q[next_state])]
                     - self.q[state][action])

                # Next step
                state = next_state

                if goal:
                    # Reached terminal state, next episode
                    break

    def choose_action(self, ss: int) -> int:
        """
        [optional] You may want to override this method.
        """
        return self.eps_greedy(ss, True)
