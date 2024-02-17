from mypa1_testing import MDP
import math


class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need
    to append the v table to this list. (include the initial value)
    """
    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        self.mdp = mdp
        self.conv_thresh = conv_thresh

        # Initialize v table
        self.v_table = dict()
        for state in self.mdp.states():
            # Create entry for state
            self.v_table.update({state: 0.0})

        # Initialize q_table
        self.q_table = self.computeq_fromv(self.v_table)

        # Initialize random policy
        self.init_random_policy()

        # Maintain v_update_history
        self.v_update_history = [self.v_table]

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """
        self.policy = dict()
        for state in self.mdp.states():
            # Create dictionary for state
            self.policy.update({state: dict()})
            # Iterate over actions and initialize probability for each
            state_actionlist = self.mdp.actions(state)
            for action in state_actionlist:
                self.policy[state].update({action: (1.0 / len(state_actionlist))})

    def computeq_fromv(self, v: dict[str, float]) -> dict[str, dict[str, float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = r + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """
        q_table = dict()
        for state in self.mdp.states():
            # Create dictionary for state
            q_table.update({state: dict()})
            # Iterate over actions and generate probabilities for each
            state_actionlist = self.mdp.actions(state)
            for action in state_actionlist:
                # Get transition states
                t_states_probs = self.mdp.T(state, action)
                q_table[state].update({action: 0.0})
                # Compute weighted average of rewards and state value
                for t_state in t_states_probs:
                    value = self.mdp.R(state) + self.v_table[t_state[0]]
                    q_table[state][action] += t_state[1] * value

        # Completed q_table
        return q_table

    def greedy_policy_improvement(self, v: dict[str, float]) -> dict[str, dict[str, float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        new_policy = dict()
        # Get action value function
        q = self.computeq_fromv(v)
        for state in q:
            # Get the max action value and number of occurrences
            max_action_value = -math.inf
            action_value_occurences = 1
            for action in q[state]:
                if abs(q[state][action] - max_action_value) < 0.000001:
                    action_value_occurences += 1
                elif q[state][action] > max_action_value:
                    max_action_value = q[state][action]
                    action_value_occurences = 1
            # Update policy
            new_state_policy = dict()
            for action in q[state]:
                # If the action value was the decided upon max action value
                if abs(q[state][action] - max_action_value) < 0.000001:
                    new_state_policy.update({action: 1.0 / action_value_occurences})
                else:
                    new_state_policy.update({action: 0.0})
            # Add to return policy
            new_policy.update({state: new_state_policy})
        # Finished computing all states, return policy
        return new_policy

    def check_term(self, v: dict[str, float], next_v: dict[str, float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows:
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        for state in self.mdp.states():
            update_delta = abs(next_v[state] - v[state])
            if update_delta > self.conv_thresh:
                return True

        # All states compliant with convergent threshold, state value has converged
        return False


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        super(PIAgent, self).__init__(mdp, conv_thresh=conv_thresh)

    def iter_policy_eval(self, pi: dict[str, dict[str, float]]) -> dict[str, float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This is a function used in PI.

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        new_v_table = dict()
        for state in pi:
            state_value = 0
            for action in pi[state]:
                for transition in self.mdp.T(state, action):
                    # Total probability of moving to state s' is policy * transition function
                    # Multiply by reward from current state and s' state value
                    state_value += pi[state][action] * transition[1] * \
                        (self.mdp.R(state) + self.v_table[transition[0]])
            # Computed all state values
            new_v_table.update({state: state_value})
        # Finished computing v for all states
        return new_v_table

    def policy_iteration(self) -> dict[str, dict[str, float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement,
        update the policy pi until convergence of the state-value function.

        This will be the function called to run PI.
        e.g.
        mdp = MDP('grid16.json')
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        # Prevent infinite loop
        iterations = 0
        while iterations < 10000:
            new_v_table = self.iter_policy_eval(self.policy)
            while self.check_term(self.v_table, new_v_table):
                self.v_table = new_v_table
                # Old table has already been appended, append new table
                self.v_update_history.append(self.v_table)
                # Continue iterative policy evaluation
                new_v_table = self.iter_policy_eval(self.policy)
            # Policy improvement
            new_policy = self.greedy_policy_improvement(self.v_table)
            if new_policy == self.policy:
                # Function exists, optimal policy found
                return self.policy
            else:
                self.policy = new_policy
                iterations += 1


class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        super(VIAgent, self).__init__(mdp, conv_thresh=conv_thresh)

    def value_iteration(self) -> dict[str, dict[str, float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration.
        After that, generate the corresponding optimal policy pi.

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        # Prevent infinite loop
        iterations = 0
        while iterations < 10000:
            # State value function for value iteration is just maximum action value
            q_table = self.computeq_fromv(self.v_table)
            new_v_table = dict()
            for state in self.mdp.states():
                if len(self.mdp.actions(state)) == 0:
                    # Terminal state, do nothing
                    new_v_table.update({state: self.mdp.R(state)})
                    continue
                # Nonterminal state
                new_v_table.update({state: -math.inf})
                for action in self.mdp.actions(state):
                    if q_table[state][action] > new_v_table[state]:
                        new_v_table[state] = q_table[state][action]
            # Found maxmimum action value for all states, check convergence
            convergence = self.check_term(self.v_table, new_v_table)
            self.v_table = new_v_table
            self.v_update_history.append(self.v_table)
            if not convergence:
                break
            else:
                iterations += 1

        new_policy = self.greedy_policy_improvement(self.v_table)
        return new_policy
