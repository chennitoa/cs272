import json


class MDP:
    """A Markov Decision Process object with the following internal variables:
    - states (list[str]): a list of state names
    - rewards (dict[str,float]): a dictionary of rewards by state names (Assume that an agent receive a
      state-specific reward when it reaches the corresponding state)
    - action_lists (dict[str:list[str]]): a dictionary of an available action list at each state
    - gamma (float): a discount factor
    - transitions (dict[str:dict[str:list[float]]]): A transition matrix: (state, action) -> a list of
      probabilities to each state (Hint: refer to mdp jason files)
    """
    def __init__(self, config_path: str) -> None:
        """load the MDP settings from a json config file
        Hint: Use json.load() to load the config file in the dictionary format

        Args:
            config_path (str): a path to the json config file

        Raises:
            ValueError: If a transition probability is invalid, raise an error.
        """
        with open(config_path, 'r') as infile:
            MDP_dict = json.load(infile)

        # Load trivial values from json
        transitions = MDP_dict['tran_prob']
        self.rewards = MDP_dict['rewards']
        self.gamma = MDP_dict['gamma']

        # Calculate states
        self.states_list = []
        for state in self.rewards:
            self.states_list.append(state)

        # Calculate action_lists
        self.action_lists = dict()
        for state in transitions:
            state_actions = []
            for action in transitions[state]:
                state_actions.append(action)
            self.action_lists.update({state: state_actions})

        # Generate transition sucessors
        self.tsuccessors = dict()
        for state in transitions:
            # Generate new dictionary with keys in place
            self.tsuccessors.update({state: dict()})
            for action in transitions[state]:
                self.tsuccessors[state].update({action: []})
                # Generate state probability tuples
                for index, trans_prob in enumerate(transitions[state][action]):
                    self.tsuccessors[state][action].append((
                        self.states_list[index],
                        trans_prob
                    ))

        # Verify probabilities
        self.__verify_probs(transitions)

    def __verify_probs(self, trans: dict[str, dict[str, list[float]]]):
        """raise ValueError if the transition matrix has invalid values. In particular,
        check that the sum of probabilities from a state to successor states is always 1.0.

        Args:
            trans (dict[str,dict[str,list[float]]]): transition matrix (transitions loaded from the json file)

        Raises:
            ValueError: If a transition probability is invalid, raise an error.
        """
        for state in trans:
            # Add up transition probabilities
            for action in trans[state]:
                if sum(trans[state][action]) != 1.0:
                    raise ValueError('Invalid transition probability.')

    def states(self) -> list[str]:
        """returns a list of all states

        Returns:
            list[str]: a list of states e.g. ["s0", "s1", "s2"]
        """
        return self.states_list

    def R(self, state: str) -> float:
        """returns a reward value of a given state

        Args:
            state (str): a state name e.g. "s0"

        Returns:
            float: a reward value
        """
        return self.rewards[state]

    def T(self, state: str, action: str) -> list:
        """returns a list of probabilities to all possible successor states

        Args:
            state (str): a state name e.g. "s0"
            action (str): an action e.g. "r"

        Returns:
            list[(str,float)]: a list of probabilities to all possible successor states
            e.g. [("s0", 0.1), ("s2", 0.3), ("s3", 0.6)]
        """
        return self.tsuccessors[state][action]

    def actions(self, state: str) -> list[str]:
        """returns a list of possible actions at a given state

        Args:
            state (str): a state name e.g. "s0"

        Returns:
            list[str]: a list of possible actions e.g. ["r", "l"]
        """
        return self.action_lists[state]
