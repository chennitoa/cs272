import unittest
import mypa1_testing


class TestMyPA(unittest.TestCase):

    def setUp(self) -> None:
        """instantiates an MDP using a jason file
        """
        self.MDP = mypa1_testing.MDP('mdp1.json')

    def test_states(self):
        """checks if all the states in the jason file are returned as a list by states() e.g. ["s0", "s1"]
        """
        self.assertEqual(self.MDP.states(), ['s0', 's1', 's2'])

    def test_reward(self):
        """checks if all the rewards in the json file are returned by R(s)
        """
        self.assertEqual(self.MDP.R('s0'), 3.0)
        self.assertEqual(self.MDP.R('s1'), 1.0)
        self.assertEqual(self.MDP.R('s2'), -1.0)

    def test_actions(self):
        """checks if all the actions in the json file are returned as a list by actions(s) e.g. ['r','l']
        """
        self.assertEqual(self.MDP.actions('s0'), ['r', 'l'])
        self.assertEqual(self.MDP.actions('s1'), ['r', 'l'])
        self.assertEqual(self.MDP.actions('s2'), ['r', 'l'])

    def test_transition(self):
        """checks if the probabilities to other states are returned as a list of pairs of state and probability e.g. [("s0", 0.1), ("s1", 0.9)]
        """
        self.assertEqual(self.MDP.T('s0', 'r'), [('s0', 0.1), ('s1', 0.4), ('s2', 0.5)])
        self.assertEqual(self.MDP.T('s0', 'l'), [('s0', 0.1), ('s1', 0.4), ('s2', 0.5)])
        self.assertEqual(self.MDP.T('s1', 'r'), [('s0', 0.1), ('s1', 0.4), ('s2', 0.5)])
        self.assertEqual(self.MDP.T('s1', 'l'), [('s0', 0.1), ('s1', 0.4), ('s2', 0.5)])
        self.assertEqual(self.MDP.T('s2', 'r'), [('s0', 0.1), ('s1', 0.4), ('s2', 0.5)])
        self.assertEqual(self.MDP.T('s2', 'l'), [('s0', 0.1), ('s1', 0.4), ('s2', 0.5)])

    def test_verify_Tmatrix(self):
        """checks if ValueError is correctly raised when a loaded jason file has an invalid probability matrix
        """
        self.assertRaises(ValueError, lambda: mypa1_testing.MDP('mdp_proberror.json'))


if __name__ == '__main__':
    unittest.main()
