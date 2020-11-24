import torch
import torch.nn as nn
import torch.nn.functional as func
from policies.generic_net import GenericNet


"""
DDPG + HER (Hindsight Experience Replay) 
https://youtu.be/0Ey02HT_1Ho?t=768
"""

def DDPG_Policy(GenericNet):

    def __init__(self, l1, l2, l3, l4, learning_rate):
        """
        :param l1: dim(state)
        :param l2: # nodes in fist hidden layer
        :param l3: # nodes in second hidden layer
        :param l4: dim(action)
        :param learning_rate: learning rate 
        """
        super(DDPG_Policy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """
        Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
        The obtained tensors can be used to obtain an action by calling select_action
        :param state: the input state(s)
        :return: the resulting pytorch tensor (the action)
        """
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        action = torch.sigmoid(self.fc3(state))
        return action

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            action = self.forward(state)
            return action