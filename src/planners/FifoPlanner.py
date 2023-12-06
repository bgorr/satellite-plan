from planners.AbstractPlanner import AbstractPlanner
from copy import deepcopy
from planners import utils






class FifoPlanner(AbstractPlanner):

    def __init__(self, settings):
        super().__init__(settings)

    # ---------------------------------------------
    # Fifo action
    # ---------------------------------------------

    def select_action(self, sat, state, num_actions, rand_action=False):
        # If using dl-action
        # if len(sat['storage']) >= self.settings['sat-storage-cap'] and num_actions == 6:
        #     action_idx = num_actions - 1  # last action is down-link
        # else:
        #     action_idx = 0  # always select first action in fifo

        action_idx = 0
        return action_idx














