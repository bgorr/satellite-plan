import numpy as np
import os
import csv
import random
import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm
from mcts_planner import monte_carlo_tree_search
from dp_planner import graph_search, graph_search_events, graph_search_events_interval
from models.SatelliteMLP import SatelliteMLP
from models.SatelliteTransformer import SatelliteTransformer
import tensorflow as tf
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import config
from deepdiff import DeepDiff
import pymap3d as pm
import math
import tensorflow_addons as tfa
import threading
from planners.AbstractRL import AbstractRL


# Utility functions
from planners import utils


class BaseRL(AbstractRL):

    def __init__(self, settings):
        super().__init__(settings)

        self.pool_size = 36  # config.cores
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings

        # Hyperparameters
        self.num_epochs = 1
        self.target_update_frequency = 5
        self.replay_batch_size = 128
        self.replay_frequency = 1
        self.buffer_init_size = 100
        self.clip_gradients = False

        # 3. Optimizers
        self.gamma = 0.9

        # 4. Metrics
        self.plot_frequency = 100
        self.sample_time_deltas = []
        self.sample_async_terms = []





















