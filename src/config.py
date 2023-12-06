import os
import pickle
from datetime import datetime
import tensorflow as tf
import platform
import json
import re
from keras.layers import TextVectorization


cores = 36

### Tensorflow Core
mixed_precision = False
if platform.system() != 'Darwin' and mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)



#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # path to /satplan
src_dir = os.path.join(root_dir, 'src')
results_dir = os.path.join(root_dir, 'results')
plots_dir = os.path.join(root_dir, 'plots')


##########################
# --> Action Decoder <-- #
##########################
num_actions = 11
sequence_len = 30


###############
# --> PPO <-- #
###############
ppo_episode_steps = 80
ppo_batch_size = 3
ppo_buffer_init_size = 1
ppo_train_policy_iterations = 100
ppo_train_value_iterations = 120
ppo_target_kl = 0.004
ppo_gamma = 0.99
ppo_lambda = 0.97

#
#      __      __                 _             _
#      \ \    / /                | |           | |
#       \ \  / /___    ___  __ _ | |__   _   _ | |  __ _  _ __  _   _
#        \ \/ // _ \  / __|/ _` || '_ \ | | | || | / _` || '__|| | | |
#         \  /| (_) || (__| (_| || |_) || |_| || || (_| || |   | |_| |
#          \/  \___/  \___|\__,_||_.__/  \__,_||_| \__,_||_|    \__, |
#                                                                __/ |
#                                                               |___/
#


def custom_standardization(input_data):
    stripped_html = tf.strings.regex_replace(input_data, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!"), ""
    )


actions = [str(idx) for idx in range(num_actions)]
special_tokens = ['[start]', '[end]']
vocab = special_tokens + actions
vocab_size = len(vocab)
tokenizer = TextVectorization(
    max_tokens=vocab_size + 2,  # Add 2 for padding and unknown tokens
    output_mode="int",
    output_sequence_length=sequence_len,
    standardize=custom_standardization,
)
tokenizer.set_vocabulary(vocab)
vocab = tokenizer.get_vocabulary()
vocab_size = len(vocab)

start_token_id = tokenizer(['[start]']).numpy()[0][0]
padding_token_id = tokenizer(['']).numpy()[0][0]
end_token_id = tokenizer(['[end]']).numpy()[0][0]
id2token = dict(enumerate(tokenizer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


def encode(input):
    encoded_input = tokenizer(input)
    return encoded_input.numpy()


@tf.function
def encode_tf(input):
    encoded_input = tokenizer(input)
    return encoded_input