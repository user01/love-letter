
import argparse
import csv
import os

import torch

from loveletter.agents.random import AgentRandom
from loveletter.arena import Arena
from loveletter.agents.a3c import AgentA3C

PARSER = argparse.ArgumentParser(
    description='Run the arena with available agents')

PARSER.add_argument('--output', type=str, default='arena.results.csv',
                    help='Path to write arena results')

ARGS = PARSER.parse_args()

print('Starting arena')
A3C_PATH = os.path.join("models", "stated_2017-05-01T22:59:33.510476_best_0.45875")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


ARENA = Arena([
    # Place agents in this list as created
    # first in the tuple is the readable name
    # second is a lambda that ONLY takes a random seed. This can be discarded
    # if the the Agent does not require a seed
    ("A3C", lambda seed: AgentA3C(A3C_PATH, dtype, seed)),
    ("Random", lambda seed: AgentRandom(seed))
], 500)

print('Run the arena for: ', ARENA.csv_header())

with open(ARGS.output, 'w') as f:
    WRITER = csv.writer(f)
    WRITER.writerow(ARENA.csv_header())
    WRITER.writerows(ARENA.csv_results_lists())

print('Complete')
