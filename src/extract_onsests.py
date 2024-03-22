import itertools
import os
import pickle
import random
import string
from collections import Counter, defaultdict
from fractions import Fraction

import music21
import numpy as np
import pandas as pd
import seaborn as sns
from iteration_utilities import deepflatten  # flatten nested lists
from music21 import instrument, key, meter, midi, note, stream

dependencies = "./asap-dataset/"

score_path = "Bach/Fugue/bwv_846/midi_score.mid"

sample_score_tmp = music21.converter.parse(os.path.join(dependencies, score_path))
sample_score_tmp.show("midi")
