import numpy as np
import pandas as pd
import os
import seaborn as sns
import pickle
import music21
from music21 import midi, note, stream, instrument, meter, key
import itertools
import random

from fractions import Fraction
from collections import defaultdict, Counter
from iteration_utilities import deepflatten #flatten nested lists

import string

dependencies = './asap-dataset/'

score_path = 'Bach/Fugue/bwv_846/midi_score.mid'

sample_score_tmp = music21.converter.parse(os.path.join(dependencies, score_path))
sample_score_tmp.show("midi")