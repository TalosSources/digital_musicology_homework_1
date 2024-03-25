from src.data import *
from src.plots import *
from src.estimators import *
from src.onset_distribution import *

df, json_data = get_dataset_metadata("Bach")

# Code for random estimator
beats_list_dict = create_midi_performance_pairs(df, json_data, "4/4")
get_beat_indices(beats_list_dict["midi_beats_list"],beats_list_dict["midi_downbeats_list"], beats_list_dict["bpm_list"])