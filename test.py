from src.data import *
from src.plots import *
from src.estimators import *
from src.onset_distribution import *

df, json_data = get_dataset_metadata("Bach")

# Code for random estimator
beats_list_dict = create_midi_performance_pairs(df, json_data, "4/4")
get_beat_indices(beats_list_dict["midi_beats_list"],beats_list_dict["midi_downbeats_list"], beats_list_dict["bpm_list"])
train_beats_list_dict, test_beats_list_dict = train_test_split(beats_list_dict, "4/4", test_size=0.2)
get_random_est_prediction(train_beats_list_dict, test_beats_list_dict)