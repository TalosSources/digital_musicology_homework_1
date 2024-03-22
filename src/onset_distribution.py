import matplotlib.pyplot as plt
import music21
from src.data import get_events_table_from_score
from pandas import DataFrame
import numpy as np

def compute_average_distribution(score_paths, sig=(4,4), subdivision=4):

    beat_locations = [i / subdivision for i in range(sig[0] * subdivision)]

    beat_frequencies = None
    for score_path in score_paths:
        freq = compute_distribution(score_path, beat_locations)
        if beat_frequencies is None: 
            beat_frequencies = freq
        else:
            beat_frequencies += freq

    beat_frequencies /= len(score_paths)

    fig, ax = plt.subplots()
    ax.plot(beat_locations, beat_frequencies, color='blue')
    ax.set_xlabel('Onset in Measure')
    ax.set_ylabel('')
    ax.set_title('Relative Frequency of Onset LLLLLocations')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, .5)


def compute_distribution(score_path, beat_locations):
    # first, measure level
    #for type, onset, tie in zip(events["event_type"], events["onset_in_measure"], events["tie_info"]):
    #    if type == "sounded" and (tie not in ["tie_continue", "tie_stop"]):
    #        occurrences[onset] = occurrences.get(onset, 0) + 1

    sample_score = music21.converter.parse(score_path)
    sample_score.show("midi")
    events = get_events_table_from_score(sample_score)

    onsets = filter_onsets(events, beat_locations)
    #print(f"unique onsets : {set(onsets)}")

    onset_count = 0
    beat_per_measure = 4

    beat_frequencies = []
    for loc in beat_locations:
        beat_frequencies.append((onsets.count(loc) / len(onsets)))
    
    return np.array(beat_frequencies)

    #fig, ax = plt.subplots()
    #ax.plot(beat_locations, beat_frequencies, color='blue')
    #ax.set_xlabel('Onset in Measure')
    #ax.set_ylabel('')
    #ax.set_title('Relative Frequency of Onset LLLLLocations')
    #ax.set_xlim(0, 4)
    #ax.set_ylim(0, .5)

def find_expressive_timing(): # TODO : arguments
    """
    This function should find where expressive timing happens given an annotations.txt file, and signature info
    A way to do it could be to find beats where the distance to the preceding and following beats aren't the same
    (indicating local slowing/accelerating)
    We could also find the average tempo of the performance and compare inter-beat distance to that average
    But that wouldn't be robust to a piece having several tempis or something
    It could return, for each beat of the measure, a number (percentage?) indicating how "likely" or "intense" expressive timings are there
    Also, I need (TODO) to support such analyses over several measures (for the above methods also)
    """

def filter_onsets(events : DataFrame, beat_locations):
    onsets = events[(events['event_type'] == "sounded") & (events['tie_info'] != "tie_stop")]['onset_in_measure']
    onsets = onsets[onsets.isin(beat_locations)]
    return onsets.tolist()