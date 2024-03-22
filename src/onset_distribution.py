import matplotlib.pyplot as plt
import music21
import numpy as np
from pandas import DataFrame

from src.data import get_events_table_from_score


def compute_average_distribution(score_paths, sig=(4, 4), subdivision=4):
    """
    Computes the average distribution over several pieces of (sounded) onsets on the beats given predefined beat locations.
    Returns a numpy array with the normalized averaged frequencies
    """

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
    ax.plot(beat_locations, beat_frequencies, color="blue")
    ax.set_xlabel("Onset in Measure")
    ax.set_ylabel("")
    ax.set_title("Average Relative Frequency of Onset Locations")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 0.5)


def compute_distribution(score_path, beat_locations):
    """
    Computes the distribution in one piece of (sounded) onsets on the beats given predefined beat locations.
    Returns a numpy array with the normalized frequencies
    """

    sample_score = music21.converter.parse(score_path)
    sample_score.show("midi")
    events = get_events_table_from_score(sample_score)

    onsets = filter_onsets(events, beat_locations)
    # print(f"unique onsets : {set(onsets)}")

    # onset_count = 0 # commented for now because they are not used
    # beat_per_measure = 4

    beat_frequencies = []
    for loc in beat_locations:
        beat_frequencies.append((onsets.count(loc) / len(onsets)))

    return np.array(beat_frequencies)


def find_expressive_timing():  # TODO : arguments
    """
    This function should find where expressive timing happens given an annotations.txt file, and signature info
    A way to do it could be to find beats where the distance to the preceding and following beats aren't the same
    (indicating local slowing/accelerating)
    We could also find the average tempo of the performance and compare inter-beat distance to that average
    But that wouldn't be robust to a piece having several tempis or something
    It could return, for each beat of the measure, a number (percentage?) indicating how "likely" or "intense" expressive timings are there
    Also, I need (TODO) to support such analyses over several measures (for the above methods also)
    """


def filter_onsets(events: DataFrame, beat_locations):
    """
    Extracts the onsets from the events and filter to exclude:
    - unsounded events
    - tied events that started previously
    - events that aren't on the studied grid (we need to justify this assumption of discarding too_precise/fractional/off-measure events)
    """
    sounded_mask = events["event_type"] == "sounded"
    tie_mask = (events["tie_info"] != "tie_stop") & (
        events["tie_info"] != "tie_continue"
    )
    onsets = events[sounded_mask & tie_mask]["onset_in_measure"]
    onsets = onsets[onsets.isin(beat_locations)]
    return onsets.tolist()
