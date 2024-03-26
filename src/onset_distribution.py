import matplotlib.pyplot as plt
import music21
import numpy as np
from pandas import DataFrame
import matplotlib.ticker as ticker
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET

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
    ax.set_xlim(0, sig[0])
    ax.set_ylim(0, 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))



def compute_distribution(score_path, beat_locations):
    """
    Computes the distribution in one piece of (sounded) onsets on the beats given predefined beat locations.
    Returns a numpy array with the normalized frequencies
    """

    sample_score = music21.converter.parse(score_path)
    # sample_score.show("midi")
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


def get_average_distribution_given_time_signature(corpus, time_signature):
    '''
    This function extracts all pieces with a specific time signature (ex. 4/4) from a corpus and
    computes the average distribution for them
    '''
    matching_pieces = [piece for piece in os.listdir(corpus) if extract_time_signature_from_xml(corpus / piece / 'xml_score.musicxml') == time_signature]
    score_paths = [corpus / piece / "midi_score.mid" for piece in matching_pieces]
    print(f'Time signature {time_signature} is present in {len(score_paths)} pieces.')
    compute_average_distribution(score_paths, sig=time_signature, subdivision=time_signature[-1])
    
def calculate_inter_onset_intervals(annotation:pd.DataFrame, beats:int) -> pd.Series:
    '''
    annotation: dataframe of the annotation file
    beats: indication how many beats in a bar. e.g. 3 in 3/4
    output: series containing time passed since last downbeat 
    ex. [1,2,3,4,5,6] -> [1,2,3,1,2,3] in a 3/4 pattern
    '''
    
    # OLD IMPLEMENTATION
    # create array containing time of last downbeat
    # bar_onsets = beats * [0]
    # for time, beat in zip(annotation.iloc[:,0], annotation.iloc[:,2]):
    #     # time of last downbeat
    #     if 'db' in beat:
    #         bar_onsets += beats * [time]

    # # remove last downbeats (list gets too long)
    # bar_onsets = bar_onsets[:-beats]
    # # calculate time passed since last downbeat
    # interval_timings = annotation.iloc[:,0] - pd.Series(bar_onsets)
    
    interval_timings = [annotation.iloc[0,0]]+[annotation.iloc[i,0] - annotation.iloc[i-1,0] for i in range(1,len(annotation))]
    interval_timings = pd.Series(interval_timings)
    
    return interval_timings

def match_regex_for_series(srs:pd.Series, pattern) -> list:
    matched_lines = []
    for idx, info in enumerate(srs):
        match = re.search(pattern, info)
        if match:
            matched_lines.append((idx, match.group(1)))
    return matched_lines

def extract_time_signatures_from_annotation(df: pd.DataFrame | str) -> list:
    # find all time signatures in beat info series
    if isinstance(df, str):
        df = pd.read_table(df, header=None)
    pattern = r"^db,([0-9]+/[0-9]+)"
    matched_lines = match_regex_for_series(df.iloc[:,2], pattern)
    time_signatures = []
    for line in matched_lines:
        idx, info = line
        ts = info.split('/')
        ts = [int(x) for x in ts]
        time_signatures.append((idx, ts))
    return time_signatures

def extract_time_signature_from_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()

    # Find the time signature element
    time_signature_element = root.find(".//time")

    if time_signature_element is not None:
        beats_element = time_signature_element.find("beats")
        beat_type_element = time_signature_element.find("beat-type")

        if beats_element is not None and beat_type_element is not None:
            beats = beats_element.text
            beat_type = beat_type_element.text
            return (int(beats), int(beat_type))
        else:
            return "Time signature not found in the XML file."
    else:
        return "Time signature not found in the XML file."
    
def normalize_interval(srs:pd.Series, beats:int) -> pd.Series:
    ones = srs.index % beats == 1
    normalization_factor = srs.loc[ones].median()
    normalized_series = srs / normalization_factor
    return normalized_series
        
