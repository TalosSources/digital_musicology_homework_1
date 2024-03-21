import json
import math
from pathlib import Path

import music21
import pandas as pd
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
DATASET_PATH = ROOT_PATH / "data" / "asap-dataset"


def get_dataset_metadata(composer):
    """
    Get subcorpus based on composer name

    Args:
        composer: str with composer name

    Returns:
        df: pd.DataFrame with metainformation for the chosen subcorpus
        json_data: dict with annotations
    """
    df = pd.read_csv(DATASET_PATH / "metadata.csv")
    df = df.loc[df["composer"] == composer]

    with open(DATASET_PATH / "asap_annotations.json") as json_file:
        json_data = json.load(json_file)

    return df, json_data


def get_midi_performance_pairs(df, json_data):
    """
    Loads pairs of midi beats and its performed version

    Args:
        df: pd.DataFrame with metainformation for the chosen subcorpus
        json_data: dict with annotations

    Returns:
        midi_beats_list: list(list) of midi beats
        velocity_beats_list: list(list) of velocities for each beat in performance versions
        performance_beats_list: list(list) of corresponding performance versions
    """
    if (ROOT_PATH / "data" / "midi_beats_list.json").exists():
        with open(ROOT_PATH / "data" / "midi_beats_list.json", "r") as f:
            midi_beats_list = json.load(f)
        with open(ROOT_PATH / "data" / "velocity_beats_list.json", "r") as f:
            velocity_beats_list = json.load(f)
        with open(ROOT_PATH / "data" / "performance_beats_list.json", "r") as f:
            performance_beats_list = json.load(f)

        return midi_beats_list, velocity_beats_list, performance_beats_list

    else:
        return create_midi_performance_pairs(df, json_data)


def create_midi_performance_pairs(df, json_data):
    """
    Creates pairs of midi beats and its performed version

    Args:
        df: pd.DataFrame with metainformation for the chosen subcorpus
        json_data: dict with annotations

    Returns:
        midi_beats_list: list(list) of midi beats
        velocity_beats_list: list(list) of velocities for each beat in performance versions
        performance_beats_list: list(list) of corresponding performance versions
    """
    midi_beats_list = []
    velocity_beats_list = []
    performance_beats_list = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        performance_path = row["midi_performance"]
        midi_beats = json_data[performance_path]["midi_score_beats"]
        performance_beats = json_data[performance_path]["performance_beats"]

        full_performance_path = ROOT_PATH / performance_path
        sample_score = music21.converter.parse(full_performance_path)
        velocity_beats = get_velocity_beats_from_score(midi_beats, sample_score)

        midi_beats_list.append(midi_beats)
        velocity_beats_list.append(velocity_beats)
        performance_beats_list.append(performance_beats)

    # save for later use
    with open(ROOT_PATH / "data" / "midi_beats_list.json", "w") as f:
        json.dump(midi_beats_list, f)
    with open(ROOT_PATH / "data" / "velocity_beats_list.json", "w") as f:
        json.dump(velocity_beats_list, f)
    with open(ROOT_PATH / "data" / "performance_beats_list.json", "w") as f:
        json.dump(performance_beats_list, f)

    return midi_beats_list, velocity_beats_list, performance_beats_list


# taken from the exercise session
def get_velocity_from_score(sample_score):
    rhythm_data_list = []
    for clef in sample_score.parts:
        global_onset = 0
        clef_name = "NotGiven"
        for measure in clef.getElementsByClass("Measure"):
            for event in measure.recurse():
                label = ""
                velocity = 0
                if isinstance(event, music21.note.Note):
                    label = "sounded"
                    velocity = event.volume.velocity
                if isinstance(event, music21.note.Rest):
                    label = "unsounded"
                try:
                    tie_info = "tie_" + event.tie.type
                except AttributeError:
                    tie_info = ""
                if label != "":
                    global_onset = ((measure.measureNumber - 1) * 4) + event.offset
                    rhythm_data_list.append(
                        (
                            clef_name,
                            measure.measureNumber,
                            label,
                            event.offset,
                            global_onset,
                            event.duration.quarterLength,
                            velocity,
                            tie_info,
                        )
                    )
    rhythm_data_df = pd.DataFrame(
        rhythm_data_list,
        columns=[
            "staff",
            "measure_number",
            "event_type",
            "onset_in_measure",
            "onset_in_score",
            "duration",
            "velocity",
            "tie_info",
        ],
    )
    return rhythm_data_df


def get_velocity_beats_from_score(midi_beats, sample_score):
    rhythm_data_df = get_velocity_from_score(sample_score)

    velocity_beats = []
    for beat in midi_beats:
        onsets_in_score = rhythm_data_df["onset_in_score"]
        condition = (onsets_in_score < beat + 0.5) & (onsets_in_score >= beat - 0.5)
        velocity = rhythm_data_df.loc[condition]["velocity"].mean()
        if math.isnan(velocity):
            velocity = 0
        velocity_beats.append(velocity)
    return velocity_beats
