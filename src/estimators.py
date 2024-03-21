import numpy as np


def random_estimator(midi_beats):
    estimated_beats = []
    for beat in midi_beats:
        estimated_beats.append(beat + np.random.randn(1).item())
    return estimated_beats


def random_velocity_estimator(velocity_beats):
    estimated_beats = []
    for velocity in velocity_beats:
        estimated_beats.append(velocity + np.random.randn(1).item())
    return estimated_beats
