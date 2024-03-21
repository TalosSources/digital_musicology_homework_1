import matplotlib.pyplot as plt
import numpy as np


def plot_transfer_function(midi_beats, performance_beats, performance_type="time"):
    """
    Lolipop chart for performance time vs beats position
    """
    plt.plot(midi_beats, performance_beats)
    plt.xlabel("Midi Beats")
    if performance_type == "time":
        plt.ylabel("Performed Beats Position (S)")
    else:
        plt.ylabel("Performed Beats Velocity")
    plt.show()


def plot_average_transfer_function(
    midi_beats_list, performance_beats_list, performance_type="time"
):
    """
    Plot transfer function averaged over subcorpus
    """
    max_length = 0
    max_midi_beats = []
    for midi_beats in midi_beats_list:
        if len(midi_beats) > max_length:
            max_length = len(midi_beats)
            max_midi_beats = midi_beats

    sum_performance_beats = np.zeros(max_length)
    amount_performance_beats = np.zeros(max_length)
    for performance_beats in performance_beats_list:
        sum_performance_beats[: len(performance_beats)] += np.array(performance_beats)
        amount_performance_beats[: len(performance_beats)] += np.ones(
            len(performance_beats)
        )
    mean_performance_beats = sum_performance_beats / amount_performance_beats

    plot_transfer_function(max_midi_beats, mean_performance_beats, performance_type)
