import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_transfer_function(
    axes,
    midi_beats,
    performance_beats,
    performance_beats_estimated_dict,
    performance_type="time",
):
    """
    Chart for performance time/velocity vs beats position
    """

    colors = ["#2c7bb6", "#fdae61", "#d7191c", "#abd9e9", "#ffffbf"]

    axes.plot(midi_beats, performance_beats, label="original", color=colors[0])

    # plot each estimator
    for i, (k, v) in enumerate(performance_beats_estimated_dict.items()):
        axes.plot(midi_beats, v, label=k, color=colors[i + 1])
    axes.set_xlabel("Midi Beats")
    if performance_type == "time":
        axes.set_ylabel("Performed Beats Position (S)")
    else:
        axes.set_ylabel("Performed Beats Velocity")
    axes.legend()
    return axes


def plot_average_transfer_function(
    axes,
    midi_beats_list,
    performance_beats_list,
    performance_beats_estimated_list_dict,
    performance_type="time",
):
    """
    Plot transfer function averaged over subcorpus
    """

    (
        max_midi_beats,
        mean_performance_beats,
        mean_performance_beats_estimated_dict,
    ) = average_over_subcorpus(
        midi_beats_list, performance_beats_list, performance_beats_estimated_list_dict
    )

    return plot_transfer_function(
        axes,
        max_midi_beats,
        mean_performance_beats,
        mean_performance_beats_estimated_dict,
        performance_type,
    )


def average_over_subcorpus(
    midi_beats_list, performance_beats_list, performance_beats_estimated_list_dict
):
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

    mean_performance_beats_estimated_dict = {}
    for (
        k,
        performance_beats_estimated_list,
    ) in performance_beats_estimated_list_dict.items():
        sum_performance_beats_estimated = np.zeros(max_length)
        amount_performance_beats_estimated = np.zeros(max_length)
        for performance_beats in performance_beats_estimated_list:
            sum_performance_beats_estimated[: len(performance_beats)] += np.array(
                performance_beats
            )
            amount_performance_beats_estimated[: len(performance_beats)] += np.ones(
                len(performance_beats)
            )
        mean_performance_beats_estimated = (
            sum_performance_beats_estimated / amount_performance_beats_estimated
        )
        mean_performance_beats_estimated_dict[k] = mean_performance_beats_estimated

    return max_midi_beats, mean_performance_beats, mean_performance_beats_estimated_dict


def plot_beat_frequencies(beat_locations, beat_frequencies, sig):
    fig, ax = plt.subplots()
    ax.plot(beat_locations, beat_frequencies, color="blue")
    ax.set_xlabel("Onset in Measure (in quarter notes)")
    ax.set_ylabel("")
    ax.set_title(
        f"Average Relative Frequency of Onset Locations\nBeethoven Sonatas in {sig}"
    )
    ax.set_xlim(0, 4 * sig[0] / sig[1])
    ax.set_ylim(0, beat_frequencies.max() * 1.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))


def plot_style_analysis(expr_by_style):
    """
    Expr by style may be a map between style name and expressiveness fp number
    """
    ...


def plot_composer_analysis(expr_by_composer):
    """
    Expr by style may be a map between composer name and expressiveness fp number
    """
    ...
