import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, estimator_type="random", data_type="time"):
        """
        Args:
            estimator_type: str random, linear, etc.
            data_type: str time or velocity
        """
        self.estimator_type = estimator_type
        self.data_type = data_type

    def fit(
        self,
        bpm_list,
        midi_beats_list,
        midi_downbeats_list,
        performance_beats_list,
        performance_downbeats_list,
        velocity_beats_list,
        perf_velocity_beats_list,
    ):
        """
        Train estimator on training data
        """
        self.midi_beats_list = midi_beats_list
        if self.data_type == "time":
            self.bpm_list = bpm_list
            self.unperformed_beats_list = midi_beats_list
            self.unperformed_downbeats_list = midi_downbeats_list
            self.performed_beats_list = performance_beats_list
            self.performed_downbeats_list = performance_downbeats_list
        else:
            self.unperformed_beats_list = velocity_beats_list
            self.performed_beats_list = perf_velocity_beats_list

        if self.estimator_type == "random":
            self.fit_random()
        elif self.estimator_type == "linear":
            self.fit_linear()

        return self

    def estimate(
        self,
        bpm_list,
        midi_beats_list,
        midi_downbeats_list,
        performance_beats_list,
        performance_downbeats_list,
        velocity_beats_list,
        perf_velocity_beats_list,
    ):
        if self.estimator_type == "random":
            if self.data_type == "time":
                return self.random_estimate(midi_beats_list, midi_downbeats_list, bpm_list)
            else:
                return self.random_estimate(velocity_beats_list)
        if self.estimator_type == "linear":
            return self.linear_estimate(midi_beats_list)


    def random_estimate(self, unperformed_beats_list, unperformed_downbeats_list, bpm_list):
        beat_indices, _ = get_beat_indices(unperformed_beats_list, unperformed_downbeats_list, bpm_list)
        estimated_durations = []
        for indices, bpm in zip(beat_indices, bpm_list): # run over test pieces
            est = [] # estimated bpm for current piece
            for j in indices: # differentiate between beats in different positions in the measure
                ind = int(j)
                est.append(np.sqrt(self.var[ind]) * np.random.randn() + self.mean[ind] + bpm) # Compute random durations from a normal distribution with mean and variance obtained from the train sub-corpus
            estimated_durations.append(est)
        return estimated_durations
    

    '''def random_estimate(self, unperformed_beats_list):
        """
        Estimate beats as unperformed_value + random
        """
        estimated_beats_list = []
        for unperformed_beats in unperformed_beats_list:
            estimated_beats = []

            for unperformed_beat in unperformed_beats:
                estimated_beats.append(
                    unperformed_beat + self.random_std * np.random.randn(1).item()
                )

            estimated_beats_list.append(estimated_beats)
        return estimated_beats_list'''

    def linear_estimate(self, midi_beats_list):
        """
        Estimate beats as mean of linregs fitted during training
        """
        estimated_beats_list = []
        for midi_beats in midi_beats_list:
            midi_array = np.array(midi_beats)
            estimated_beats = np.zeros_like(midi_array)
            for linreg in self.estimators:
                estimated_beats += linreg.intercept + linreg.slope * midi_array
            estimated_beats = estimated_beats / len(self.estimators)
            estimated_beats_list.append(estimated_beats.tolist())
        return estimated_beats_list

    def fit_random(self):
        """
        Given training data, find mean and variance for random noise estimator
        """
        beat_indices, beats_per_measure = get_beat_indices(self.unperformed_beats_list, self.unperformed_downbeats_list, self.bpm_list)
        performance_beat_durations = get_beat_durations(self.performed_beats_list)
        mean_durations, variance_durations = get_mean_variance_durations(beat_indices, performance_beat_durations, beats_per_measure)

        self.mean = mean_durations
        self.var = variance_durations

    def fit_linear(self):
        """
        Given training data, fit linear predictors
        """
        estimators = []
        for midi_beats, performed_beats in zip(
            self.midi_beats_list, self.performed_beats_list
        ):
            linreg = ss.linregress(x=midi_beats, y=performed_beats)
            estimators.append(linreg)
        self.estimators = estimators


def get_estimator_predictions(
    train_beats_list_dict, test_beats_list_dict, estimator_type="random"
):
    time_estimator = Estimator(estimator_type=estimator_type, data_type="time")
    time_estimator = time_estimator.fit(**train_beats_list_dict)
    performance_beats_estimated_list = time_estimator.estimate(**test_beats_list_dict)

    velocity_estimator = Estimator(estimator_type=estimator_type, data_type="velocity")
    velocity_estimator = velocity_estimator.fit(**train_beats_list_dict)
    velocity_beats_estimated_list = velocity_estimator.estimate(**test_beats_list_dict)

    return performance_beats_estimated_list, velocity_beats_estimated_list


def get_beat_indices(midi_beats_list, midi_downbeats_list, bpm_list):
    '''Separarates beats depending on the position in the measure'''
    beat_indices = []
    for beats, downbeats, bpm in zip(midi_beats_list, midi_downbeats_list, bpm_list):
        bps = bpm / 60.0 # beats per second
        nb = bps * (downbeats[1] - downbeats[0]) # beats per measure
        indices = [((j-downbeats[0]) * bps) % nb for j in beats] # indices of beats: 0 is the downbeat, 1 is the beat after, etc.
        indices = [round(j) for j in indices] # round to integers
        indices.pop() # remove last beat from each piece (no duration given)
        beat_indices.append(indices)

    return beat_indices, int(nb)

def get_beat_durations(performance_beats_list):
    '''Returns performance beat durations'''
    beat_durations = []
    for beats in performance_beats_list:
        durations = [60.0 / (beats[j+1]-beats[j]) for j in range(len(beats)-1)] # compute beat durations in bpm
        beat_durations.append(durations)

    return beat_durations

def get_mean_variance_durations(beat_indices, bpm_list, performance_beat_durations, beats_per_measure):
    '''Get mean and variance of the difference in beat bpm from midi to performance, depending on the position in the measure'''
    mean_durations = [0] * beats_per_measure
    variance_durations = [0] * beats_per_measure
    for indices, bpm, durations in zip(beat_indices, bpm_list, performance_beat_durations):
        for j in range(beats_per_measure):
            filtered_durations = [durations[i]-bpm for i in range(len(durations)) if indices[i] == j]
            mean_durations[j] += np.mean(filtered_durations)
            variance_durations[j] += np.var(filtered_durations)

    mean_durations = [mean / len(beat_indices) for mean in mean_durations]
    variance_durations = [var / len(beat_indices) for var in variance_durations]

    return mean_durations, variance_durations

def get_random_estimate(means, variances, test_list, idx):
    beat_indices, beats_per_measure = get_beat_indices(test_list["midi_beats_list"], test_list["midi_downbeats_list"], test_list["bpm_list"])
    performance_beat_durations = get_beat_durations(test_list["performance_beats_list"])
    midi_beat_durations = get_beat_durations(test_list["midi_beats_list"])
    midi_beats = test_list["midi_beats_list"][idx]
    indices = beat_indices[idx]
    performance_durations = performance_beat_durations[idx]
    midi_durations = midi_beat_durations[idx]
    midi_bpm = test_list["bpm_list"][idx]
    estimated_durations = []
    for j in indices:
        ind = int(j)
        estimated_durations.append(np.sqrt(variances[ind]) * np.random.randn() + means[ind] + midi_bpm) # Compute random durations from a normal distribution with mean and variance obtained from the train sub-corpus

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(midi_beats[:-1], estimated_durations, label="Estimated bpm")
    ax.plot(midi_beats[:-1], performance_durations, label="Performance bpm")
    ax.plot(midi_beats[:-1], midi_durations, label="MIDI bpm")
    ax.legend()
    fig.savefig("test.png")