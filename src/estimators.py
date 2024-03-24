import numpy as np
import scipy.stats as ss


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
        midi_beats_list,
        velocity_beats_list,
        performance_beats_list,
        perf_velocity_beats_list,
    ):
        """
        Train estimator on training data
        """
        self.midi_beats_list = midi_beats_list
        if self.data_type == "time":
            self.unperformed_beats_list = midi_beats_list
            self.performed_beats_list = performance_beats_list
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
        midi_beats_list,
        velocity_beats_list,
        performance_beats_list,
        perf_velocity_beats_list,
    ):
        if self.estimator_type == "random":
            if self.data_type == "time":
                return self.random_estimate(midi_beats_list)
            else:
                return self.random_estimate(velocity_beats_list)
        if self.estimator_type == "linear":
            return self.linear_estimate(midi_beats_list)

    def random_estimate(self, unperformed_beats_list):
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
        return estimated_beats_list

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
        Given training data, find std for random noise estimator
        """
        self.random_std = 1  # TODO

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

def get_random_est_prediction(train_list, test_list):
    beat_indices = get_beat_indices(train_list["midi_beats_list"], train_list["midi_downbeats_list"])
    midi_beat_durations, performance_beat_durations = get_beat_durations(beat_indices, train_list["midi_beats_list"],
                                                        train_list["performance_beats_list"])
    mean_durations = get_mean_durations(beat_indices, peformance_beat_durations)
    variance_durations = get_variance_durations(beat_indices, performance_beat_durations)

    # Get estimated performance tempo for a random test piece
    np.random.seed(1)
    idx = np.random.randint(0, len(test_list["midi_beats_list"]))
    get_random_estimate(mean_durations, variance_durations, test_list, idx)

def get_beat_indices(midi_beats_list, midi_downbeats_list):
    pass

def get_beat_durations(beat_indices, midi_beats_list, performance_beats_list):
    pass

def get_mean_durations(beat_indices, peformance_beat_durations):
    pass

def get_variance_durations(beat_indices, performance_beat_durations):
    pass

def get_random_estimate(mean, variance, test_list, idx):
    pass