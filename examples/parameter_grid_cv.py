#!/usr/bin/env python3

import hashlib

import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import parallel_backend

import odds_ratio_criterion

# parameters
num_variables = 3
num_repetitions = 10
criteria = ["odds_ratio", "gini", "entropy"]
lengths = [x for x in range(16, 128 + 1, 16)]


class TransitionMap:
    '''A random state transition map generated over all possible states for a given number of binary variables.'''

    def __init__(self, variables_count, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        self._transition_map = np.arange(0, 2**variables_count, dtype=int)
        random_state.shuffle(self._transition_map)

        self._generate_states(variables_count)

    def __repr__(self):
        return "TransitionMap({transition_map})".format(transition_map=self._transition_map)

    def _generate_states(self, variables_count):
        self._states = np.zeros((self._transition_map.shape[0], 2 * variables_count), dtype=int)

        for step in range(0, self._states.shape[0]):
            for number, offset in [
                (step, self.variables_count),
                (self._transition_map[step], 2 * self.variables_count),
            ]:
                for shift_index in range(int(number).bit_length()):
                    self._states[step, offset - shift_index - 1] = (number >> shift_index) & 1

    @property
    def data(self):
        return self._states[:, : self.variables_count]

    @property
    def targets(self):
        return self._states[:, self.variables_count :]

    @property
    def variables_count(self):
        return self._states.shape[1] // 2

    @property
    def states_count(self):
        return self._states.shape[0]

    def __getitem__(self, key):
        return self._transition_map[key]


class TrajectorySplitter:
    '''Cross-validation generator that generates trajectories from a state transition map.'''

    def __init__(self, transition_map, trajectory_starting_points, trajectory_lengths):
        self._transition_map = transition_map
        self._trajectory_start_points = trajectory_starting_points
        self._trajectory_lengths = trajectory_lengths

        self._current_index = 0

    def __repr__(self):
        return "TrajectorySplitter({transition_map}, {starting_points}, {length})".format(
            transition_map=self._transition_map,
            starting_points=self._trajectory_start_points,
            length=self._trajectory_lengths,
        )

    def _generate_trajectory(self):
        index = self._trajectory_start_points[self._current_index % len(self._trajectory_start_points)]
        trajectory_length = self._trajectory_lengths[self._current_index // len(self._trajectory_start_points)]
        trajectory = np.zeros((trajectory_length,), dtype=int)
        for step in range(0, trajectory_length):
            trajectory[step] = index
            index = self._transition_map[index]
        return trajectory

    def trajectory_lengths_per_fold(self):
        result = []
        for index in range(len(self._trajectory_start_points) * len(self._trajectory_lengths)):
            result.append(self._trajectory_lengths[index // len(self._trajectory_start_points)])

        return result

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self._trajectory_start_points) * len(self._trajectory_lengths):
            training_indexes = self._generate_trajectory()
            testing_indexes = [x for x in range(self._transition_map.states_count)]

            self._current_index += 1
            return training_indexes, testing_indexes
        raise StopIteration


if __name__ == "__main__":
    model = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))

    hyperparameters = dict(
        estimator__criterion=criteria,
        estimator__max_depth=[None] + [x for x in range(1, num_variables + 1, 2)],
        estimator__max_leaf_nodes=[None] + [x for x in range(3, num_variables + 1, num_variables // 3)],
    )
    scores = ["precision_micro", "recall_micro"]

    transition_map = TransitionMap(num_variables, random_state=42)
    staring_points = np.random.RandomState(42).randint(low=0, high=transition_map.states_count, size=num_repetitions)
    cv_splitter = TrajectorySplitter(transition_map, staring_points, lengths)

    hasher = hashlib.sha256()
    params_str = str([num_repetitions, hyperparameters, cv_splitter])
    hasher.update(params_str.encode("ascii"))
    file_name = hasher.hexdigest()

    with parallel_backend("threading"):
        grid_cv = GridSearchCV(
            model,
            hyperparameters,
            scoring=scores,
            n_jobs=4,
            refit=False,
            cv=cv_splitter,
            verbose=4,
            pre_dispatch="2*n_jobs",
            error_score="raise",
            return_train_score=True,
        )

        hyperparameters_tuning = grid_cv.fit(transition_map.data, transition_map.targets)

        pd.DataFrame(hyperparameters_tuning.cv_results_).to_pickle(f"{file_name}.pkl")

        with open(f"{file_name}.txt", "w") as text:
            text.write(params_str)
