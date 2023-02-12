#!/usr/bin/env python3
# Example using OddsRatioCriterion XNBoostClassifier using binary data

import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from odds_ratio_criterion import XNBoostClassifier

# parameters
splitting_criterion = "odds_ratio"
num_trees = 1
num_repetitions = 100
num_variables = 5
trajectory_length = 10
tree_depth = num_variables

target_names = ["False", "True"]

precision = np.zeros((num_repetitions, num_variables), dtype=float)
recall = np.zeros((num_repetitions, num_variables), dtype=float)
specificity = np.zeros((num_repetitions, num_variables), dtype=float)

print(f"Running {num_repetitions} iterations of model fitting to trajectories "
      f"with {trajectory_length} steps generated from a random "
      f"binary transition map with {num_variables} variables using "
      f"XNBoostClassifier with {num_trees} decision trees that were "
      f"constructed with {splitting_criterion} and maximal depth {tree_depth}.")

for repetition_index in range(0, num_repetitions):
    random_seed = np.random.RandomState(repetition_index)

    transition_map = np.arange(0, 2**num_variables, dtype=int)
    random_seed.shuffle(transition_map)

    index = random_seed.randint(0, 2**num_variables)
    trajectory = np.zeros((trajectory_length, 2*num_variables), dtype=bool)
    for step in range(0, trajectory_length):
        for number, offset in [(index, num_variables), (transition_map[index], 2*num_variables)]:
            for shift_index in range(int(number).bit_length()):
                trajectory[step, offset-shift_index-1] = (number >> shift_index) & 1
        index = transition_map[index]

    states = np.zeros((2**num_variables, 2*num_variables), dtype=bool)
    for step in range(0, 2**num_variables):
        for number, offset in [(step, num_variables), (transition_map[step], 2*num_variables)]:
            for shift_index in range(int(number).bit_length()):
                states[step, offset-shift_index-1] = (number >> shift_index) & 1

    # model fitting
    X_trajectory = trajectory[:, :num_variables]
    Y_trajectory = trajectory[:, num_variables:]

    # reporting
    X_states = states[:, :num_variables]
    Y_states = states[:, num_variables:]

    for variable_index in range(num_variables):

        model = XNBoostClassifier(
            DecisionTreeClassifier(
                max_depth=tree_depth,
                criterion=splitting_criterion,
            ),
            n_estimators=num_trees,
            loss_exponent=4,
            random_state=random_seed,
        )
        model.fit(X_trajectory, np.squeeze(Y_trajectory[:, variable_index]))

        report = classification_report(
            np.squeeze(Y_states[:, variable_index]),
            model.predict(X_states),
            target_names=target_names,
            output_dict=True
        )

        precision[repetition_index, variable_index] = report["True"]["precision"]
        recall[repetition_index, variable_index] = report["True"]["recall"]
        specificity[repetition_index, variable_index] = report["False"]["recall"]

for variable_index in range(num_variables):
    print(f"Variable {variable_index}")
    print(f"\tMedian precision: {np.median(precision[:, variable_index])}")
    print(f"\tMedian recall: {np.median(recall[:, variable_index])}")
    print(f"\tMedian specificity: {np.median(specificity[:, variable_index])}")
