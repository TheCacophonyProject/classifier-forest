# Script to load previously extracted track features and perform
# cross-validation to generate confusion matrices, ROC curves, etc
#
# Jamie Heather, Carbon Critical, July 2022
# jamie@carboncritical.org

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import numpy as np
import os
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import time
import json
import logging
from pathlib import Path
import sys
import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from utils import FEAT_LABELS, EXTRA_FEATURES
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle
from collections import Counter

data_folder = r""

# Hardcoded settings
REJECT_OTHERS = True  # Set to True to ignore anything that is not explicitly listed in the 'groups' dictionary below, or False if you would prefer to include 'Others' as a class
PROBABILITY_THRESHOLD = 0.7  # In the second round of cross-validation only predictions with confidence greater than this will be included.
MAX_SIZE = 999  # The maximum size a tracked object is allowed to have to be included in training. This is just here to investigate how the classifier works on small / far away animals that never get close to the camera to be well resolved. Set to 7 to cut out about half of the tracks, or a big value (999) to keep everything.
NUM_TREES = 100  # Number of trees in the random forest. 100 seems to be plenty.
MAX_TREE_DEPTH = 40  # Maximum tree depth. Values between 4 and 8 seem OK, with 6 a good balance. But might depend on the nature of the classification (predators vs birds, predators vs everything, etc) and the composition of the training data.
NUM_FOLDS = 5  # Number of folds to use in cross-validation. 5 is fine if the dataset contains more than a few hundred samples of each class.
MIN_SAMPLES_SPLIT = 2  # Default 2
MIN_SAMPLES_LEAF = 2  # Default 1
EXTRA = ["avg", "std", "max", "min", "diff"]

ALL_FEATURES = []
for extra_lbl in EXTRA:
    for f in FEAT_LABELS:
        ALL_FEATURES.append(f"{extra_lbl}-{f}")
ALL_FEATURES.extend(EXTRA_FEATURES)
important_features = [
    "std-fill_factor",
    "max-peak_snr",
    "std-move_1",
    "max-fill_factor",
    "std-hist_diff",
    "diff-hist_diff",
    "max-hist_diff",
    "min-hist_diff",
    "diff-fill_factor",
    "max-sqrt_area",
    "std-mean_snr",
    "max-min_rel_speed",
    "min-fill_factor",
    "std-rel_move_1",
    "diff-rel_x_move_1",
    "diff-move_1",
    "std-sqrt_area",
    "avg-move_3",
    "diff-elongation",
    "diff-move_5",
    "std-min_speed_x",
    "max-max_speed_x",
    "avg-max_speed_y",
    "max-elongation",
    "diff-move_3",
    "max-rel_x_move_3",
]
#
# important_features = [
#     "max-peak_snr",
#     "std-mean_snr",
#     "diff-move_1",
#     "std-move_1",
#     "std-fill_factor",
#     "max-min_rel_speed",
# ]

# for all
# important_features = [
#     "speed_distance_ratio",
#     "max-peak_snr",
#     "diff-peak_snr",
#     "max-sqrt_area",
#     "std-fill_factor",
#     "diff-sqrt_area",
#     "std-peak_snr",
#     "avg-hist_diff",
#     "std-rel_y_move_1",
#     "max-elongation",
#     "diff-elongation",
#     "std-move_1",
#     "max-mean_snr",
#     "min-hist_diff",
#     "diff-hist_diff",
#     "max-fill_factor",
#     "burst_min",
#     "diff-mean_snr",
#     "std-mean_snr",
#     "std-elongation",
#     "diff-fill_factor",
#     "std-move_5",
#     "std-hist_diff",
#     "max-rel_y_move_1",
#     "max-hist_diff",
#     "std-sqrt_area",
#     "diff-min_rel_speed_x",
#     "std-max_speed_x",
#     "diff-move_5",
#     "std-rel_y_move_5",
#     "max-min_rel_speed_x",
#     "std-rel_move_1",
#     "burst_chance",
#     "diff-rel_x_move_1",
#     "std-avg_speed",
#     "std-min_speed",
#     "min-mean_snr",
#     "diff-min_rel_speed",
#     "std-min_speed_x",
#     "max-min_rel_speed",
#     "burst_per_frame",
#     "std-rel_x_move_1",
#     "std-rel_move_5",
#     "diff-rel_y_move_1",
#     "max-rel_move_1",
#     "std-min_rel_speed_y",
#     "diff-max_speed",
#     "birst_mean",
#     "max-rel_x_move_1",
#     "max-max_speed",
#     "diff-min_speed_x",
#     "max-move_5",
#     "diff-max_speed_x",
#     "std-min_rel_speed_x",
#     "std-move_3",
#     "max-rel_x_move_3",
#     "std-rel_x_move_5",
#     "max-move_1",
#     "std-avg_rel_speed_y",
#     "min-fill_factor",
#     "max-rel_move_5",
#     "burst_max",
#     "std-avg_rel_speed_x",
#     "min-sqrt_area",
#     "avg-sqrt_area",
#     "max-rel_move_3",
#     "max-avg_speed_y",
#     "diff-min_rel_speed_y",
#     "avg-mean_snr",
#     "speed_ratio",
#     "diff-avg_rel_speed",
#     "max-max_speed_y",
#     "max-max_speed_x",
#     "max-min_rel_speed_y",
#     "avg-fill_factor",
#     "diff-rel_x_move_3",
#     "avg-max_rel_speed",
#     "diff-rel_move_3",
#     "diff-rel_move_5",
#     # very close to 0
#     "max-max_rel_speed",
#     "min-max_speed_x",
#     "diff-move_3",
#     "min-rel_move_5",
#     "avg-rel_move_5",
#     "std-rel_y_move_3",
#     "avg-max_speed_y",
#     "avg-rel_y_move_5",
#     "total frames",
#     "std-max_rel_speed",
#     "min-move_5",
#     "avg-avg_rel_speed",
#     "max-move_3",
#     "diff-min_speed_y",
#     "diff-avg_speed",
#     "avg-max_speed_x",
#     "max-rel_y_move_3",
# ]
groups = [
    ["rodent", "mustelid", "leporidae", "hedgehog", "possum", "cat", "wallaby", "pest"],
    ["bird", "bird/kiwi", "penguin"],
    ["human", "false-positive", "insect"],
    ["vehicle"],
]
#
groups = [
    [
        "rodent",
        "mustelid",
        "leporidae",
        "hedgehog",
        "possum",
        "cat",
        "wallaby",
        "pest",
        "bird",
        "bird/kiwi",
        "penguin",
        "human",
        "vehicle",
    ],
    ["false-positive", "insect"],
]
# groups = [
# ["rodent", "mustelid", "leporidae", "hedgehog", "possum", "cat"],
# ["bird", "bird/kiwi", "penguin"],
# ["wallaby"],
# ["human"],
# Comment out this line (and make sure REJECT_OTHERS is set to False) to do binary classification (Predators vs everything else). Or enable this line and set REJECT_OTHERS to True to do Predators vs Birds, or False to do Predators vs Birds vs Everything else
# ]
# about 14 for al features is good
# MAX_FEATURES = 6  # Defauilt is sqrt of features (sqrt(52))

group_labels = ["pests", "birds", "FP", "vehicle"]
group_labels = ["all", "FP"]



def load_data(data_file, groups):
    # Seed random number generator for repeatability
    np.random.seed(0)

    with open(data_file, "rb") as f:
        train = pickle.load(f)

    # Reject any tracks where the max size (feature #34) is too large
    mask = train["X"][:, 34] < MAX_SIZE
    train["X"] = train["X"][mask, :]
    train["Y"] = train["Y"][mask]
    train["I"] = train["I"][mask]
    print(f"Keeping {100*np.mean(mask):.1f}% of samples based on max size")

    X = train["X"]
    I = train["I"]
    num_samples = X.shape[0]
    num_feats = X.shape[1]
    num_classes = len(groups)

    # Get group indices and counts
    y = num_classes * np.ones(
        num_samples
    )  # Anything that doesn't match one of the groups defined above will be given a high label, and we'll decide below whether to keep them or not
    counts = np.zeros(num_classes + 1)
    other_labels = set()
    for i in range(num_samples):
        track_label = train["Y"][i]
        found = False
        for j in range(num_classes):
            if track_label in groups[j]:
                found = True
                y[i] = j
                counts[j] += 1
                break

        if not found:
            other_labels.add(track_label)
    # Display group info (should have a few hundred at least in each class for reliable classification)
    print(f'{"class":9}   {"count":9}   {"labels"}')
    for j in range(num_classes):
        print(f"{j:9}   {counts[j]:9}   {groups[j]}")
    print(
        f'{num_classes:9}   {num_samples-np.sum(counts):9}   {"Other"}({other_labels})'
    )
    # print(other_labels)
    if REJECT_OTHERS:
        print("Getting rid of others")
        mask = y < num_classes
        X = X[mask, :]
        y = y[mask]
        I = I[mask]
        num_samples = X.shape[0]
    else:
        # set to toher class

        groups.append(["other"])
        group_labels.append(f"other")
        mask = y >= num_classes
        other_labels = set(y[mask])
        y[mask] = len(groups) - 1
        num_classes += 1
    return X, y, I, counts, num_classes


def show_confusion(group_labels, actual_classes, predicted_classes, num_samples):
    # Confusion matrix
    num_classes = len(group_labels)
    print("     P R E D I C T E D")
    space = ""
    print(f"{space:12}", end="")
    for g_l in group_labels:
        print(f"{g_l:13}", end="")
    print("")
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        print(group_labels[i].ljust(9, " "), end="")
        total = np.sum(actual_classes == i)

        for j in range(num_classes):
            s = np.sum(np.logical_and(actual_classes == i, predicted_classes == j))
            if total == 0:
                percent = 0
            else:
                percent = round(s / total * 100, 1)
            formatted = f"{s:5} ( {percent}% )"
            print(f"{formatted:13}", end="")
            confusion_matrix[i][j] = s

        print(f" | {total:9}")
    print("---------------------------------------")
    print("Total predictions")
    for j in range(num_classes):
        s = np.sum(predicted_classes == j)
        print(f"{s:9}", end="")
    print(f" | {num_samples:9}")
    return confusion_matrix
def accuracy(data_file, model_file):
    meta_file = model_file.with_suffix(".txt")
    hyper_params = json.load(open(meta_file, "r"))
    X, y, I, counts, num_classes = load_data(data_file, hyper_params["groups"])

    model = joblib.load(model_file)
    predicted_classes = model.predict(X)
    predicted_prob = model.predict_proba(
        X
    )
    num_samples = X.shape[0]

    print("     P R E D I C T E D")
    confusion_matrix = show_confusion(
        hyper_params["labels"], y, predicted_classes, num_samples
    )

    mask = np.max(predicted_prob, axis=1) > PROBABILITY_THRESHOLD
    actual_classes_masked = y[mask]
    predicted_classes_masked = predicted_classes[mask]
    predicted_prob_masked = predicted_prob[mask]
    print("REJECT LOW CONFIDENCE     P R E D I C T E D")
    confusion_matrix_confident = show_confusion(
        hyper_params["labels"], actual_classes_masked, predicted_classes_masked, num_samples
    )

def main():
    init_logging()
    args = parse_args()
    accuracy(args.data_file, args.model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default="train-new.pickle",
        # type=str,
        help="Location of trianing data pickle file",
    )

    parser.add_argument(
        "model",
        # type=str,
        help="Model to load",
        )
    args = parser.parse_args()
    args.data_file = Path(args.data_file)
    args.model = Path(args.model)


    return args


def init_logging(timestamps=False):
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    if timestamps:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    main()
