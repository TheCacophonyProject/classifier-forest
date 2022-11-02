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
MAX_FEATURES = 14  # Defauilt is sqrt of features (sqrt(52))
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
important_features = [
    "max-peak_snr",
    "std-mean_snr",
    "diff-move_1",
    "std-move_1",
    "std-fill_factor",
    "max-min_rel_speed",
]
groups = [
    ["rodent", "mustelid", "leporidae", "hedgehog", "possum", "cat", "wallaby", "pest"],
    ["bird", "bird/kiwi", "penguin"],
    ["human", "false-positive", "insect"],
    ["vehicle"],
]

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


group_labels = ["pests", "birds", "FP", "vehicle"]
group_labels = ["all", "FP"]

# TODO
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print("Model Performance")
    print("Average Error: {:0.4f} degrees.".format(np.mean(errors)))
    print("Accuracy = {:0.2f}%.".format(accuracy))

    return accuracy


#
# lOSS FUNCTION based of confident predictions
def confident_loss_func(ground_truth, predictions):
    probs = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1)
    non_confident = (ground_truth == predictions).sum()

    mask = probs > PROBABILITY_THRESHOLD
    masked_t = ground_truth[mask]
    masked_p = predictions[mask]
    confident_count = (masked_t == masked_p).sum()

    return confident_count / len(predictions)


def squashed_bird_loss_func(ground_truth, predictions):
    probs = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1)
    bird_mask = ground_truth == 1
    num_birds = bird_mask.sum()
    # print("num birds", num_birds, "from X", len(ground_truth))
    birds = predictions[bird_mask]
    squashed_birds = (birds == 0).sum()
    # print("birds predicted as pests", squashed_birds)
    squashed_percent = squashed_birds / len(ground_truth)
    # print("squashed_birds", squashed_percent)
    # print(ground_truth, predictions)
    # print(birds)
    # mask = probs > PROBABILITY_THRESHOLD
    # masked_t = ground_truth[mask]
    # masked_p = predictions[mask]
    confident_count = (ground_truth == predictions).sum()

    score = confident_count / len(predictions)
    # print("score is", score, squashed_percent, score - squashed_percent)
    return score - squashed_percent


#
# # loss_func will negate the return value of my_custom_loss_func,
# #  which will be np.log(2), 0.693, given the values for ground_truth
# #  and predictions defined below.
# score = make_scorer(my_custom_loss_func, greater_is_better=True, needs_proba=True)
# ground_truth = [[1, 1]]
# predictions = [0, 1]
# from sklearn.dummy import DummyClassifier
#
# clf = DummyClassifier(strategy="most_frequent", random_state=0)
# clf = clf.fit(ground_truth, predictions)
# loss(clf, ground_truth, predictions)
#
# score(clf, ground_truth, predictions)


def grid_search(args):
    param_grid = {
        "class_weight": ["balanced"],
        "bootstrap": [True],
        "max_depth": [40],  # probably NOne is best
        "max_features": [8, 10, 12],
        "min_samples_leaf": [1, 2, 3],
        "min_samples_split": [2, 4],
        "n_estimators": [100],
    }
    X, y, I, counts, num_classes = load_data(args.data_file, groups)
    rf = RandomForestClassifier()
    loss = make_scorer(
        squashed_bird_loss_func, greater_is_better=True, needs_proba=True
    )

    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, n_jobs=-1, verbose=3, scoring=loss
    )
    # grid_search = RandomizedSearchCV(
    #     estimator=rf,
    #     param_distributions=param_grid,
    #     n_iter=100,
    #     cv=3,
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1,
    #     return_train_score=True,
    #     scoring=loss,
    # )

    grid_search.fit(X, y)
    print("Grid search best params", grid_search.best_params_)
    results = grid_search.cv_results_["params"]
    scores = grid_search.cv_results_["mean_test_score"]

    # for k, v in grid_search.cv_results_.items():
    #     print("k is", k)
    #     print(v)
    for r, s in zip(results, scores):
        print("Params:", r, " Score: ", s)
    best_grid = grid_search.best_estimator_
    # best_grid.fit(X, y)

    # not much point without a test set
    # y_pred = best_grid.predict(X)
    # p_prob = best_grid.predict_proba(X)
    #
    # show_confusion(group_labels, y, y_pred, 0)
    #
    # mask = np.max(p_prob, axis=1) > PROBABILITY_THRESHOLD
    # actual_classes_masked = y[mask]
    # predicted_classes_masked = y_pred[mask]
    # # predicted_prob_masked = predicted_prob[mask]
    # print("REJECT LOW CONFIDENCE     P R E D I C T E D")
    # confusion_matrix_confident = show_confusion(
    #     group_labels, actual_classes_masked, predicted_classes_masked, 0
    # )


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


def feature_mask():
    mask = np.arange(len(ALL_FEATURES))
    mask = mask != -1
    feature_indexes = []
    for f in important_features:
        feature_indexes.append(ALL_FEATURES.index(f))
        print("using", f)
    feature_indexes = np.array(feature_indexes)
    mask[feature_indexes] = False
    return feature_indexes
    # mask


def train(args):
    X, y, I, counts, num_classes = load_data(args.data_file, groups)
    num_samples = X.shape[0]

    # Random forest has lots of settings in addition to the ones here.
    # Would be good to run a grid search to find ideal values at some point before deployment.
    model = RandomForestClassifier(
        n_estimators=NUM_TREES,
        max_depth=MAX_TREE_DEPTH,
        class_weight="balanced",
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
    )
    # Run cross-validation
    kfold = GroupKFold(n_splits=NUM_FOLDS)
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_prob = np.empty([0, num_classes])
    fold = 0
    X_shuffled, y_shuffled, groups_shuffled = shuffle(X, y, I, random_state=0)

    # subset = 20000
    # X_shuffled = X_shuffled[:subset]
    # y_shuffled = y_shuffled[:subset]
    # groups_shuffled = groups_shuffled[:subset]
    f_mask = feature_mask()
    X_shuffled = np.take(X_shuffled, f_mask, axis=1)
    global ALL_FEATURES
    ALL_FEATURES = important_features
    for train_index, test_index in kfold.split(X_shuffled, y_shuffled, groups_shuffled):

        fold += 1
        print(f"Cross validating, fold {fold} of {NUM_FOLDS}...")

        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        p_pred = model.predict_proba(
            X_test
        )  # Probabilities are useful for filtering and generating ROC curves
        actual_classes = np.append(actual_classes, y_test)
        predicted_classes = np.append(predicted_classes, y_pred)
        predicted_prob = np.append(predicted_prob, p_pred, axis=0)
    print("     P R E D I C T E D")
    confusion_matrix = show_confusion(
        group_labels, actual_classes, predicted_classes, num_samples
    )

    mask = np.max(predicted_prob, axis=1) > PROBABILITY_THRESHOLD
    actual_classes_masked = actual_classes[mask]
    predicted_classes_masked = predicted_classes[mask]
    predicted_prob_masked = predicted_prob[mask]
    print("REJECT LOW CONFIDENCE     P R E D I C T E D")
    confusion_matrix_confident = show_confusion(
        group_labels, actual_classes_masked, predicted_classes_masked, num_samples
    )

    # ROC curves (only available for binary classification)
    # if num_classes == 2:
    #
    #     # All predictions
    #     for i in range(2):
    #         RocCurveDisplay.from_predictions(
    #             actual_classes, predicted_prob[:, i], pos_label=i, name="ROC curve"
    #         )
    #         plt.grid()
    #         plt.show()
    #
    #     # Ignoring low probabilities
    #     for i in range(2):
    #         RocCurveDisplay.from_predictions(
    #             actual_classes_masked,
    #             predicted_prob_masked[:, i],
    #             pos_label=i,
    #             name="ROC curve (masked)",
    #         )
    #         plt.grid()
    #         plt.show()
    if args.permutation:
        test_i = int(len(X_shuffled) * 0.8)
        train_X = X_shuffled[:test_i]
        train_y = y_shuffled[:test_i]

        test_X = X_shuffled[test_i:]
        test_y = y_shuffled[test_i:]
        model.fit(train_X, train_y)
        r = permutation_importance(model, test_X, test_y, n_repeats=5, random_state=0)
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(
                    f"{ALL_FEATURES[i]:<8} "
                    f"{r.importances_mean[i]:.3f}"
                    f" +/- {r.importances_std[i]:.3f}"
                )

    if args.show_features or args.save_file:
        print("Fitting to all data")
        model.fit(X_shuffled, y_shuffled)

    if args.show_features:
        show_features(model)
    if args.save_file:
        model_file = args.save_file.with_suffix(".joblib")
        print("Saving too", model_file)
        joblib.dump(model, model_file.resolve())
        model_meta = {
            "groups": list(groups),
            "labels": list(group_labels),
            "counts": list(counts),
            "confusion_matrix_confident": confusion_matrix_confident.tolist(),
            "confusion": confusion_matrix.tolist(),
            "time": time.time(),
            "hyperparams": {
                "num_trees": NUM_TREES,
                "max_tree_depth": MAX_TREE_DEPTH,
                "min_samples_split": MIN_SAMPLES_SPLIT,
                "min_smaples_leaf": MIN_SAMPLES_LEAF,
                "max_features": MAX_FEATURES,
            },
        }
        meta_file = args.save_file.with_suffix(".txt")
        with open(meta_file.resolve(), "w") as f:
            json.dump(
                model_meta,
                f,
                indent=4,
            )


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


def show_features(model):
    # Train on everything to get feature importances
    print("Training on everything...")
    # model.fit(X, y)
    feat_import = model.feature_importances_

    print("Feature importances:")
    for i in range(len(ALL_FEATURES)):
        print(f"{i+1:3}   {ALL_FEATURES[i]:20} {100*feat_import[i]:.1f}%")

    inds = np.argsort(feat_import)[::-1]
    print("Feature importances (ranked):")
    for i, index in enumerate(inds):
        print(f"{i+1:3}   {ALL_FEATURES[index]:20} {100*feat_import[index]:.1f}%")


def main():
    init_logging()
    args = parse_args()
    if args.grid_search:
        grid_search(args)
    else:
        train(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default="train-new.pickle",
        # type=str,
        help="Location of trianing data pickle file",
    )
    parser.add_argument(
        "-p",
        "--permutation",
        action="count",
        # type=str,
        help="Show permutation features",
    )
    parser.add_argument(
        "-f",
        "--show-features",
        action="count",
        # type=str,
        help="Show important features",
    )
    parser.add_argument(
        "-g",
        "--grid-search",
        action="count",
        # type=str,
        help="Do a grid search",
    )

    parser.add_argument(
        "-s",
        "--save-file",
        default="random_forest",
        type=str,
        help="Save to",
    )

    args = parser.parse_args()
    args.data_file = Path(args.data_file)
    if args.save_file is None or len(args.save_file) == 0:
        args.save_file = None
    else:
        args.save_file = Path(args.save_file)

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
