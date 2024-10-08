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

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

data_folder = r""

# Hardcoded settings
REJECT_OTHERS = False  # Set to True to ignore anything that is not explicitly listed in the 'groups' dictionary below, or False if you would prefer to include 'Others' as a class
PROBABILITY_THRESHOLD = 0.7  # In the second round of cross-validation only predictions with confidence greater than this will be included.
MAX_SIZE = 999  # The maximum size a tracked object is allowed to have to be included in training. This is just here to investigate how the classifier works on small / far away animals that never get close to the camera to be well resolved. Set to 7 to cut out about half of the tracks, or a big value (999) to keep everything.
NUM_TREES = 100  # Number of trees in the random forest. 100 seems to be plenty.
MAX_TREE_DEPTH = 6  # Maximum tree depth. Values between 4 and 8 seem OK, with 6 a good balance. But might depend on the nature of the classification (predators vs birds, predators vs everything, etc) and the composition of the training data.
NUM_FOLDS = 5  # Number of folds to use in cross-validation. 5 is fine if the dataset contains more than a few hundred samples of each class.

FEATURES = [
    "sqrt_area",
    "elongation",
    "peak_snr",
    "mean_snr",
    "fill_factor",
    "histogram_diff",
    "thermal_max",
    "thermal_std",
    "filtered_max",
    "filtered_min",
    "filtered_std",
]


def main():

    with open(os.path.join(data_folder, "features.npy"), "rb") as f:
        all_tags = np.load(f)
        all_features = np.load(f)
        all_ids = np.load(f)
    assert len(all_tags) == len(all_features)
    np.random.seed(0)

    fp_tags = ["water", "false-positive", "insect"]
    labels = ["animal", "false-positive", "vehicle"]
    ignore_labels = ["not identifiable", "other"]
    num_classes = len(labels)
    Y = []
    X = []
    groups = []
    for tag, feature, uid in zip(all_tags, all_features, all_ids):
        if tag in ignore_labels:
            continue
        if tag in fp_tags:
            Y.append(labels.index("false-positive"))
        elif tag == "vehicle":
            Y.append(labels.index("vehicle"))
        else:
            Y.append(labels.index("animal"))
        X.append(feature)
        groups.append(uid)

    # Random forest has lots of settings in addition to the ones here.
    # Would be good to run a grid search to find ideal values at some point before deployment.
    model = RandomForestClassifier(
        n_estimators=NUM_TREES,
        max_depth=MAX_TREE_DEPTH,
        class_weight="balanced",
    )

    # Run cross-validation
    kfold = GroupKFold(n_splits=NUM_FOLDS)
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_prob = np.empty([0, num_classes])
    fold = 0
    X = np.array(X)
    Y = np.array(Y)
    groups = np.array(groups)
    for train_index, test_index in kfold.split(X, Y, groups):

        fold += 1
        print(f"Cross validating, fold {fold} of {NUM_FOLDS}...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p_pred = model.predict_proba(
            X_test
        )  # Probabilities are useful for filtering and generating ROC curves

        actual_classes = np.append(actual_classes, y_test)
        predicted_classes = np.append(predicted_classes, y_pred)
        predicted_prob = np.append(predicted_prob, p_pred, axis=0)

    # Confusion matrix
    print_confusion(labels, actual_classes, predicted_classes, predicted_prob, None)
    print_confusion(labels, actual_classes, predicted_classes, predicted_prob, 0.7)

    # ROC curves (only available for binary classification)
    if num_classes == 2:

        # All predictions
        for i in range(2):
            RocCurveDisplay.from_predictions(
                actual_classes, predicted_prob[:, i], pos_label=i, name="ROC curve"
            )
            plt.grid()
            plt.show()

        # Ignoring low probabilities
        for i in range(2):
            RocCurveDisplay.from_predictions(
                actual_classes_masked,
                predicted_prob_masked[:, i],
                pos_label=i,
                name="ROC curve (masked)",
            )
            plt.grid()
            plt.show()

    # Train on everything to get feature importances
    print("Training on everything...")
    model.fit(X, Y)
    feat_import = model.feature_importances_

    print("Feature importances:")
    for i in range(len(FEATURES)):
        print(f"{i+1:3}   {FEATURES[i]:20} {100*feat_import[i]:.1f}%")

    inds = np.argsort(feat_import)
    print("Feature importances (ranked):")
    for i in range(len(FEATURES)):
        print(f"{i+1:3}   {FEATURES[inds[-1-i]]:20} {100*feat_import[inds[-1-i]]:.1f}%")


def print_confusion(
    labels, actual_classes, predicted_classes, predicted_prob, threshold
):
    if threshold is not None:
        mask = np.max(predicted_prob, axis=1) > threshold
        actual_classes_masked = actual_classes[mask]
        predicted_classes_masked = predicted_classes[mask]
        predicted_prob_masked = predicted_prob[mask]
        threshold = f"{threshold} C O N F"

    else:
        threshold = ""
        actual_classes_masked = actual_classes
        predicted_classes_masked = predicted_classes
        predicted_prob_masked = predicted_prob
    print(f"{threshold}  P R E D I C T E D")
    for i in range(len(labels)):

        print(labels[i], end="")
        total = np.sum(actual_classes == i)

        total = np.sum(actual_classes == i)
        for j in range(len(labels)):
            s = np.sum(
                np.logical_and(
                    actual_classes_masked == i, predicted_classes_masked == j
                )
            )
            print(f"{s:9} ({round(100*s/total)})", end="")
        print(f" | {total:9}")
    print("---------------------------------------")
    for j in range(len(labels)):
        s = np.sum(predicted_classes_masked == j)
        print(f"{s:9}", end="")
    print(f" | {len(actual_classes_masked):9}")


if __name__ == "__main__":
    main()


# How should track labels be grouped together?
groups = [
    ["rodent", "mustelid", "leporidae", "hedgehog", "possum", "cat"],
    #    ['bird','bird/kiwi'],                                          # Comment out this line (and make sure REJECT_OTHERS is set to False) to do binary classification (Predators vs everything else). Or enable this line and set REJECT_OTHERS to True to do Predators vs Birds, or False to do Predators vs Birds vs Everything else
]
