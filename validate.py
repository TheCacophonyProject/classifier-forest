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
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import argparse
import time
import json

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
    "thermal_min",
    "thermal_std",
    "filtered_max",
    "filtered_min",
    "filtered_std",
]


USED_FEATURES = list(np.arange(len(FEATURES)))
hist_index = FEATURES.index("histogram_diff")
USED_FEATURES.remove(hist_index)

USED_FEATURES = np.array(USED_FEATURES)
from sklearn.model_selection import GridSearchCV


def grid_search(x_train, y_train):

    print("DOIng a grid search")
    param_grid = {
        # 'n_estimators': [ 50, 100, 150,200],
        # 'max_features': ['sqrt', 'log2', None],
        # 'max_depth': [3, 6, 9],
        "max_leaf_nodes": [3, 6, 9],
    }

    grid_search = GridSearchCV(RandomForestClassifier(n_jobs=8), param_grid=param_grid)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_estimator_)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-search", help="Model to load and do preds", action="store_true"
    )
    parser.add_argument(
        "--save-file", help="Model to load and do preds", default="model.pkl"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    save_file = Path(args.save_file)
    print("Loading features.npy")
    with open(os.path.join(data_folder, "features.npy"), "rb") as f:
        all_tags = np.load(f)
        all_features = np.load(f)
        all_ids = np.load(f)
        all_track_ids = np.load(f)
    assert len(all_tags) == len(all_features)
    np.random.seed(0)

    fp_tags = ["water", "false-positive", "insect"]
    labels = ["animal", "false-positive"]
    labels = ["rodent", "mustelid", "animal", "false-positive"]
    ignore_labels = ["not identifiable", "other"]
    num_classes = len(labels)
    Y = []
    X = []
    groups = []
    tags_used = set()
    remapped = {
        "rat": "rodent",
        "mouse": "rodent",
        "ferret": "mustelid",
        "weasel": "mustelid",
        "stoat": "mustelid",
    }
    for tag, feature, uid in zip(all_tags, all_features, all_ids):
        re_tag = remapped.get(tag, tag)
        if re_tag in ignore_labels:
            continue
        tags_used.add(str(tag))
        if re_tag in fp_tags:
            Y.append(labels.index("false-positive"))
        # elif tag == "vehicle":
        # Y.append(labels.index("vehicle"))
        elif re_tag in labels:
            Y.append(labels.index(re_tag))
        else:
            Y.append(labels.index("animal"))
        X.append(feature)
        groups.append(uid)

    for i, f in enumerate(FEATURES):
        if i not in USED_FEATURES:
            print("Exclidng feature ", f)
    tags_used = list(tags_used)
    tags_used.sort()
    print("Using tags ", tags_used)
    if args.grid_search:
        grid_search(np.array(X), np.array(Y))
        return
    # Random forest has lots of settings in addition to the ones here.
    # Would be good to run a grid search to find ideal values at some point before deployment.
    model = RandomForestClassifier(
        n_estimators=NUM_TREES,
        max_depth=MAX_TREE_DEPTH,
        class_weight="balanced",
        n_jobs=8,
    )

    # Run cross-validation
    kfold = GroupKFold(n_splits=NUM_FOLDS, shuffle=True)
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_prob = np.empty([0, num_classes])

    track_actual_classes = np.empty([0], dtype=int)
    track_predicted_classes = np.empty([0], dtype=int)
    track_predicted_prob = np.empty([0, num_classes])
    print("Num classes", num_classes)
    fold = 0
    X = np.array(X)
    X = X[:, USED_FEATURES]
    Y = np.array(Y)
    groups = np.array(groups)
    for train_index, test_index in kfold.split(X, Y, groups):

        fold += 1
        print(f"Cross validating, fold {fold} of {NUM_FOLDS}...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test, y_tracks = (
            Y[train_index],
            Y[test_index],
            all_track_ids[test_index],
        )

        ids = all_ids[test_index]
        group_test = groups[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p_pred = model.predict_proba(X_test)
        # cant set number of classes manually so if a class isn't present it crashes
        track_prob, track_y, track_ids, track_predicted = track_accuracy(
            y_test, p_pred, y_tracks
        )
        track_actual_classes = np.append(track_actual_classes, track_y)
        track_predicted_classes = np.append(track_predicted_classes, track_predicted)
        track_predicted_prob = np.append(track_predicted_prob, track_prob, axis=0)

        actual_classes = np.append(actual_classes, y_test)
        predicted_classes = np.append(predicted_classes, y_pred)
        predicted_prob = np.append(predicted_prob, p_pred, axis=0)

    # Confusion matrix
    print("Track level confusion")
    print_confusion(
        labels,
        track_actual_classes,
        track_predicted_classes,
        track_predicted_prob,
        None,
    )
    print_confusion(
        labels, track_actual_classes, track_predicted_classes, track_predicted_prob, 0.7
    )
    print("Frame level confusion")
    print_confusion(labels, actual_classes, predicted_classes, predicted_prob, None)
    print_confusion(labels, actual_classes, predicted_classes, predicted_prob, 0.7)

    # # ROC curves (only available for binary classification)
    # if num_classes == 2:

    #     # All predictions
    #     for i in range(2):
    #         RocCurveDisplay.from_predictions(
    #             actual_classes, predicted_prob[:, i], pos_label=i, name="ROC curve"
    #         )
    #         plt.grid()
    #         plt.show()

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

    # Train on everything to get feature importances
    print("Training on everything...")
    model.fit(X, Y)
    feat_import = model.feature_importances_
    print("Feature importances:")
    for i, f_i in enumerate(USED_FEATURES):
        print(f"{i:3}   {FEATURES[f_i]:20} {100*feat_import[i]:.1f}%")

    inds = np.argsort(feat_import)
    print("Feature importances (ranked):")
    for i, ind in enumerate(inds):
        feature = FEATURES[USED_FEATURES[ind]]
        print(f"{i}   {feature:20} {100*feat_import[ind]:.1f}%")

    # save
    with save_file.open("wb") as f:
        pickle.dump(model, f, protocol=5)
    metadata = {}
    metadata["tags_used"] = tags_used
    metadata["fp_tags"] = fp_tags
    metadata["ignore_labels"] = ignore_labels
    metadata["labels"] = labels
    metadata["datetime2"] = time.time()
    with f.with_suffix(".json").open("w") as f:
        json.dump(metadata, f)


def track_accuracy(actual_classes, predicted_probs, track_ids):
    track_probs = {}
    for y, prob, track_id in zip(actual_classes, predicted_probs, track_ids):
        if track_id not in track_probs:
            track_probs[track_id] = {"probs": []}

        track_probs[track_id]["probs"].append(prob)
        track_probs[track_id]["y"] = y
        track_probs[track_id]["track_id"] = track_id
    probs = []
    tracks = []
    track_y = []
    track_predicted = []
    for track_id, item in track_probs.items():
        prob = np.mean(np.array(item["probs"]), axis=0)
        probs.append(prob)
        track_y.append(item["y"])
        tracks.append(item["track_id"])
        track_predicted.append(np.argmax(prob))

    return (
        np.array(probs),
        np.array(track_y),
        np.array(tracks),
        np.array(track_predicted),
    )


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
            if total == 0:
                percent = "0"
            else:
                percent = round(100 * s / total)

            print(f"{s:9} ({percent})", end="")
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
