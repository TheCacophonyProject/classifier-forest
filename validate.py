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

# More hardcoded nastiness!
FEAT_LABELS = [
    "avg_sqrt_area",
    "avg_elongation",
    "avg_peak_snr",
    "avg_mean_snr",
    "avg_fill_factor",
    "avg_move_1",
    "avg_rel_move_1",
    "avg_rel_x_move_1",
    "avg_rel_y_move_1",
    "avg_move_3",
    "avg_rel_move_3",
    "avg_rel_x_move_3",
    "avg_rel_y_move_3",
    "avg_move_5",
    "avg_rel_move_5",
    "avg_rel_x_move_5",
    "avg_rel_y_move_5",
    "std_sqrt_area",
    "std_elongation",
    "std_peak_snr",
    "std_mean_snr",
    "std_fill_factor",
    "std_move_1",
    "std_rel_move_1",
    "std_rel_x_move_1",
    "std_rel_y_move_1",
    "std_move_3",
    "std_rel_move_3",
    "std_rel_x_move_3",
    "std_rel_y_move_3",
    "std_move_5",
    "std_rel_move_5",
    "std_rel_x_move_5",
    "std_rel_y_move_5",
    "max_sqrt_area",
    "max_elongation",
    "max_peak_snr",
    "max_mean_snr",
    "max_fill_factor",
    "max_move_1",
    "max_rel_move_1",
    "max_rel_x_move_1",
    "max_rel_y_move_1",
    "max_move_3",
    "max_rel_move_3",
    "max_rel_x_move_3",
    "max_rel_y_move_3",
    "max_move_5",
    "max_rel_move_5",
    "max_rel_x_move_5",
    "max_rel_y_move_5",
    "track_length",
]

# How should track labels be grouped together?
groups = [
    ["rodent", "mustelid", "leporidae", "hedgehog", "possum", "cat"],
    [
        "bird",
        "bird/kiwi",
    ],  # Comment out this line (and make sure REJECT_OTHERS is set to False) to do binary classification (Predators vs everything else). Or enable this line and set REJECT_OTHERS to True to do Predators vs Birds, or False to do Predators vs Birds vs Everything else
]
group_labels = ["pests", "birds"]
if not REJECT_OTHERS:
    groups.append(["other"])
    group_labels.append("other")
# Seed random number generator for repeatability
np.random.seed(0)

with open(os.path.join(data_folder, "train-new.pickle"), "rb") as f:
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
for i in range(num_samples):
    track_label = train["Y"][i]
    for j in range(num_classes):
        if track_label in groups[j]:
            y[i] = j
            counts[j] += 1
            break

# Display group info (should have a few hundred at least in each class for reliable classification)
print(f'{"class":9}   {"count":9}   {"labels"}')
for j in range(num_classes):
    print(f"{j:9}   {counts[j]:9}   {groups[j]}")
print(f'{"?":9}   {num_samples-np.sum(counts):9}   {"Other"}')

if REJECT_OTHERS:
    print("Getting rid of others")
    mask = y < num_classes
    X = X[mask, :]
    y = y[mask]
    I = I[mask]
    num_samples = X.shape[0]
else:
    # set to toher class
    mask = y >= num_classes
    y[mask] = len(groups) - 1

print(num_samples)


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
for train_index, test_index in kfold.split(X, y, I):

    fold += 1
    print(f"Cross validating, fold {fold} of {NUM_FOLDS}...")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    p_pred = model.predict_proba(
        X_test
    )  # Probabilities are useful for filtering and generating ROC curves

    actual_classes = np.append(actual_classes, y_test)
    predicted_classes = np.append(predicted_classes, y_pred)
    predicted_prob = np.append(predicted_prob, p_pred, axis=0)


# Confusion matrix
print("     P R E D I C T E D")
space = ""
print(f"{space:9}", end="")
for g_l in group_labels:
    print(f"{g_l:9}", end="")
print(f"total")
for i in range(num_classes):
    print(group_labels[i], end="")
    for j in range(num_classes):
        s = np.sum(np.logical_and(actual_classes == i, predicted_classes == j))
        print(f"{s:9}", end="")
    s = np.sum(actual_classes == i)
    print(f" | {s:9}")
print("---------------------------------------")
print("Total predictions")
for j in range(num_classes):
    s = np.sum(predicted_classes == j)
    print(f"{s:9}", end="")
print(f" | {num_samples:9}")


# Reject low confidence predictions
mask = np.max(predicted_prob, axis=1) > PROBABILITY_THRESHOLD
actual_classes_masked = actual_classes[mask]
predicted_classes_masked = predicted_classes[mask]
predicted_prob_masked = predicted_prob[mask]

# New confusion matrix
print("REJECT LOW CONFIDENCE     P R E D I C T E D")
print(f"{space:9}", end="")
for g_l in group_labels:
    print(f"{g_l:9}", end="")
print(f"total")
for i in range(num_classes):
    print(group_labels[i], end="")
    for j in range(num_classes):
        s = np.sum(
            np.logical_and(actual_classes_masked == i, predicted_classes_masked == j)
        )
        print(f"{s:9}", end="")
    s = np.sum(actual_classes_masked == i)
    print(f" | {s:9}")
print("---------------------------------------")
for j in range(num_classes):
    s = np.sum(predicted_classes_masked == j)
    print(f"{s:9}", end="")
print(f" | {len(actual_classes_masked):9}")

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
model.fit(X, y)
feat_import = model.feature_importances_

print("Feature importances:")
for i in range(len(FEAT_LABELS)):
    print(f"{i+1:3}   {FEAT_LABELS[i]:20} {100*feat_import[i]:.1f}%")

inds = np.argsort(feat_import)
print("Feature importances (ranked):")
for i in range(len(FEAT_LABELS)):
    print(f"{i+1:3}   {FEAT_LABELS[inds[-1-i]]:20} {100*feat_import[inds[-1-i]]:.1f}%")
