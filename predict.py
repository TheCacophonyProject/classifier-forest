from forestmodel import extract_features

import joblib
import numpy as np
from pathlib import Path
import sys
import argparse
import pickle
import json
from multiprocessing import Pool

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="Model to load",
    )
    parser.add_argument("cptv_dir", help="Dir to load")
    parser.add_argument("confusion_file", help="Save confusionstoo")

    args = parser.parse_args()
    args.cptv_dir = Path(args.cptv_dir)
    args.model = Path(args.model)
    args.confusion_file = Path(args.confusion_file)

    return args


def main():
    args = parse_args()
    print("Loading ", args.model)
    with args.model.open("rb") as f:
        model = pickle.load(f)
    meta_f = args.model.with_suffix(".json")
    with meta_f.open("r") as f:
        metadata = json.load(f)
    labels = metadata["labels"]
    files = list(args.cptv_dir.glob(f"**/*.cptv"))
    files.sort()
    model_results = {}
    y_true = []
    y_pred = []
    remapped = {
        "rat": "rodent",
        "mouse": "rodent",
        "ferret": "mustelid",
        "weasel": "mustelid",
        "stoat": "mustelid",
    }
    labels.append("None")
    with Pool(processes=4, initargs=(5,)) as pool:
        for result in pool.imap_unordered(extract_features, files):
            if result is None:
                print("Could not load file ", result)
                continue
            tags, features, _, track_ids, clip_ids = result
            for tag, feature, track_id in zip(tags, features, track_ids):
                tag = remapped.get(tag, tag)
                prediction = model.predict_proba([feature])
                y_true.append(labels.index(tag))
                assert len(prediction) == 1
                prediction = prediction[0]
                # print("prediction is ",np.round(100*prediction))

                max_i = np.argmax(prediction)
                max_p = prediction[max_i]
                if max_p >= 0.7:
                    y_pred.append(max_i)
                else:
                    y_pred.append(len(labels) - 1)
                # predictions = model_results.setdefault(track_id, {"predictions":[],"y_true":tag})
                # predictions["predictions"].append(prediction[0])
    # result = extract_features(cptv_file, human_tagged=False)
    # if result is None:
    #     print("Got no features")
    #     return
    # tags, features, ids, track_ids = result
    # model = joblib.load("model.pkl")
    # model_results = {}
    # assert len(tags) == len(track_ids)
    # for tag, feature, id, track_id in zip(tags, features, ids, track_ids):
    #     prediction = model.predict_proba([feature])
    #     predictions = model_results.setdefault(track_id, [])
    #     predictions.append(prediction[0])
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(args.confusion_file.with_suffix(".png"), format="png")
    np.save(str(args.confusion_file.with_suffix(".npy")), cm)
    # for k, v in model_results.items():
    #     pred = np.mean(v, axis=0)
    #     best_p = np.argmax(pred)
    #     best_lbl = labels[best_p]
    #     best_conf = round(pred[best_p] * 100)
    #     print(f"Prediction for track {k} is {best_lbl}:{best_conf}%")
    #     for p in v:
    #         print(np.round(100 * p))


# from tensorflow examples
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(24, 24))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    counts = cm.copy()
    threshold = counts.max() / 2.0

    # Normalize the confusion matrix.

    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)
    cm = np.uint8(np.round(cm * 100))

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if counts[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


if __name__ == "__main__":
    main()
