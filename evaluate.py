from forestmodel import extract_features

import joblib
import numpy as np
from pathlib import Path
import sys
import argparse
import logging
import json
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool

def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        default=None,
        help="Model to load and do preds",
    )
    parser.add_argument(
        "cptv_dir",
        default=None,
        help="Directory of cptvs",
    )
    parser.add_argument(
        "confusion",
        default=None,
        help="Confusion file",
    )

    args = parser.parse_args()
    args.model = Path(args.model)
    args.cptv_dir = Path(args.cptv_dir)
    args.confusion = Path(args.confusion)

    return args


fp_tags = ["water", "false-positive", "insect"]
ignore_labels = ["not identifiable", "other"]

def tag_to_labels(tag,labels):
    if tag in ignore_labels:
        return None
    if tag in fp_tags:
        return labels.index("false-positive")
    elif tag == "vehicle" and "vehicle" in labels:
        return labels.index("vehicle")
    else:
        return labels.index("animal")
# test stuff
def main():
    init_logging()
    args = parse_args()

    model = joblib.load(args.model)

    meta_file = args.model.with_suffix(".json")
    with meta_file.open("r") as t:
        # add in some metadata stats
        meta_data = json.load(t) 
    labels = meta_data["labels"]
    print("Model labels are",labels)
    y_true = []
    
    model_results = {}

    files = list(args.cptv_dir.glob(f"**/*.cptv"))
    all_tags = []
    all_features = []
    track_ids = []
    with Pool(processes=8) as pool:
        for result in pool.imap_unordered(extract_features, files):
            if result is None:
                continue
            tags, track_features , track_ids,clip_id = result
            for features,tag in zip(track_features,tags):
                all_tags.append(tag)
                all_features.append(np.array(features))
            track_ids.extend(track_ids)

    for tag, track_features,  track_id in zip(all_tags, all_features, track_ids):
        y =tag_to_labels(tag,labels)
        if y is None:
            continue
        prediction = model.predict_proba(track_features)
        model_results[track_id] = (prediction,y)

    labels.append("nothing")
    y_pred = []
    threshold = 0.7
    for k, v in model_results.items():
        pred = np.mean(v[0], axis=0)
        best_p = np.argmax(pred)
        prob = pred[best_p]
        best_lbl = labels[best_p]
        y_true.append(v[1])

        if prob>= threshold:
            y_pred.append(best_p)
        else:
            y_pred.append(len(labels)-1)        
        best_conf = round(pred[best_p] * 100)
        print(f"{v[1]} - Prediction for track {k} is {best_lbl}:{best_conf}%")
        # for p in v[0]:
        #     print(np.round(100 * p))
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    print("Saving confusion ", args.confusion)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    np.save(str(args.confusion.with_suffix(".npy")), cm)

    figure = plot_confusion_matrix(cm, class_names=labels)

    plt.savefig(args.confusion.with_suffix(".png"), format="png")


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
