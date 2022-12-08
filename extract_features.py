# Script to extract track features from a set of CPTV files,
# in preparation for training/testing a classifier
#
# Jamie Heather, Carbon Critical, July 2022
# jamie@carboncritical.org

import argparse
import logging
import json
import os
import numpy as np
import pickle
import sys
import utilsshort
import utils
from pathlib import Path

from trackdatabase import TrackDatabase
from datasetstructures import TrackHeader
from dateutil.parser import parse as parse_date

FILTERED_STATS = {
    "confidence": 0,
    "trap": 0,
    "banned": 0,
    "date": 0,
    "tags": 0,
    "segment_mass": 0,
    "no_data": 0,
    "not-confirmed": 0,
    "tag_names": set(),
    "notags": 0,
    "bad_track_json": 0,
}
EXCLUDED_LABELS = ["poor tracking", "part", "untagged", "unidentified"]
INCLUDED_LABELS = None  # include all
# Point to top-level folder containing subfolders with cptv files
# data_folder = r"D:\Data\cacophony"
#
# X = np.zeros((0, 52))  # Hardcoded to 52 features for now
# Y = np.zeros(0)
# I = np.zeros(0)
# counter = 0
#
# list_subs = os.listdir(data_folder)
# for sub in list_subs:
#
#     sub_folder = os.path.join(data_folder, sub)
#     if os.path.isdir(sub_folder):
#
#         print(sub_folder)
#
#         # Get directory listing
#         list_files = os.listdir(sub_folder)
#
#         for file in list_files:
#
#             # Ignore directories
#             cptvFile = os.path.join(sub_folder, file)
#             if os.path.isfile(cptvFile):
#
#                 # Check for .cptv file extension
#                 fileparts = os.path.splitext(file)
#                 if fileparts[1].lower() == ".cptv":
#
#                     # Check corresponding .txt file is present
#                     txtFile = os.path.join(sub_folder, fileparts[0] + ".txt")
#                     if os.path.isfile(txtFile):
#
#                         # Read metadata
#                         with open(txtFile, "rt") as f:
#                             data = json.load(f)
#
#                         x, y = utils.process_sequence(cptvFile, data)
#                         X = np.concatenate((X, x))
#                         Y = np.concatenate((Y, y))
#                         I = np.concatenate((I, counter * np.ones(len(y))))
#                         counter += 1

# Dump everything out to pickle file
# train = {
#     "X": X,
#     "Y": Y,
#     "I": I,
# }
# with open("train.pickle", "wb") as f:
#     pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
#


def process(db, after_date=None):
    clips = db.get_all_clip_ids(after_date=after_date)
    x_data = []
    y_data = []
    groups = []
    ids = []
    label_counts = {}
    counter = 0
    for clip_id, tracks in clips.items():
        # if clip_id != "4827" and clip_id != "1345135":
        #     continue
        # print("Loading ", clip_id)
        clip_meta = db.get_clip_meta(clip_id)
        background = db.get_clip_background(clip_id)

        tracks = db.get_clip_tracks(clip_id)
        for track_meta in tracks:
            if filter_track(clip_meta, track_meta):
                continue
            track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta)
            # if track_header.label != "false-positive":
            # continue
            if track_header is None:
                print("bad data")
                continue
            print(clip_id, "Track ", track_header.track_id, track_header.label)

            frames = db.get_track(clip_id, track_meta["id"], channels=0)
            X, y = utilsshort.process_track(
                track_header, frames, background, segment_frames=None
            )
            if X is None:
                print("Didn't use", track_header.unique_id)
                continue
            has_nan = np.isnan(X).any()
            if has_nan:
                print("Skipping for nans", track_header.unique_id)
                continue
            x_data.extend(X)
            y = [y] * len(X)
            y_data.extend(y)
            groups.extend([counter] * len(X))
            ids.append(track_header.unique_id)
            if track_header.label in label_counts:
                label_counts[track_header.label] += 1
            else:
                label_counts[track_header.label] = 0

        counter += 1
        # break
    # Dump everything out to pickle file
    train = {
        "X": np.array(x_data),
        "Y": np.array(y_data),
        "I": np.array(groups),
        "ids": np.array(ids),
        "label_counts": label_counts,
    }
    with open("train-new.pickle", "wb") as f:
        pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)


def process_cptv(cptv_file):

    # Check for .cptv file extension

    for folder_path, _, files in os.walk(cptv_file):
        for name in files:
            if os.path.splitext(name)[1] in [".cptv"]:
                cptv_file = Path(os.path.join(folder_path, name))
                meta_file = cptv_file.with_suffix(".txt")
                if meta_file.exists():
                    # Read metadata
                    with open(meta_file, "rt") as f:
                        data = json.load(f)
                    print("loading", data["id"])
                    x, y = utils.process_sequence(cptv_file, data)


def filter_track(clip_meta, track_meta):
    if "tag" not in track_meta:
        FILTERED_STATS["notags"] += 1
        return True
    if INCLUDED_LABELS is not None and track_meta["tag"] not in INCLUDED_LABELS:
        FILTERED_STATS["tags"] += 1
        FILTERED_STATS["tag_names"].add(track_meta["tag"])
        return True
    track_tags = track_meta.get("track_tags")
    if track_tags is not None:
        try:
            track_tags = json.loads(track_tags)
        except:
            logging.error(
                "Error loading track tags json for %s clip %s track %s",
                track_tags,
                clip_meta.get("id"),
                track_meta.get("id"),
            )
            FILTERED_STATS["bad_track_json"] += 1

            return True
        bad_tags = [
            tag["what"]
            for tag in track_tags
            if not tag.get("automatic", False) and tag.get("what") in EXCLUDED_LABELS
        ]
        if len(bad_tags) > 0:
            FILTERED_STATS["tag_names"] |= set(bad_tags)

            FILTERED_STATS["tags"] += 1
            return True
    # always let the false-positives through as we need them even though they would normally
    # be filtered out.
    if "bounds_history" not in track_meta or len(track_meta["bounds_history"]) == 0:
        FILTERED_STATS["no_data"] += 1
        return True

    if track_meta["tag"] == "false-positive":
        return False

    # for some reason we get some records with a None confidence?
    if track_meta.get("confidence", 1.0) <= 0.6:
        FILTERED_STATS["confidence"] += 1
        return True

    # remove tracks of trapped animals
    if (
        "trap" in clip_meta.get("event", "").lower()
        or "trap" in clip_meta.get("trap", "").lower()
    ):
        FILTERED_STATS["trap"] += 1
        return True

    return False


def main():
    init_logging()
    args = parse_args()
    if args.cptv:
        process_cptv(args.cptv)
    else:
        db = TrackDatabase(str(args.db_file), read_only=True)
        process(db, args.date)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "db_file",
        default=None,
        # type=str,
        help="Location of h5py dataset file",
    )
    parser.add_argument(
        "--cptv",
        default=None,
        # type=str,
        help="Location of cptv file",
    )
    parser.add_argument("-d", "--date", help="Use clips after this")

    args = parser.parse_args()
    if args.date:
        args.date = parse_date(args.date)

    args.db_file = Path(args.db_file)
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
