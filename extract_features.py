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
import utils
from pathlib import Path

from trackdatabase import TrackDatabase
from datasetstructures import TrackHeader

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


def process(db):
    clips = db.get_all_clip_ids()
    x_data = []
    y_data = []
    groups = []
    ids = []
    counter = 0
    for clip_id, tracks in clips.items():
        print("Loading ", clip_id)
        clip_meta = db.get_clip_meta(clip_id)
        background = db.get_clip_background(clip_id)
        tracks = db.get_clip_tracks(clip_id)
        for track_meta in tracks:
            if filter_track(clip_meta, track_meta):
                filtered += 1
                continue
            track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta)
            if track_header is None:
                print("bad data")
                continue
            print("Track ", track_header.track_id, track_header.label)
            frames = db.get_track(clip_id, track_meta["id"], channels=0)
            X, y = utils.process_track(track_header, frames, background)
            x_data.append(X)
            y_data.append(y)
            groups.append(counter)

            ids.append(track_header.unique_id)
        counter += 1
        # break
    # Dump everything out to pickle file
    train = {
        "X": np.array(x_data),
        "Y": np.array(y_data),
        "I": np.array(groups),
        "ids": np.array(ids),
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


def filter_track(clip, track):
    return False


def main():
    init_logging()
    args = parse_args()
    if args.cptv:
        process_cptv(args.cptv)
    else:
        db = TrackDatabase(str(args.db_file), read_only=True)
        process(db)


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
    args = parser.parse_args()
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
