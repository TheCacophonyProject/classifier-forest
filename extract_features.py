# Script to extract track features from a set of CPTV files,
# in preparation for training/testing a classifier
#
# Jamie Heather, Carbon Critical, July 2022
# jamie@carboncritical.org


import json
import os
import numpy as np
import pickle

import utils

# Point to top-level folder containing subfolders with cptv files
data_folder = r"D:\Data\cacophony"

X = np.zeros((0, 52))  # Hardcoded to 52 features for now
Y = np.zeros(0)
I = np.zeros(0)
counter = 0

list_subs = os.listdir(data_folder)
for sub in list_subs:

    sub_folder = os.path.join(data_folder, sub)
    if os.path.isdir(sub_folder):

        print(sub_folder)

        # Get directory listing
        list_files = os.listdir(sub_folder)

        for file in list_files:

            # Ignore directories
            cptvFile = os.path.join(sub_folder, file)
            if os.path.isfile(cptvFile):

                # Check for .cptv file extension
                fileparts = os.path.splitext(file)
                if fileparts[1].lower() == ".cptv":

                    # Check corresponding .txt file is present
                    txtFile = os.path.join(sub_folder, fileparts[0] + ".txt")
                    if os.path.isfile(txtFile):

                        # Read metadata
                        with open(txtFile, "rt") as f:
                            data = json.load(f)

                        x, y = utils.process_sequence(cptvFile, data)
                        X = np.concatenate((X, x))
                        Y = np.concatenate((Y, y))
                        I = np.concatenate((I, counter * np.ones(len(y))))
                        counter += 1

# Dump everything out to pickle file
train = {
    "X": X,
    "Y": Y,
    "I": I,
}
with open("train.pickle", "wb") as f:
    pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
