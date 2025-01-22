import logging
import numpy as np
from pathlib import Path
import joblib

import cv2
from cptv_rs_python_bindings import CptvReader
from datetime import timedelta
from region import Region
from rectangle import Rectangle

crop_rectangle = Rectangle(2, 2, 160 - 2 * 2, 120-2 * 2)

FEAT_LABELS = [
    "sqrt_area",
    "elongation",
    "peak_snr",
    "mean_snr",
    "fill_factor",
    # "move_1",
    # "rel_move_1",
    # "rel_x_move_1",
    # "rel_y_move_1",
    # "move_3",
    # "rel_move_3",
    # "rel_x_move_3",
    # "rel_y_move_3",
    # "move_5",
    # "rel_move_5",
    # "rel_x_move_5",
    # "rel_y_move_5",
    # "max_speed",
    # "min_speed",
    # "avg_speed",
    # "max_speed_x",
    # "min_speed_x",
    # "avg_speed_x",
    # "max_speed_y",
    # "min_speed_y",
    # "avg_speed_y",
    # "max_rel_speed",
    # "min_rel_speed",
    # "avg_rel_speed",
    # "max_rel_speed_x",
    # "min_rel_speed_x",
    # "avg_rel_speed_x",
    # "max_rel_speed_y",
    # "min_rel_speed_y",
    # "avg_rel_speed_y",
    # "hist_diff",
]
# EXTRA_FEATURES = [
#     "speed_distance_ratio",
#     "speed_ratio",
#     "burst_min",
#     "burst_max",
#     "birst_mean",
#     "burst_chance",
#     "burst_per_frame",
#     "total frames",
# ]

EXTRA = ["avg", "std", "max", "min", "diff"]

ALL_FEATURES = []
for extra_lbl in EXTRA:
    for f in FEAT_LABELS:
        ALL_FEATURES.append(f"{extra_lbl}-{f}")
# ALL_FEATURES.extend(EXTRA_FEATURES)

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

EXCLUDED_TAGS = ["poor tracking", "part", "untagged", "unidentified"]
import json


def extract_features(cptv_file, human_tagged=True):
    cptv_file = Path(cptv_file)
    meta_file = cptv_file.with_suffix(".txt")
    if not meta_file.exists():
        print("No meta for ", cptv_file)
        return None
    with meta_file.open("r") as t:
        # add in some metadata stats
        meta_data = json.load(t)
    frames = None
    background = None
    ffc_frames = None
    all_features = []
    all_tags = []
    all_tracks = []
    try:
        if "Tracks" not in meta_data:
            return None
        for track in meta_data["Tracks"]:
            human_tags = []
            human_tag = "untagged"
            if human_tagged:
                tags = [
                    tag["what"] for tag in track["tags"] if tag["automatic"] == False
                ]
                tags = set(tags)
                if len(tags) > 2:
                    continue
                if len(tags) == 0:
                    continue
                human_tag = list(tags)[0]
                if human_tag in EXCLUDED_TAGS:
                    continue
            # print("Using track with tag", human_tag, track["id"])
            if frames is None:
                frames, background, ffc_frames = load_frames(cptv_file, meta_data)

            track_features = forest_features(frames, background, ffc_frames, track)
            true_tags = [human_tag] * len(track_features)
            all_tags.extend(true_tags)
            all_features.extend(track_features)
            all_tracks.extend([track["id"]] * len(track_features))
        assert len(all_tags) == len(all_features)
    except:
        pass
    return all_tags, all_features, [meta_data["id"]] * len(all_tags), all_tracks


FFC_PERIOD = timedelta(seconds=9.9)


def is_affected_by_ffc(cptv_frame):
    if hasattr(cptv_frame, "ffc_status") and cptv_frame.ffc_status in [1, 2]:
        return True

    if cptv_frame.time_on is None or cptv_frame.last_ffc_time is None:
        return False
    if isinstance(cptv_frame.time_on, int):
        return (cptv_frame.time_on - cptv_frame.last_ffc_time) < FFC_PERIOD.seconds
    return (cptv_frame.time_on - cptv_frame.last_ffc_time) < FFC_PERIOD


def load_frames(cptv_file, meta_data):
    ffc_frames = []
    cptv_frames = []
    tracker_version = meta_data.get("tracker_version")

    background = None
    frame_i = 0
    reader = CptvReader(str(cptv_file))
    header = reader.get_header()
    while True:
        frame = reader.next_frame()
        if frame is None:
            break
        if frame.background_frame:
            background = frame.pix
            # bug in previous tracker version where background was first frame
            if tracker_version >= 10:
                continue
        ffc = is_affected_by_ffc(frame)
        if ffc:
            ffc_frames.append(frame_i)
        cptv_frames.append(frame.pix)
        frame_i += 1
    frames = np.uint16(cptv_frames)
    if background is None:
        background = np.mean(frames, axis=0)

    return frames, background, ffc_frames

FPS = 9

def forest_features(
    frames,
    background,
    ffc_frames,
    track_meta,
):
    frame_features = []
    all_features = []
    f_count = 0
    prev_count = 0
    back_med = np.median(background)
    regions = []
    start = None
    end = None
    for i, r in enumerate(track_meta.get("positions")):
        if isinstance(r, list):
            region = Region.region_from_array(r[1])
            if region.frame_number is None:
                if i == 0:
                    frame_number = round(r[0] * FPS)
                    region.frame_number = frame_number
                else:
                    region.frame_number = prev_frame + 1
        else:
            region = Region.region_from_json(r)
        if region.frame_number is None:
            if "frameTime" in r:
                if i == 0:
                    region.frame_number = round(r["frameTime"] * 9)
                else:
                    region.frame_number = prev_frame + 1
        prev_frame = region.frame_number
        region.frame_number = region.frame_number
        assert region.frame_number >= 0
        regions.append(region)
        if start is None:
            start = region.frame_number
        end = region.frame_number
    for region in regions:
        # for i, frame in enumerate(track_frames):
        # region = regions[i]
       
        region.crop(crop_rectangle)
        if region.blank or region.area<= 1:
            prev_count = 0

            continue
        if region.frame_number in ffc_frames:
            continue
        frame = frames[region.frame_number]
        feature = FrameFeatures(region)
        sub_back = region.subimage(background)
        feature.calc_histogram(sub_back, frame, normalize=True)
        t_median = np.median(frame)
        cropped_frame = region.subimage(frame)
        thermal = cropped_frame

        thermal = thermal + back_med - t_median

        feature.calculate(thermal, sub_back)

        frame_features.append(feature)
        features = feature.features()
        all_features.append(features)
        prev_count += 1
    # Compute statistics for all tracks that have the min required duration
    return np.array(all_features)


class FrameFeatures:
    def __init__(self, region, buff_len=5):
        # self.thermal = thermal
        self.region = region
        self.cent = None
        self.extent = None
        self.theta = None
        self.sqrt_area = None
        self.std_back = None
        self.peak_snr = None
        self.mean_snr = None
        self.fill_factor = None
        self.histogram_diff = 0
        self.thermal_min = None
        self.thermal_max = None
        self.thermal_std = None
        self.filtered_max = None
        self.filtered_std = None
        self.filtered_min = None

    def calculate(self, thermal, sub_back):
        self.thermal_min = np.amin(thermal)
        self.thermal_max = np.amax(thermal)
        self.thermal_std = np.std(thermal)
        filtered = thermal - sub_back
        filtered = np.abs(filtered)

        self.filtered_max = np.amax(filtered)
        self.filtered_min = np.amin(filtered)

        self.filtered_std = np.std(filtered)

        # Calculate weighted centroid and second moments etc
        cent, extent, theta = intensity_weighted_moments(filtered, self.region)

        self.cent = cent
        self.extent = extent
        self.theta = theta
        # Instantaneous shape features
        area = np.pi * extent[0] * extent[1]
        self.sqrt_area = np.sqrt(area)
        self.elongation = extent[0] / extent[1]
        self.std_back = np.std(sub_back) + 1.0e-9

        # Instantaneous intensity features
        self.peak_snr = (self.thermal_max - np.mean(sub_back)) / self.std_back
        self.mean_snr = self.thermal_std / self.std_back
        self.fill_factor = np.sum(filtered) / area

    def features(self):
        return np.array(
            [
                self.sqrt_area,
                self.elongation,
                self.peak_snr,
                self.mean_snr,
                self.fill_factor,
                self.histogram_diff,
                self.thermal_max,
                self.thermal_min,
                self.thermal_std,
                self.filtered_max,
                self.filtered_min,
                self.filtered_std,
            ]
        )

    def calc_histogram(self, sub_back, crop_t, normalize=False):
        if normalize:
            max_v = np.amax(sub_back)
            min_v = np.amin(sub_back)
            sub_back = (np.float32(sub_back) - min_v) / (max_v - min_v)
            max_v = np.amax(crop_t)
            min_v = np.amin(crop_t)
            crop_t = (np.float32(crop_t) - min_v) / (max_v - min_v)

            sub_back *= 255
            crop_t *= 255

        # sub_back = np.uint8(sub_back)
        # crop_t = np.uint8(crop_t)
        sub_back = sub_back[..., np.newaxis]
        crop_t = crop_t[..., np.newaxis]
        h_bins = 60
        histSize = [h_bins]
        channels = [0]
        hist_base = cv2.calcHist(
            [sub_back],
            channels,
            None,
            histSize,
            [0, 255],
            accumulate=False,
        )
        cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_track = cv2.calcHist(
            [crop_t],
            channels,
            None,
            histSize,
            [0, 255],
            accumulate=False,
        )
        # print(hist_track)
        cv2.normalize(
            hist_track,
            hist_track,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
        )
        self.histogram_diff = cv2.compareHist(hist_base, hist_track, 0)


# Find centre of mass and size/orientation of the hot spot
def intensity_weighted_moments(sub, region=None):
    tot = np.sum(sub)
    # print(tot, "using", region)
    if tot <= 0.0:
        # Zero image - replace with ones so calculations can continue
        sub = np.ones(sub.shape)
        tot = sub.size

    # Calculate weighted centroid
    Y, X = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]
    cx = np.sum(sub * X) / tot
    cy = np.sum(sub * Y) / tot
    X = X - cx
    Y = Y - cy
    cent = np.array([region.x + cx, region.y + cy])

    # Second moments matrix
    mxx = np.sum(X * X * sub) / tot
    mxy = np.sum(X * Y * sub) / tot
    myy = np.sum(Y * Y * sub) / tot
    M = np.array([[mxx, mxy], [mxy, myy]])

    # Extent and angle
    w, v = np.linalg.eigh(M)
    w = np.abs(w)
    if w[0] < w[1]:
        w = w[::-1]
        v = v[:, ::-1]
    extent = (
        np.sqrt(w) + 0.5
    )  # Add half a pixel so that a single bright pixel has non-zero extent
    theta = np.arctan2(v[1, 0], v[0, 0])

    return cent, extent, theta


import sys
from multiprocessing import Pool


def main():
    init_logging()
    load_dir = Path(sys.argv[1])
    files = list(load_dir.glob(f"**/*.cptv"))
    all_tags = []
    all_features = []
    all_ids = []
    with Pool(processes=8) as pool:
        for result in pool.imap_unordered(extract_features, files):
            if result is None:
                continue
            tags, features, ids, track_ids = result
            all_tags.extend(tags)
            all_features.extend(features)
            all_ids.extend(ids)
    print("Got tags and features", len(all_tags), len(all_features))
    with open("features.npy", "wb") as f:
        np.save(f, np.array(all_tags))
        np.save(f, np.array(all_features))
        np.save(f, np.array(all_ids))

def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
            root_logger.removeHandler(handler)
    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )
if __name__ == "__main__":
    main()
