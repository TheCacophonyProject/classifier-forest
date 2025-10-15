import logging
import numpy as np
from pathlib import Path
import joblib

import cv2
from cptv_rs_python_bindings import CptvReader
from datetime import timedelta
from region import Region
from rectangle import Rectangle

crop_rectangle = Rectangle(2, 2, 160 - 2 * 2, 120 - 2 * 2)

FRAME_FEATURES = [
    "sqrt_area",
    "elongation",
    "peak_snr",
    "mean_snr",
    "fill_factor",
    "move_1",
    "rel_move_1",
    "rel_x_move_1",
    "rel_y_move_1",
    "move_3",
    "rel_move_3",
    "rel_x_move_3",
    "rel_y_move_3",
    "move_5",
    "rel_move_5",
    "rel_x_move_5",
    "rel_y_move_5",
    "max_speed",
    "min_speed",
    "avg_speed",
    "max_speed_x",
    "min_speed_x",
    "avg_speed_x",
    "max_speed_y",
    "min_speed_y",
    "avg_speed_y",
    "max_rel_speed",
    "min_rel_speed",
    "avg_rel_speed",
    "max_rel_speed_x",
    "min_rel_speed_x",
    "avg_rel_speed_x",
    "max_rel_speed_y",
    "min_rel_speed_y",
    "avg_rel_speed_y",
    "hist_diff",
]
EXTRA_FEATURES = [
    "speed_distance_ratio",
    "speed_ratio",
    "burst_min",
    "burst_max",
    "birst_mean",
    "burst_chance",
    "burst_per_frame",
    "total frames",
]

EXTRA = ["avg", "std", "max", "min", "diff"]

BURST_FEATURES = []
for extra_lbl in EXTRA:
    for f in FRAME_FEATURES:
        BURST_FEATURES.append(f"{extra_lbl}-{f}")
BURST_FEATURES.extend(EXTRA_FEATURES)


FRAME_FEATURES.append("comparison frames")


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

buff_len = 1


def worker_init(b_len):
    global buff_len
    buff_len = b_len


def extract_features(cptv_file, human_tagged=True):
    cptv_file = Path(cptv_file)
    meta_file = cptv_file.with_suffix(".txt")
    if not meta_file.exists():
        print("No meta for ", cptv_file)
        return None

    frames = None
    background = None
    ffc_frames = None
    all_features = []
    all_tags = []
    all_tracks = []
    frame_features = []
    meta_data = None
    try:
        with meta_file.open("r") as t:
            # add in some metadata stats
            meta_data = json.load(t)
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

                if human_tag not in [
                    "mustelid",
                    "rodent",
                    "rat",
                    "mouse",
                    "ferret",
                    "weasel",
                    "stoat",
                ]:
                    continue
            # print("Got a track")
            # print("Using track with tag", human_tag, track["id"])
            if frames is None:
                frames, background, ffc_frames = load_frames(cptv_file, meta_data)
            burst_features, features = forest_features(
                frames,
                background,
                ffc_frames,
                track,
                buff_len,
                meta_data["id"],
                track["id"],
            )
            if burst_features is None:
                continue
            all_tags.append(human_tag)
            all_tracks.append(track["id"])
            if burst_features is not None:
                all_features.append(burst_features)
            frame_features.append(features)
        assert len(all_tags) == len(all_features)
    except Exception as e:
        logging.error("Exception on %s", cptv_file, exc_info=True)
    return (
        all_tags,
        all_features,
        frame_features,
        all_tracks,
        meta_data["id"] if meta_data else None,
    )


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
    frames, background, ffc_frames, track_meta, buf_len=1, clip_id=0, track_id=0
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

    maximum_features = None
    minimum_features = None
    avg_features = None
    for region in regions:
        # for i, frame in enumerate(track_frames):
        # region = regions[i]

        region.crop(crop_rectangle)
        if region.blank or region.area <= 1:
            prev_count = 0

            continue
        if region.frame_number in ffc_frames:
            continue
        if len(frames) <= region.frame_number:
            continue
        frame = frames[region.frame_number]
        feature = FrameFeatures(region, buf_len)
        sub_back = region.subimage(background)
        cropped_frame = region.subimage(frame)
        thermal = cropped_frame
        feature.calc_histogram(sub_back, thermal, normalize=True)
        t_median = np.median(frame)

        thermal = thermal + back_med - t_median
        feature.calculate(thermal, sub_back)
        if buf_len > 1:
            count_back = min(buf_len, prev_count)

            for i in range(count_back):
                prev = frame_features[-i - 1]
                vel = feature.cent - prev.cent
                feature.speed[i] = np.sqrt(np.sum(vel * vel))
                feature.rel_speed[i] = feature.speed[i] / feature.sqrt_area
                feature.rel_speed_x[i] = np.abs(vel[0]) / feature.sqrt_area
                feature.rel_speed_y[i] = np.abs(vel[1]) / feature.sqrt_area
                feature.speed_x[i] = np.abs(vel[0])
                feature.speed_y[i] = np.abs(vel[1])
                feature.comparison_frames += 1
            frame_features.append(feature)
        frame_features.append(feature)
        features = feature.features()
        all_features.append(features)
        features = features[:-1]
        prev_count += 1
        f_count += 1
        if buf_len > 1:
            if maximum_features is None:
                maximum_features = features.copy()
                minimum_features = features.copy()

                avg_features = features.copy()
            else:
                maximum_features = np.maximum(features, maximum_features)
                non_zero = features != 0
                current_zero = minimum_features == 0
                minimum_features[current_zero] = features[current_zero]
                minimum_features[non_zero] = np.minimum(
                    minimum_features[non_zero], features[non_zero]
                )
                # Aggregate
                avg_features += features
    # Compute statistics for all tracks that have the min required duration
    if buf_len == 1:
        return None, np.array(all_features)
    N = f_count - np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            3,
            3,
            3,
            3,
            5,
            5,
            5,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )  # Normalise each measure by however many samples went into it
    if f_count == 0:
        logging.error("No frames for %s -%s", clip_id, track_id)
        return None, None
    frame_features_2 = np.array(all_features).copy()
    avg_features /= N
    all_features = np.array(all_features)
    std_features = np.sqrt(
        np.sum((all_features[:, :-1] - avg_features) ** 2, axis=0) / N
    )
    diff_features = maximum_features - minimum_features
    burst_features = calculate_burst_features(frame_features, avg_features[5])

    if np.any(np.isinf(avg_features)) or np.any(np.isnan(avg_features)):
        logging.error("Nan or inf detected for %s - %s ", clip_id, track_id)
        avg_features = np.nan_to_num(avg_features, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.hstack(
        (
            avg_features,
            std_features,
            maximum_features,
            minimum_features,
            diff_features,
            burst_features,
            f_count,
        )
    )
    # frame_f = frame_features_2[0]
    # assert len(frame_f)== len(FRAME_FEATURES), f"{len(frame_f)} does not match features {len(FRAME_FEATURES)}"
    # for v, name in zip(frame_f, FRAME_FEATURES):
    #     print("Have ", v, " for ", name)
    # 1/0
    return X, frame_features_2


def calculate_burst_features(frames, mean_speed):
    #

    cut_off = max(2, (1 + mean_speed))
    speed_above = len([f for f in frames if f.speed[0] > cut_off])
    speed_below = len([f for f in frames if f.speed[0] <= cut_off])

    burst_frames = 0
    burst_ratio = []
    burst_history = []
    total_birst_frames = 0
    low_speed_distance = 0
    high_speed_distance = 0
    for i, frame in enumerate(frames):
        if frame.speed[0] < cut_off:
            low_speed_distance += frame.speed[0]
        else:
            high_speed_distance += frame.speed[0]
        if i > 0:
            prev = frames[i - 1]
            if prev.speed[0] > cut_off and frame.speed[0] > cut_off:
                burst_frames += 1
            else:
                if burst_frames > 0:
                    burst_start = i - burst_frames - 1
                    if len(burst_history) > 0:
                        # length of non burst frames is from previous burst end
                        prev = burst_history[-1]
                        burst_start -= prev[0] + prev[1]
                    burst_history.append((i - burst_frames - 1, burst_frames + 1))
                    burst_ratio.append(burst_start / (burst_frames + 1))
                    total_birst_frames += burst_frames + 1
                    burst_frames = 0
    burst_ratio = np.array(burst_ratio)
    if speed_above == 0:
        speed_ratio = 0
        speed_distance_ratio = 0
    else:
        speed_distance_ratio = low_speed_distance / high_speed_distance
        speed_ratio = speed_below / speed_above

    if len(burst_ratio) == 0:
        burst_min = 0
        burst_max = 0
        burst_mean = 0
    else:
        burst_min = np.amin(burst_ratio)
        burst_max = np.amax(burst_ratio)
        burst_mean = np.mean(burst_ratio)
    burst_chance = len(burst_ratio) / len(frames)
    burst_per_frame = total_birst_frames / len(frames)
    return np.array(
        [
            speed_distance_ratio,
            speed_ratio,
            burst_min,
            burst_max,
            burst_mean,
            burst_chance,
            burst_per_frame,
        ]
    )


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
        self.buff_len = buff_len
        self.comparison_frames = 0
        if self.buff_len > 1:
            self.rel_speed = np.zeros(buff_len)
            self.rel_speed_x = np.zeros(buff_len)
            self.rel_speed_y = np.zeros(buff_len)
            self.speed_x = np.zeros(buff_len)
            self.speed_y = np.zeros(buff_len)
            self.speed = np.zeros(buff_len)

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
        if self.buff_len == 1:
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

        non_zero = np.array([s for s in self.speed if s > 0])
        max_speed = 0
        min_speed = 0
        avg_speed = 0
        if len(non_zero) > 0:
            max_speed = np.amax(non_zero)
            min_speed = np.amin(non_zero)
            avg_speed = np.mean(non_zero)

        non_zero = np.array([s for s in self.speed_x if s > 0])
        max_speed_x = 0
        min_speed_x = 0
        avg_speed_x = 0
        if len(non_zero) > 0:
            max_speed_x = np.amax(non_zero)
            min_speed_x = np.amin(non_zero)
            avg_speed_x = np.mean(non_zero)

        non_zero = np.array([s for s in self.speed_y if s > 0])
        max_speed_y = 0
        min_speed_y = 0
        avg_speed_y = 0
        if len(non_zero) > 0:
            max_speed_y = np.amax(non_zero)
            min_speed_y = np.amin(non_zero)
            avg_speed_y = np.mean(non_zero)

        non_zero = np.array([s for s in self.rel_speed if s > 0])
        max_rel_speed = 0
        min_rel_speed = 0
        avg_rel_speed = 0
        if len(non_zero) > 0:
            max_rel_speed = np.amax(non_zero)
            min_rel_speed = np.amin(non_zero)
            avg_rel_speed = np.mean(non_zero)

        non_zero = np.array([s for s in self.rel_speed_x if s > 0])
        max_rel_speed_x = 0
        min_rel_speed_x = 0
        avg_rel_speed_x = 0
        if len(non_zero) > 0:
            max_rel_speed_x = np.amax(non_zero)
            min_rel_speed_x = np.amin(non_zero)
            avg_rel_speed_x = np.mean(non_zero)

        non_zero = np.array([s for s in self.rel_speed_y if s > 0])
        max_rel_speed_y = 0
        min_rel_speed_y = 0
        avg_rel_speed_y = 0
        if len(non_zero) > 0:
            max_rel_speed_y = np.amax(non_zero)
            min_rel_speed_y = np.amin(non_zero)
            avg_rel_speed_y = np.mean(non_zero)

        return np.array(
            [
                self.sqrt_area,
                self.elongation,
                self.peak_snr,
                self.mean_snr,
                self.fill_factor,
                self.speed[0],
                self.rel_speed[0],
                self.rel_speed_x[0],
                self.rel_speed_y[0],
                self.speed[2],
                self.rel_speed[2],
                self.rel_speed_x[2],
                self.rel_speed_y[2],
                self.speed[4],
                self.rel_speed[4],
                self.rel_speed_x[4],
                self.rel_speed_y[4],
                max_speed,
                min_speed,
                avg_speed,
                max_speed_x,
                min_speed_x,
                avg_speed_x,
                max_speed_y,
                min_speed_y,
                avg_speed_y,
                max_rel_speed,
                min_rel_speed,
                avg_rel_speed,
                max_rel_speed_x,
                min_rel_speed_x,
                avg_rel_speed_x,
                max_rel_speed_y,
                min_rel_speed_y,
                avg_rel_speed_y,
                self.histogram_diff,
                self.comparison_frames,
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
            sub_back = np.uint8(sub_back)
            crop_t = np.uint8(crop_t)

        assert sub_back.shape == crop_t.shape
        # sub_back = np.uint8(sub_back)
        # crop_t = np.uint8(crop_t)
        sub_back = sub_back[..., np.newaxis]
        crop_t = crop_t[..., np.newaxis]
        h_bins = 60
        histSize = [h_bins]
        channels = [0]
        hist_base = cv2.calcHist(
            [np.uint8(sub_back)],
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
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cptv_dir", help="Dir to load")
    parser.add_argument(
        "--save-file", help="Model to load and do preds", default="features.npy"
    )
    parser.add_argument(
        "--buff-len",
        help="Buf len should be 5 for burst features 1 for frame by frame",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    args.save_file = Path(args.save_file)
    args.cptv_dir = Path(args.cptv_dir)

    return args


def main():
    init_logging()
    args = parse_args()
    load_dir = args.cptv_dir
    print("Loading", load_dir)
    files = list(load_dir.glob(f"**/*.cptv"))
    files.sort()
    all_tags = []
    all_features = []
    all_ids = []
    all_track_ids = []

    burst_tags = []
    burst_features = []
    burst_ids = []
    burst_track_ids = []
    total_files = len(files)
    done = 0
    # probably should not bother repeat track ids etc and just handle this on load
    with Pool(processes=4, initializer=worker_init, initargs=(args.buff_len,)) as pool:
        for result in pool.imap_unordered(extract_features, files):
            if done % 100 == 0:
                print(f"{done} / {total_files}")
            done += 1
            if result is None:
                continue
            tags, features, frame_features, track_ids, clip_id = result
            for track_features, tag, track_id in zip(features, tags, track_ids):
                burst_tags.append(tag)
                burst_features.append(track_features)
                burst_ids.append(clip_id)
                burst_track_ids.append(track_id)
                # ([track_id] * len(track_features))
            assert len(set(burst_track_ids)) == len(burst_track_ids)

            for track_features, tag, track_id in zip(frame_features, tags, track_ids):
                all_tags.extend([tag] * len(track_features))
                all_features.extend(track_features)
                all_ids.extend([clip_id] * len(track_features))
                all_track_ids.extend([track_id] * len(track_features))
        assert len(set(burst_track_ids)) == len(burst_track_ids)

        assert len(all_tags) == len(all_features)
        assert len(all_ids) == len(all_tags)
        assert len(all_track_ids) == len(all_tags)
        assert len(burst_tags) == len(burst_features)
        assert len(burst_ids) == len(burst_features)
        assert len(burst_track_ids) == len(burst_features)
    print("Got tags and features", np.array(all_features).shape)
    print("Saving to ", args.save_file)
    with args.save_file.open("wb") as f:
        np.save(f, np.array([1]))
        np.save(f, np.array(all_tags))
        np.save(f, np.array(all_features))
        np.save(f, np.array(all_ids))
        np.save(f, np.array(all_track_ids))

    print("Got burst features", np.array(burst_features).shape)
    burst_features_f = args.save_file.parent / f"{args.save_file.stem}-burst.npy"
    print("Saving burst to ", burst_features_f)
    with burst_features_f.open("wb") as f:
        np.save(f, np.array([5]))

        np.save(f, np.array(burst_tags))
        np.save(f, np.array(burst_features))
        np.save(f, np.array(burst_ids))
        np.save(f, np.array(burst_track_ids))


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
