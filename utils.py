# Function for extracting features for all tracks in a CPTV file
#
# Jamie Heather, Carbon Critical, July 2022
# jamie@carboncritical.org


from cptv import CPTVReader

import cv2
import numpy as np

BUFF_LEN = 5
FEAT_LABELS = [
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


class Frame:
    def __init__(self, region):
        # self.thermal = thermal
        self.region = region
        self.cent = None
        self.extent = None
        self.thera = None
        self.rel_speed = np.zeros(BUFF_LEN)
        self.rel_speed_x = np.zeros(BUFF_LEN)
        self.rel_speed_y = np.zeros(BUFF_LEN)
        self.speed_x = np.zeros(BUFF_LEN)
        self.speed_y = np.zeros(BUFF_LEN)
        self.speed = np.zeros(BUFF_LEN)
        self.histogram_diff = 0

    def calculate(self, thermal, sub_back):
        self.thermal_max = np.amax(thermal)
        self.thermal_std = np.std(thermal)
        filtered = thermal - sub_back
        filtered = np.abs(filtered)
        f_max = filtered.max()

        if f_max > 0.0:
            filtered /= f_max

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

    def histogram(self, sub_back, crop_t):
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
        h_bins = 50
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

    def features(self):
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
            ]
        )


def process_track(
    track,
    frame_data,
    background,
    FRAMES_PER_SEC=9.0,
    ENLARGE_FACTOR=4,
    PLAYBACK_DELAY=1,
):
    frames = []
    minimum_features = None
    avg_features = None
    std_features = None
    maximum_features = None
    f_count = 0
    prev_count = 0
    low_speed_distance = 0
    high_speed_distance = 0
    burst_frames = 0
    burst_history = []
    last_burst = 0
    if track.num_frames <= BUFF_LEN:
        return None, None
    for f, region in frame_data:
        if region.blank or region.width == 0 or region.height == 0:
            prev_count = 0
            continue

        sub_back = region.subimage(background).copy()
        max_v = np.amax(background)
        min_v = np.amin(background)
        norm_back = (np.float32(background) - min_v) / (max_v - min_v)
        norm_back *= 255

        frame = Frame(region)
        # print("max b", np.amax(background))
        # cv2.imshow("background is", np.uint8(norm_back))
        # cv2.waitKey(10000)
        # break
        frame.histogram(sub_back, f)
        # median has been rounded from db so slight difference compared to doing from cptv
        median = np.float64(track.frame_temp_median[f_count])
        f_count += 1
        f = np.float64(f)
        f = f + np.median(background) - median

        frame.calculate(f, sub_back)
        count_back = min(BUFF_LEN, prev_count)
        for i in range(count_back):
            prev = frames[-i - 1]
            vel = frame.cent - prev.cent
            frame.speed[i] = np.sqrt(np.sum(vel * vel))
            frame.speed_x[i] = np.abs(vel[0])
            frame.speed_y[i] = np.abs(vel[1])

            frame.rel_speed[i] = frame.speed[i] / frame.sqrt_area
            frame.rel_speed_x[i] = np.abs(vel[0]) / frame.sqrt_area
            frame.rel_speed_y[i] = np.abs(vel[1]) / frame.sqrt_area
        # cv2.imshow("F", np.uint8(f))

        # cv2.waitKey(100)
        # if count_back >= 5:
        # 1 / 0
        frames.append(frame)
        features = frame.features()
        prev_count += 1
        if maximum_features is None:
            minimum_features = features
            maximum_features = features
            avg_features = features
            std_features = features * features
        else:
            # let min be any non zero
            for i, (new, min_f) in enumerate(zip(features, minimum_features)):
                if min_f == 0:
                    minimum_features[i] = new
                elif new != 0 and new < min_f:
                    minimum_features[i] = new
            # minimum_features = np.minimum(features, minimum_features)
            maximum_features = np.maximum(features, maximum_features)
            # Aggregate
            avg_features += features
            std_features += features * features

    # Compute statistics for all tracks that have the min required duration
    valid_counter = 0
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
    avg_features /= N
    std_features = np.sqrt(std_features / N - avg_features**2)
    diff_features = maximum_features - minimum_features
    # gp better check this

    birst_features = calculate_burst_features(frames)
    X = np.hstack(
        (
            avg_features,
            std_features,
            maximum_features,
            minimum_features,
            diff_features,
            birst_features,
            np.array([track.num_frames]),
        )
    )
    return X, track.label


def debug_features(X):
    # for frame in frames:
    EXTRA = ["avg", "std", "max", "min", "diff"]
    other_l = EXTRA[0]
    other_i = 0
    for i, a in enumerate(X):
        index = i % len(FEAT_LABELS)
        if index == 0:
            if other_i >= len(EXTRA):
                break
            other_l = EXTRA[other_i]
            other_i += 1
            print("SPLIT\n\n", other_l)
        print(other_l, FEAT_LABELS[index], " - ", a)

    for i, a in enumerate(X[-len(EXTRA_FEATURES) :]):
        print(EXTRA_FEATURES[i], a)


def calculate_burst_features(frames):
    #
    avg_speeds = np.array([f.speed[0] for f in frames])
    mean_speed = np.mean(avg_speeds)
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


def process_sequence(
    cptvFile, data, FRAMES_PER_SEC=9.0, ENLARGE_FACTOR=4, PLAYBACK_DELAY=1
):

    NUM_FEATS = 17
    BUFF_LEN = 5

    num_tracks = len(data["Tracks"])
    X = np.zeros((num_tracks, 3 * NUM_FEATS + 1))
    Y = ["unknown" for i in range(num_tracks)]

    if num_tracks == 0:
        return X, Y

    # Dictionary to keep hold of everything for every track
    hist = [
        {
            "label": "unknown",
            "len": 0,
            "stats": {
                "avg": np.zeros(NUM_FEATS),
                "max": np.zeros(NUM_FEATS),
                "std": np.zeros(NUM_FEATS),
            },
            "last": [
                {
                    "xy": np.zeros(2),
                    "area": 0,
                }
                for j in range(BUFF_LEN)
            ],
        }
        for i in range(num_tracks)
    ]

    # Find start and end frames for each track
    track_start = [round(track["start"] * FRAMES_PER_SEC) for track in data["Tracks"]]
    track_end = [round(track["end"] * FRAMES_PER_SEC) for track in data["Tracks"]]

    # Check each track for a manual classification
    for track_counter in range(num_tracks):
        for tag in data["Tracks"][track_counter]["tags"]:
            if tag["automatic"] == False:
                hist[track_counter]["label"] = tag["what"]
                break

    # Start reading image frames + tracker data
    with open(cptvFile, "rb") as f:

        try:
            reader = CPTVReader(f)

        except:

            # Unable to read file - output empty arrays
            X = np.zeros((0, 3 * NUM_FEATS + 1))
            Y = ["unknown" for i in range(0)]
            return X, Y

        frame_counter = -1
        back_img = None
        got_background = False

        # Loop over all image frames
        for frame in reader:

            # Skip over background frames
            if frame.background_frame:
                back_img = frame.pix.astype(float)
                got_background = True
                continue

            # Can't proceed without the background frame
            if got_background == False:
                X = np.zeros((0, 3 * NUM_FEATS + 1))
                Y = ["unknown" for i in range(0)]
                return X, Y

            frame_counter += 1

            # Check if any tracks overlap this frame (note there can be more than one!)
            overlapping_tracks = [
                i
                for i in range(num_tracks)
                if frame_counter >= track_start[i] and frame_counter < track_end[i]
            ]

            # Skip frame if nothing is going on
            if len(overlapping_tracks) == 0:
                continue

            img = frame.pix.astype(float)
            found_valid_tracks = False
            # Try to fix brightness fluctations to match with background frame
            img += np.median(back_img) - np.median(img)

            # img += np.median(back_img) - np.median(img)
            for track_counter in overlapping_tracks:
                if hist[track_counter]["label"] == "unknown":
                    continue
                # Get tracked position
                positions = data["Tracks"][track_counter]["positions"]
                pos = positions[frame_counter - track_start[track_counter]]
                assert pos["order"] == frame_counter

                min_size = 2
                if pos["width"] < min_size or pos["height"] < min_size:
                    continue
                x1 = pos["x"]
                y1 = pos["y"]
                x2 = pos["x"] + pos["width"]
                y2 = pos["y"] + pos["height"]

                sub = img[y1:y2, x1:x2] - back_img[y1:y2, x1:x2]
                sub = np.abs(sub)
                sub_max = sub.max()

                if sub_max > 0.0:
                    sub /= sub_max

                # Calculate weighted centroid and second moments etc
                cent, extent, theta = intensity_weighted_moments_old(sub, [x1, y1])

                # Instantaneous shape features
                area = np.pi * extent[0] * extent[1]
                sqrt_area = np.sqrt(area)
                elongation = extent[0] / extent[1]

                # Instantaneous intensity features
                std_back = np.std(back_img[y1:y2, x1:x2]) + 1.0e-9
                peak_snr = (
                    np.max(img[y1:y2, x1:x2]) - np.mean(back_img[y1:y2, x1:x2])
                ) / std_back

                mean_snr = np.std(img[y1:y2, x1:x2]) / std_back
                fill_factor = np.sum(sub) / area

                # Time-based features
                speed = np.zeros(BUFF_LEN)
                rel_speed = np.zeros(BUFF_LEN)
                rel_speed_x = np.zeros(BUFF_LEN)
                rel_speed_y = np.zeros(BUFF_LEN)
                for k in range(BUFF_LEN):
                    # track is 10
                    # at k = 1
                    # take this and
                    if hist[track_counter]["len"] > k:
                        # hist k will be  10 - 0 - 1 = 8 % 5 = 3
                        # comparing hist[4] which will be last frame

                        j = (hist[track_counter]["len"] - k - 1) % BUFF_LEN
                        vel = cent - hist[track_counter]["last"][j]["xy"]
                        speed[k] = np.sqrt(np.sum(vel * vel))
                        rel_speed[k] = speed[k] / sqrt_area
                        rel_speed_x[k] = np.abs(vel[0]) / sqrt_area
                        rel_speed_y[k] = np.abs(vel[1]) / sqrt_area

                        # if k >= 4:
                        # 1 / 0
                # Bundle all features into vector
                feats = np.array(
                    [
                        sqrt_area,
                        elongation,
                        peak_snr,
                        mean_snr,
                        fill_factor,
                        speed[0],
                        rel_speed[0],
                        rel_speed_x[0],
                        rel_speed_y[0],
                        speed[2],
                        rel_speed[2],
                        rel_speed_x[2],
                        rel_speed_y[2],
                        speed[4],
                        rel_speed[4],
                        rel_speed_x[4],
                        rel_speed_y[4],
                    ]
                )
                # 0, 1, 2, 3, 4, 5, 0, ,1 , 2, 3, 4, 5
                # Remember some stuff for next time
                j = hist[track_counter]["len"] % BUFF_LEN
                hist[track_counter]["last"][j]["xy"] = cent
                hist[track_counter]["last"][j]["area"] = area
                hist[track_counter]["len"] += 1
                hist[track_counter]["id"] = data["Tracks"][track_counter]["id"]
                # Aggregate
                hist[track_counter]["stats"]["avg"] += feats
                hist[track_counter]["stats"]["std"] += feats * feats
                hist[track_counter]["stats"]["max"] = np.maximum(
                    feats, hist[track_counter]["stats"]["max"]
                )

            #     # Prepare image for display (first time around)
            #     if not found_valid_tracks:
            #         found_valid_tracks = True
            #         imgMin = img.min()
            #         imgMax = img.max() + 1
            #         rgb = (255.0 * (img - imgMin) / (imgMax - imgMin)).astype(np.uint8)
            #         width = ENLARGE_FACTOR * rgb.shape[1]
            #         height = ENLARGE_FACTOR * rgb.shape[0]
            #         rgb = cv2.resize(
            #             rgb, (width, height), interpolation=cv2.INTER_NEAREST
            #         )
            #         rgb = cv2.merge([rgb, rgb, rgb])
            #
            #     # Overlay tracking details
            #     x1 = ENLARGE_FACTOR * pos["x"] - 1
            #     y1 = ENLARGE_FACTOR * pos["y"] - 1
            #     x2 = ENLARGE_FACTOR * (pos["x"] + pos["width"])
            #     y2 = ENLARGE_FACTOR * (pos["y"] + pos["height"])
            #     # cv2.rectangle(rgb, (x1,y1), (x2,y2), (255,0,0))
            #     cv2.putText(
            #         rgb,
            #         hist[track_counter]["label"],
            #         (x1, y1 - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 255, 0),
            #     )
            #     cent = (
            #         round(ENLARGE_FACTOR * cent[0]),
            #         round(ENLARGE_FACTOR * cent[1]),
            #     )
            #     axes = (
            #         round(2 * ENLARGE_FACTOR * extent[0]),
            #         round(2 * ENLARGE_FACTOR * extent[1]),
            #     )
            #     cv2.ellipse(
            #         rgb, cent, axes, round(np.degrees(theta)), 0, 360, (0, 255, 0)
            #     )
            #
            # # Update display
            # if found_valid_tracks:
            #     cv2.imshow("image", rgb)
            #     cv2.waitKey(PLAYBACK_DELAY)

    # Compute statistics for all tracks that have the min required duration
    valid_counter = 0
    for track_counter in range(num_tracks):
        num_frames = hist[track_counter]["len"]
        if num_frames > BUFF_LEN:
            N = num_frames - np.array(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5]
            )  # Normalise each measure by however many samples went into it
            hist[track_counter]["stats"]["avg"] /= N
            hist[track_counter]["stats"]["std"] = np.sqrt(
                hist[track_counter]["stats"]["std"] / N
                - hist[track_counter]["stats"]["avg"] ** 2
            )

            X[valid_counter, 0:NUM_FEATS] = hist[track_counter]["stats"]["avg"]
            X[valid_counter, NUM_FEATS : 2 * NUM_FEATS] = hist[track_counter]["stats"][
                "std"
            ]
            X[valid_counter, 2 * NUM_FEATS : 3 * NUM_FEATS] = hist[track_counter][
                "stats"
            ]["max"]
            X[valid_counter, 3 * NUM_FEATS] = num_frames
            Y[valid_counter] = hist[track_counter]["label"]

            # print("Track", hist[track_counter]["id"], hist[track_counter]["label"])
            # for a in X[valid_counter]:
            #     print(a)
            valid_counter += 1
    X = X[0:valid_counter, :]
    Y = Y[0:valid_counter]

    return X, Y


# Find centre of mass and size/orientation of the hot spot
def intensity_weighted_moments_old(sub, region=None):

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
    cent = np.array([region[0] + cx, region[1] + cy])

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
