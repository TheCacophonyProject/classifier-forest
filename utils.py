# Function for extracting features for all tracks in a CPTV file
#
# Jamie Heather, Carbon Critical, July 2022
# jamie@carboncritical.org


from cptv import CPTVReader

import cv2
import numpy as np


def process_sequence(cptvFile, data, FRAMES_PER_SEC=9.0, ENLARGE_FACTOR=4, PLAYBACK_DELAY=1):

    NUM_FEATS = 17
    BUFF_LEN = 5

    num_tracks = len(data["Tracks"])
    X = np.zeros((num_tracks, 3*NUM_FEATS+1))
    Y = ['unknown' for i in range(num_tracks)]
    
    if num_tracks == 0:
        return X, Y

    # Dictionary to keep hold of everything for every track
    hist = [{
            'label': "unknown",
            'len': 0,
            'stats': {
                'avg': np.zeros(NUM_FEATS),
                'max': np.zeros(NUM_FEATS),
                'std': np.zeros(NUM_FEATS),
            },
            'last': [{
                'xy': np.zeros(2),
                'area': 0,
            } for j in range(BUFF_LEN)],
        } for i in range(num_tracks)]

    # Find start and end frames for each track
    track_start = [round(track["start"] * FRAMES_PER_SEC) for track in data["Tracks"]]
    track_end = [round(track["end"] * FRAMES_PER_SEC) for track in data["Tracks"]]

    # Check each track for a manual classification
    for track_counter in range(num_tracks):
        for tag in data["Tracks"][track_counter]["tags"]:
            if tag["automatic"]==False:
                hist[track_counter]["label"] = tag["what"]
                break

    # Start reading image frames + tracker data
    with open(cptvFile, "rb") as f:

        try:
            reader = CPTVReader(f)

        except:

            # Unable to read file - output empty arrays
            X = np.zeros((0, 3*NUM_FEATS+1))
            Y = ['unknown' for i in range(0)]
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
                X = np.zeros((0, 3*NUM_FEATS+1))
                Y = ['unknown' for i in range(0)]
                return X, Y

            frame_counter += 1

            # Check if any tracks overlap this frame (note there can be more than one!)
            overlapping_tracks = [i for i in range(num_tracks) if frame_counter >= track_start[i] and frame_counter < track_end[i]]

            # Skip frame if nothing is going on
            if len(overlapping_tracks) == 0:
                continue

            img = frame.pix.astype(float)
            found_valid_tracks = False

            # Try to fix brightness fluctations to match with background frame
            img += np.median(back_img) - np.median(img)

            for track_counter in overlapping_tracks:
                
                if hist[track_counter]["label"] == "unknown":
                    continue

                # Get tracked position
                positions = data["Tracks"][track_counter]["positions"]
                pos = positions[frame_counter-track_start[track_counter]]
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
                cent, extent, theta = intensity_weighted_moments(sub, [x1,y1])

                # Instantaneous shape features
                area = np.pi * extent[0] * extent[1]
                sqrt_area = np.sqrt(area)
                elongation = extent[0] / extent[1]
                
                # Instantaneous intensity features
                std_back = np.std(back_img[y1:y2, x1:x2]) + 1.0e-9
                peak_snr = (np.max(img[y1:y2, x1:x2]) - np.mean(back_img[y1:y2, x1:x2])) / std_back
                mean_snr = np.std(img[y1:y2, x1:x2]) / std_back
                fill_factor = np.sum(sub) / area

                # Time-based features
                speed = np.zeros(BUFF_LEN)
                rel_speed = np.zeros(BUFF_LEN)
                rel_speed_x = np.zeros(BUFF_LEN)
                rel_speed_y = np.zeros(BUFF_LEN)
                for k in range(BUFF_LEN):
                    if hist[track_counter]["len"] > k:
                        j = (hist[track_counter]["len"] - k - 1) % BUFF_LEN
                        vel = cent - hist[track_counter]["last"][j]["xy"]
                        speed[k] = np.sqrt(np.sum(vel*vel))
                        rel_speed[k] = speed[k] / sqrt_area
                        rel_speed_x[k] = np.abs(vel[0]) / sqrt_area
                        rel_speed_y[k] = np.abs(vel[1]) / sqrt_area

                # Bundle all features into vector
                feats = np.array([
                    sqrt_area,
                    elongation,
                    peak_snr,
                    mean_snr,
                    fill_factor,
                    speed[0], rel_speed[0], rel_speed_x[0], rel_speed_y[0],
                    speed[2], rel_speed[2], rel_speed_x[2], rel_speed_y[2],
                    speed[4], rel_speed[4], rel_speed_x[4], rel_speed_y[4],
                    ])

                # Remember some stuff for next time
                j = hist[track_counter]["len"] % BUFF_LEN
                hist[track_counter]["last"][j]["xy"] = cent
                hist[track_counter]["last"][j]["area"] = area
                hist[track_counter]["len"] += 1

                # Aggregate
                hist[track_counter]["stats"]["avg"] += feats
                hist[track_counter]["stats"]["std"] += feats*feats
                hist[track_counter]["stats"]["max"] = np.maximum(feats, hist[track_counter]["stats"]["max"])

                # Prepare image for display (first time around)
                if not found_valid_tracks:
                    found_valid_tracks = True
                    imgMin = img.min()
                    imgMax = img.max() + 1
                    rgb = (255.0*(img - imgMin) / (imgMax - imgMin)).astype(np.uint8)
                    width = ENLARGE_FACTOR * rgb.shape[1]
                    height = ENLARGE_FACTOR * rgb.shape[0]
                    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)
                    rgb = cv2.merge([rgb,rgb,rgb])

                # Overlay tracking details
                x1 = ENLARGE_FACTOR * pos["x"] - 1
                y1 = ENLARGE_FACTOR * pos["y"] - 1
                x2 = ENLARGE_FACTOR * (pos["x"] + pos["width"])
                y2 = ENLARGE_FACTOR * (pos["y"] + pos["height"])
                #cv2.rectangle(rgb, (x1,y1), (x2,y2), (255,0,0))
                cv2.putText(rgb, hist[track_counter]["label"], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                cent = (round(ENLARGE_FACTOR*cent[0]), round(ENLARGE_FACTOR*cent[1]))
                axes = (round(2*ENLARGE_FACTOR*extent[0]), round(2*ENLARGE_FACTOR*extent[1]))
                cv2.ellipse(rgb, cent, axes, round(np.degrees(theta)), 0, 360, (0,255,0))

            # Update display
            if found_valid_tracks:
                cv2.imshow("image", rgb)
                cv2.waitKey(PLAYBACK_DELAY)

    # Compute statistics for all tracks that have the min required duration
    valid_counter = 0
    for track_counter in range(num_tracks):
        num_frames = hist[track_counter]["len"]
        if num_frames > BUFF_LEN:
            N = num_frames - np.array([0,0,0,0,0,1,1,1,1,3,3,3,3,5,5,5,5])  # Normalise each measure by however many samples went into it
            hist[track_counter]["stats"]["avg"] /= N
            hist[track_counter]["stats"]["std"] = np.sqrt(hist[track_counter]["stats"]["std"] / N - hist[track_counter]["stats"]["avg"]**2)
            X[valid_counter, 0:NUM_FEATS] = hist[track_counter]["stats"]["avg"]
            X[valid_counter, NUM_FEATS:2*NUM_FEATS] = hist[track_counter]["stats"]["std"]
            X[valid_counter, 2*NUM_FEATS:3*NUM_FEATS] = hist[track_counter]["stats"]["max"]
            X[valid_counter, 3*NUM_FEATS] = num_frames
            Y[valid_counter] = hist[track_counter]["label"]
            valid_counter += 1

    X = X[0:valid_counter,:]
    Y = Y[0:valid_counter]

    return X, Y


# Find centre of mass and size/orientation of the hot spot
def intensity_weighted_moments(sub, xy_corner=[0,0]):    

    tot = np.sum(sub)
    if tot <= 0.0:
        # Zero image - replace with ones so calculations can continue
        sub = np.ones(sub.shape)
        tot = sub.size

    # Calculate weighted centroid
    Y,X = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]    
    cx = np.sum(sub * X) / tot
    cy = np.sum(sub * Y) / tot
    X = X - cx
    Y = Y - cy
    cent = np.array([
        xy_corner[0] + cx, 
        xy_corner[1] + cy])

    # Second moments matrix
    mxx = np.sum(X*X*sub) / tot
    mxy = np.sum(X*Y*sub) / tot
    myy = np.sum(Y*Y*sub) / tot
    M = np.array([
        [mxx, mxy], 
        [mxy, myy]])

    # Extent and angle
    w,v = np.linalg.eigh(M)
    w = np.abs(w)
    if w[0] < w[1]:
        w = w[::-1]
        v = v[:, ::-1]
    extent = np.sqrt(w) + 0.5               # Add half a pixel so that a single bright pixel has non-zero extent
    theta = np.arctan2(v[1,0], v[0,0])

    return cent, extent, theta