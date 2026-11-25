"""Head pose from 68 landmarks via solvePnP + same/different direction grouping."""
import cv2, numpy as np

IDX = [30, 8, 36, 45, 48, 54]  # nose tip, chin, eye outer corners, mouth corners
# generic 3D face, camera convention (x right, y down, z away from camera), mm-ish units
MODEL_3D = np.array([(0, 0, 0), (0, 330, 65), (-225, -170, 135), (225, -170, 135),
                     (-150, 150, 125), (150, 150, 125)], np.float64)


def head_pose(pts68, frame_w, frame_h, focal=None):
    """pts68: (68,2) in frame pixels. Returns (yaw, pitch, roll) in degrees."""
    f = focal or frame_w
    K = np.array([[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, np.asarray(pts68, np.float64)[IDX], K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    yaw = -np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def group_directions(yaws, thr=15.0):
    """1-D clustering on yaw. Returns (groups as index lists, n_groups, size of largest group)."""
    order = np.argsort(yaws); groups = [[int(order[0])]]
    for a, b in zip(order[:-1], order[1:]):
        if yaws[b] - yaws[a] > thr: groups.append([])
        groups[-1].append(int(b))
    return groups, len(groups), max(len(g) for g in groups)