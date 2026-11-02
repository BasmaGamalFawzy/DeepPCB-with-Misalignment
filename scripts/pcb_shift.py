"""Shared transform code - used both to build the fixed evaluation sets and (optionally) for on-the-fly
shift augmentation during detector training, so both methods and all runs see the same geometry.

Convention: the REFERENCE (template) image is never touched. The INSPECTED image and its boxes are
moved together. dx>0 moves content right, dy>0 moves content down (image y axis points down).
Uncovered border pixels are filled by edge replication; they lie inside the excluded border MARGIN."""
import numpy as np, cv2

MARGIN = 8          # px excluded on every side for evaluation; > max |shift| (5) and > residual worst case
                    # (1.5 px + 0.5 deg rotation about the centre of a 640 px image ~ 3.95 px at the corners)
MAX_DISP = 6        # px, upper bound on how far any transform here moves a pixel (5 int; 1.5+3.95 residual)
IGNORE_MARGIN = MARGIN + MAX_DISP   # original boxes closer than this to an edge are ignored under ALL conditions,
                                    # which guarantees every non-ignored box stays inside the valid region
INT_SHIFTS = [0, 1, 2, 3, 5]
RESIDUAL_SHIFT = (0.25, 1.5)   # px, magnitude of the sub-pixel translation
RESIDUAL_ROT = 0.5             # deg, max |rotation|

def affine(dx=0.0, dy=0.0, rot_deg=0.0, size=(640, 640)):
    """2x3 matrix: rotate about the image centre by rot_deg (counter-clockwise, OpenCV convention), then translate."""
    h, w = size
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), rot_deg, 1.0)
    M[0, 2] += dx; M[1, 2] += dy
    return M

def warp_image(img, M, integer=False):
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if integer else cv2.INTER_LINEAR   # integer shifts stay pixel-exact
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)

def warp_boxes(boxes, M):
    """boxes: (N,4) x1,y1,x2,y2 -> axis-aligned envelope of the 4 transformed corners."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    if len(b) == 0: return b
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], 1)        # N,4,2
    pts = pts @ M[:, :2].T + M[:, 2]
    return np.concatenate([pts.min(1), pts.max(1)], 1)

def ignore_flags(orig_boxes, size=(640, 640), margin=IGNORE_MARGIN):
    """A defect is 'boundary-truncated' (ignore=1) if its ORIGINAL box comes within `margin` px of the image
    edge (MARGIN + MAX_DISP). Decided once on the original geometry so the same boxes are ignored under every shift."""
    h, w = size; b = np.asarray(orig_boxes, float).reshape(-1, 4)
    return ((b[:, 0] < margin) | (b[:, 1] < margin) | (b[:, 2] > w - margin) | (b[:, 3] > h - margin)).astype(int)

def sample_int_conditions(rng):
    """For one pair: list of (condition, dx, dy). Magnitude s in INT_SHIFTS, axis x and y each, sign drawn by rng."""
    out = [('s0', 0, 0)]
    for s in INT_SHIFTS[1:]:
        sx, sy = rng.choice([-1, 1], 2)
        out += [(f's{s}x', int(sx * s), 0), (f's{s}y', 0, int(sy * s))]
    return out

def sample_residual(rng):
    mag = rng.uniform(*RESIDUAL_SHIFT); ang = rng.uniform(0, 2 * np.pi)
    return float(mag * np.cos(ang)), float(mag * np.sin(ang)), float(rng.uniform(-RESIDUAL_ROT, RESIDUAL_ROT)), float(mag)

def random_train_shift(img, boxes, rng, p=0.5, max_int=5, residual_p=0.5):
    """On-the-fly augmentation helper for the 'with shift augmentation' detector run."""
    if rng.random() > p: return img, np.asarray(boxes, float)
    if rng.random() < residual_p:
        dx, dy, rot, _ = sample_residual(rng); M = affine(dx, dy, rot, img.shape[:2]); integer = False
    else:
        s = int(rng.integers(1, max_int + 1)); ax = rng.integers(2); sg = rng.choice([-1, 1])
        M = affine(sg * s * (ax == 0), sg * s * (ax == 1), 0, img.shape[:2]); integer = True
    return warp_image(img, M, integer), warp_boxes(boxes, M)
