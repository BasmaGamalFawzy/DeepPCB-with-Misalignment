"""Step 2a - Near-duplicate scan (perceptual hash of the reference/template images) and a
sanity check that the original pairs are already aligned (phase correlation)."""
import os, csv, json, itertools, collections
import numpy as np, cv2
ROOT = os.environ.get('PCB_ROOT'); OUT = os.environ.get('PCB_OUT')
pairs = list(csv.DictReader(open(os.path.join(OUT, 'index', 'pairs.csv'))))

def gray(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_GRAYSCALE); return im

def dhash(im, n=16):
    s = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).flatten()

H = {}; shifts = []
for i, p in enumerate(pairs):
    t = gray(p['temp']); s = gray(p['test'])
    H[p['pair_id']] = dhash(t)
    if i % 5 == 0:
        (dx, dy), resp = cv2.phaseCorrelate(t.astype(np.float32), s.astype(np.float32))
        shifts.append((p['pair_id'], round(dx, 3), round(dy, 3), round(resp, 3)))
ids = [p['pair_id'] for p in pairs]; split = {p['pair_id']: p['official_split'] for p in pairs}
grp = {p['pair_id']: p['group'] for p in pairs}
M = np.array([H[i] for i in ids])
# pairwise hamming distances (bits out of 256)
D = (M[:, None, :] != M[None, :, :]).sum(-1)
close = []
for a, b in zip(*np.where(np.triu(D <= 20, 1))):
    close.append(dict(a=ids[a], b=ids[b], hamming=int(D[a, b]), split_a=split[ids[a]], split_b=split[ids[b]],
                      group_a=grp[ids[a]], group_b=grp[ids[b]]))
with open(os.path.join(OUT, 'index', 'near_duplicates_template_dhash.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['a', 'b', 'hamming', 'split_a', 'split_b', 'group_a', 'group_b'])
    w.writeheader(); w.writerows(close)
c = collections.Counter((min(x['split_a'], x['split_b']), max(x['split_a'], x['split_b']), x['hamming'] <= 5) for x in close)
sh = np.array([[s[1], s[2]] for s in shifts])
rep = dict(n_close_pairs_le20=len(close), by_split_and_le5={str(k): v for k, v in c.items()},
           cross_group_close=sum(x['group_a'] != x['group_b'] for x in close),
           hamming_hist_min_per_image=np.bincount(np.minimum(np.sort(D + np.eye(len(D), dtype=int) * 999, 1)[:, 0], 40)).tolist(),
           align_check_n=len(shifts), align_abs_dx_max=float(np.abs(sh[:, 0]).max()), align_abs_dy_max=float(np.abs(sh[:, 1]).max()),
           align_abs_median=[float(np.median(np.abs(sh[:, 0]))), float(np.median(np.abs(sh[:, 1])))],
           align_n_over_0p5px=int((np.abs(sh) > 0.5).any(1).sum()))
json.dump(rep, open(os.path.join(OUT, 'reports', 'duplicates_alignment.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))
