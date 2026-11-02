"""Step 7 - Export for Ultralytics YOLO (YOLOv10-N).

Each pair becomes ONE 3-channel PNG so a standard RGB-pretrained detector sees both images:
    channel 0 = reference, channel 1 = inspected, channel 2 = |reference - inspected|
Labels: one .txt per image, YOLO format `class xc yc w h` (class 0-5 = open..pin-hole, coords / 640).
ALL boxes are written (incl. those flagged ignore): YOLO has no 'ignore' concept, and dropping a real,
fully visible defect would teach the model that it is background. The ignore rule is applied later by
our own evaluation script (reads index/labels.csv), not by Ultralytics.

Subsets written to OUT/yolo/images/<subset>/ and OUT/yolo/labels/<subset>/:
  train_noaug   800  train pairs, s0 only
  train_aug    2400  each train pair 3x: s0 + 1 integer shift (1-5 px, random axis/sign) + 1 residual
                     (sub-pixel + rotation). Seed stream [6314, pair_id, 1] - independent of the eval sets.
  val_<cond>, test_<cond>   one subset per evaluation condition (s0, s1x ... s5y, r)
Train for E epochs on train_noaug and E/3 epochs on train_aug so both runs take the same number of
gradient steps - otherwise 'augmentation helps' could just mean 'trained 3x longer'."""
import os, sys, csv, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); import pcb_shift as T
OUT = os.environ.get('PCB_OUT'); S = os.path.join(OUT, 'sets'); Y = os.path.join(OUT, 'yolo')
NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
imgs = list(csv.DictReader(open(os.path.join(OUT, 'index', 'images.csv'))))
labs = collections.defaultdict(list)
for l in csv.DictReader(open(os.path.join(OUT, 'index', 'labels.csv'))): labs[l['image_id']].append(l)
rd = lambda p: cv2.imread(os.path.join(S, p), cv2.IMREAD_GRAYSCALE)

def composite(ref, ins):
    return np.dstack([ref, ins, cv2.absdiff(ref, ins)])

SHARD = int(os.environ.get('SHARD', 0)); NSHARD = int(os.environ.get('NSHARD', 1))   # run shards in parallel

def write(subset, name, img3, boxes, classes):
    for d in ('images', 'labels'): os.makedirs(os.path.join(Y, d, subset), exist_ok=True)
    cv2.imwrite(os.path.join(Y, 'images', subset, name + '.png'), img3)
    H, W = img3.shape[:2]; lines = []
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
        lines.append(f"{c - 1} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    open(os.path.join(Y, 'labels', subset, name + '.txt'), 'w').write('\n'.join(lines) + ('\n' if lines else ''))

counts = collections.Counter(); manifest = []
for i, x in enumerate(imgs):
    if i % NSHARD != SHARD: continue
    ref = rd(x['ref_path']); ins = rd(x['inspected_path'])
    L = labs[x['image_id']]
    B = [[float(l[k]) for k in ('x1', 'y1', 'x2', 'y2')] for l in L]; C = [int(l['cls']) for l in L]
    if x['split'] == 'train':
        write('train_noaug', x['image_id'], composite(ref, ins), B, C); counts['train_noaug'] += 1
        write('train_aug', x['image_id'], composite(ref, ins), B, C); counts['train_aug'] += 1
        rng = np.random.default_rng([6314, int(x['pair_id']), 1])
        s = int(rng.integers(1, 6)); ax = int(rng.integers(2)); sg = int(rng.choice([-1, 1]))
        dxr, dyr, rot, _ = T.sample_residual(rng)
        for tag, (dx, dy, r, integer) in [('ai', (sg * s * (ax == 0), sg * s * (ax == 1), 0.0, True)),
                                          ('ar', (dxr, dyr, rot, False))]:
            M = T.affine(dx, dy, r, ins.shape)
            write('train_aug', f"{x['pair_id']}_{tag}", composite(ref, T.warp_image(ins, M, integer)),
                  T.warp_boxes(np.array(B), M), C)
            counts['train_aug'] += 1
            manifest.append(dict(image=f"{x['pair_id']}_{tag}", pair_id=x['pair_id'], dx=round(float(dx), 4),
                                 dy=round(float(dy), 4), rot_deg=round(float(r), 4)))
    else:
        sub = f"{x['split']}_{x['condition']}"
        write(sub, x['image_id'], composite(ref, ins), B, C); counts[sub] += 1
with open(os.path.join(Y, f'train_aug_transforms_part{SHARD}.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)

names = '\n'.join(f"  {i}: {n}" for i, n in enumerate(NAMES))
for run, tr in [('noaug', 'train_noaug'), ('aug', 'train_aug')]:
    open(os.path.join(Y, f'data_{run}.yaml'), 'w').write(
        f"# Ultralytics dataset file. Edit `path` if you unzip somewhere other than /content/yolo\n"
        f"path: /content/yolo\ntrain: images/{tr}\nval: images/val_s0\nnames:\n{names}\n")
print(SHARD, dict(counts))
