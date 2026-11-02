"""Step 4 - Build the fixed, seeded evaluation sets for val and test (identical for baseline and detector),
plus clean (s0) copies of train. All images saved as lossless 8-bit grayscale PNG.

Layout (under OUT/sets):
  ref/<pair_id>.png                      reference/template, written once, never transformed
  <split>/<condition>/<pair_id>.png      inspected image after transform
Conditions: s0, s{1,2,3,5}{x,y} (integer, pixel-exact) and r (residual: sub-pixel shift + small rotation).
Manifests: index/images.csv (one row per inspected image with its transform) and index/labels.csv
(one row per box per image, with ignore flag)."""
import os, sys, csv, json, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import pcb_shift as T
ROOT = os.environ.get('PCB_ROOT'); OUT = os.environ.get('PCB_OUT'); SEED = 6314
rows = list(csv.DictReader(open(os.path.join(OUT, 'index', 'splits.csv'))))
boxes = collections.defaultdict(list)
for b in csv.DictReader(open(os.path.join(OUT, 'index', 'boxes.csv'))):
    boxes[b['pair_id']].append(([int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])], int(b['cls'])))
S = os.path.join(OUT, 'sets'); os.makedirs(os.path.join(S, 'ref'), exist_ok=True)
img_rows, lab_rows = [], []
for r in sorted(rows, key=lambda r: r['pair_id']):
    pid = r['pair_id']
    rng = np.random.default_rng([SEED, int(pid)])           # per-pair seed: order-independent, reproducible
    ref = cv2.imread(os.path.join(ROOT, r['temp']), cv2.IMREAD_GRAYSCALE)
    ins = cv2.imread(os.path.join(ROOT, r['test']), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(S, 'ref', pid + '.png'), ref)
    B = np.array([b for b, _ in boxes[pid]], float); C = [c for _, c in boxes[pid]]
    ign = T.ignore_flags(B, ins.shape)
    conds = T.sample_int_conditions(rng); dxr, dyr, rot, mag = T.sample_residual(rng)
    if r['split'] == 'train': conds = conds[:1]              # train: clean only (augment on the fly)
    else: conds = conds + [('r', dxr, dyr, rot)]
    for c in conds:
        name, dx, dy = c[0], c[1], c[2]; rd = c[3] if len(c) > 3 else 0.0
        integer = name != 'r'
        M = T.affine(dx, dy, rd, ins.shape)
        out = T.warp_image(ins, M, integer) if name != 's0' else ins
        nb = T.warp_boxes(B, M)
        rel = os.path.join(r['split'], name, pid + '.png')
        os.makedirs(os.path.join(S, r['split'], name), exist_ok=True)
        cv2.imwrite(os.path.join(S, rel), out)
        img_rows.append(dict(image_id=f"{pid}_{name}", pair_id=pid, group=r['group'], split=r['split'], condition=name,
                             ref_path='ref/' + pid + '.png', inspected_path=rel.replace(os.sep, '/'),
                             dx=round(dx, 4), dy=round(dy, 4), rot_deg=round(rd, 4),
                             shift_mag=round(float(np.hypot(dx, dy)), 4),
                             native_dx=r['native_dx'], native_dy=r['native_dy'], native_misaligned=r['native_misaligned'],
                             test_has_trainval_duplicate=r['test_has_trainval_duplicate'], n_boxes=len(B), n_ignored=int(ign.sum())))
        for (x1, y1, x2, y2), cl, ig in zip(nb, C, ign):
            lab_rows.append(dict(image_id=f"{pid}_{name}", pair_id=pid, split=r['split'], condition=name,
                                 x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2), cls=cl, ignore=int(ig)))
for name, data in [('images.csv', img_rows), ('labels.csv', lab_rows)]:
    with open(os.path.join(OUT, 'index', name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
cnt = collections.Counter((x['split'], x['condition']) for x in img_rows)
rep = dict(seed=SEED, margin_px=T.MARGIN, ignore_margin_px=T.IGNORE_MARGIN, n_images=len(img_rows), n_labels=len(lab_rows),
           images_per_split_condition={f"{k[0]}/{k[1]}": v for k, v in sorted(cnt.items())},
           ignored_boxes_per_pair_total=int(sum(int(x['n_ignored']) for x in img_rows if x['condition'] == 's0')),
           residual_stats={k: [round(float(np.min(v)), 3), round(float(np.max(v)), 3)] for k, v in
                           dict(shift_mag=[x['shift_mag'] for x in img_rows if x['condition'] == 'r'],
                                rot_deg=[x['rot_deg'] for x in img_rows if x['condition'] == 'r']).items()})
json.dump(rep, open(os.path.join(OUT, 'reports', 'shift_sets.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))
