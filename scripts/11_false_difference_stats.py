"""Step 11 - How much false difference each misalignment level creates (numbers for the report dashboard).

For every val/test image: binarise reference and inspected (< 128 = copper), count pixels that differ inside the
valid region [8, 632)^2, split into pixels inside a labelled defect box (real change) and outside (false
difference). Results averaged per shift level (x and y pooled) -> reports/false_difference.json. Needs numpy + Pillow."""
import os, csv, json, collections
import numpy as np
from PIL import Image
OUT = os.environ.get('PCB_OUT', os.path.join(os.path.dirname(__file__), '..'))
S = os.path.join(OUT, 'sets'); M = 8
labs = collections.defaultdict(list)
for l in csv.DictReader(open(os.path.join(OUT, 'index', 'labels.csv'))): labs[l['image_id']].append(l)
level = lambda c: c if c == 'r' else c[1]                       # s0 -> '0', s1x/s1y -> '1', ..., residual -> 'r'
ref_cache, acc = {}, collections.defaultdict(lambda: collections.defaultdict(list))
for i in csv.DictReader(open(os.path.join(OUT, 'index', 'images.csv'))):
    if i['split'] == 'train': continue
    if i['ref_path'] not in ref_cache:
        ref_cache.clear(); ref_cache[i['ref_path']] = np.asarray(Image.open(os.path.join(S, i['ref_path']))) < 128
    ref = ref_cache[i['ref_path']]
    ins = np.asarray(Image.open(os.path.join(S, i['inspected_path']))) < 128
    diff = (ref != ins)[M:-M, M:-M]
    box = np.zeros(ref.shape, bool)
    for l in labs[i['image_id']]:
        x1, y1, x2, y2 = (int(round(float(l[k]))) for k in ('x1', 'y1', 'x2', 'y2'))
        box[max(y1, 0):y2, max(x1, 0):x2] = True
    box = box[M:-M, M:-M]
    a = acc[level(i['condition'])]
    a['diff_pct'].append(100 * diff.mean())
    a['false_pct'].append(100 * (diff & ~box).mean())
    a['inbox_px'].append(int((diff & box).sum())); a['false_px'].append(int((diff & ~box).sum()))
rep = {k: dict(n_images=len(v['diff_pct']), diff_pct=round(float(np.mean(v['diff_pct'])), 3),
               false_pct=round(float(np.mean(v['false_pct'])), 3),
               false_to_defect_ratio=round(float(np.sum(v['false_px']) / max(np.sum(v['inbox_px']), 1)), 1))
       for k, v in sorted(acc.items())}
json.dump(rep, open(os.path.join(OUT, 'reports', 'false_difference.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))
