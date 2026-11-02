"""Step 1 - Inspect DeepPCB: index every pair, read labels, basic stats.
Output: index/pairs.csv (one row per pair), index/boxes.csv (one row per box), reports/inspect.json"""
import os, json, csv, collections
import numpy as np, cv2

ROOT = os.environ.get('PCB_ROOT', os.path.join(os.path.dirname(__file__), '..', '..', 'DeepPCB-master', 'DeepPCB-master', 'PCBData'))
OUT = os.environ.get('PCB_OUT', os.path.join(os.path.dirname(__file__), '..'))
CLASSES = {1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'copper', 6: 'pin-hole'}

def read_list(name):
    rows = []
    with open(os.path.join(ROOT, name)) as f:
        for line in f.read().split('\n'):
            line = line.strip()
            if not line: continue
            img, lab = line.split()
            stem = img[:-4]                      # group20085/20085/20085000
            rows.append(dict(pair_id=os.path.basename(stem), group=img.split('/')[0],
                             temp=stem + '_temp.jpg', test=stem + '_test.jpg', label=lab))
    return rows

os.makedirs(os.path.join(OUT, 'index'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'reports'), exist_ok=True)
pairs, boxes = [], []
stats = collections.Counter(); shapes = collections.Counter(); problems = []
for split, fname in [('trainval', 'trainval.txt'), ('test', 'test.txt')]:
    for r in read_list(fname):
        r['official_split'] = split
        for k in ('temp', 'test', 'label'):
            if not os.path.exists(os.path.join(ROOT, r[k])): problems.append(f"missing {r[k]}")
        t = cv2.imread(os.path.join(ROOT, r['temp']), cv2.IMREAD_UNCHANGED)
        s = cv2.imread(os.path.join(ROOT, r['test']), cv2.IMREAD_UNCHANGED)
        shapes[(str(t.shape), str(s.shape))] += 1
        r['temp_channels'] = 1 if t.ndim == 2 else t.shape[2]
        r['test_channels'] = 1 if s.ndim == 2 else s.shape[2]
        H, W = s.shape[:2]
        n = 0
        for line in open(os.path.join(ROOT, r['label'])):
            v = line.replace(',', ' ').split()
            if len(v) != 5: continue
            x1, y1, x2, y2, c = map(int, v)
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H): problems.append(f"bad box {r['label']}: {v}")
            boxes.append(dict(pair_id=r['pair_id'], group=r['group'], official_split=split,
                              x1=x1, y1=y1, x2=x2, y2=y2, cls=c, cls_name=CLASSES.get(c, '?'),
                              w=x2 - x1, h=y2 - y1))
            stats[(split, CLASSES.get(c, '?'))] += 1; n += 1
        r['n_boxes'] = n
        pairs.append(r)

def dump(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
dump(os.path.join(OUT, 'index', 'pairs.csv'), pairs)
dump(os.path.join(OUT, 'index', 'boxes.csv'), boxes)

ws = np.array([b['w'] for b in boxes]); hs = np.array([b['h'] for b in boxes])
rep = dict(n_pairs={s: sum(p['official_split'] == s for p in pairs) for s in ('trainval', 'test')},
           n_boxes=len(boxes), shapes={f"{k[0]} | {k[1]}": v for k, v in shapes.items()},
           class_counts={f"{k[0]}/{k[1]}": v for k, v in sorted(stats.items())},
           box_w_pct=[int(np.percentile(ws, q)) for q in (0, 5, 50, 95, 100)],
           box_h_pct=[int(np.percentile(hs, q)) for q in (0, 5, 50, 95, 100)],
           boxes_per_pair=[int(np.percentile([p['n_boxes'] for p in pairs], q)) for q in (0, 50, 100)],
           groups={s: dict(collections.Counter(p['group'] for p in pairs if p['official_split'] == s)) for s in ('trainval', 'test')},
           problems=problems[:50], n_problems=len(problems))
json.dump(rep, open(os.path.join(OUT, 'reports', 'inspect.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))
