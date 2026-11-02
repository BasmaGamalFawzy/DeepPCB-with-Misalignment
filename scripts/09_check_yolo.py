"""Step 8 - Verify the YOLO export.
(a) every image has a label file and the right number of lines
(b) YOLO labels converted back to pixels match index/labels.csv (max abs error, px)
(c) composite channels: ch0 == reference, ch1 == inspected, ch2 == |ch0 - ch1|
(d) integer-shift augmented train images: pixels inside each moved box == pixels inside the original box
(e) visual sheet qa/yolo_check.png drawn from the YOLO .txt files themselves"""
import os, csv, glob, collections, json
import numpy as np, cv2
OUT = os.environ.get('PCB_OUT'); Y = os.path.join(OUT, 'yolo'); S = os.path.join(OUT, 'sets')
labs = collections.defaultdict(list)
for l in csv.DictReader(open(os.path.join(OUT, 'index', 'labels.csv'))): labs[l['image_id']].append(l)
imgs = {x['image_id']: x for x in csv.DictReader(open(os.path.join(OUT, 'index', 'images.csv')))}
def read_yolo(p):
    out = []
    for line in open(p).read().split('\n'):
        if not line.strip(): continue
        c, xc, yc, w, h = line.split(); xc, yc, w, h = (float(v) * 640 for v in (xc, yc, w, h))
        out.append((int(c) + 1, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return out
res = collections.Counter(); maxerr = 0.0
for p in glob.glob(os.path.join(Y, 'images', '*', '*.png')):
    sub = p.split(os.sep)[-2]; name = os.path.basename(p)[:-4]
    lp = os.path.join(Y, 'labels', sub, name + '.txt')
    res['images'] += 1; res['missing_label'] += int(not os.path.exists(lp))
    if name in imgs:                                   # every non-augmented image
        yl = read_yolo(lp); ref = labs[name]
        res['count_mismatch'] += int(len(yl) != len(ref))
        for a, b in zip(yl, ref):
            res['class_mismatch'] += int(a[0] != int(b['cls']))
            maxerr = max(maxerr, max(abs(a[i + 1] - float(b[k])) for i, k in enumerate(('x1', 'y1', 'x2', 'y2'))))
        if res['images'] % 25 == 0:                    # channel check on a sample
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED); x = imgs[name]
            r = cv2.imread(os.path.join(S, x['ref_path']), 0); s = cv2.imread(os.path.join(S, x['inspected_path']), 0)
            res['channel_checked'] += 1
            res['channel_fail'] += int(not (np.array_equal(im[..., 0], r) and np.array_equal(im[..., 1], s)
                                            and np.array_equal(im[..., 2], cv2.absdiff(r, s))))
# (d) integer-shift augmented images
tf = {r['image']: r for r in csv.DictReader(open(os.path.join(Y, 'train_aug_transforms.csv')))}
for name, t in tf.items():
    if not name.endswith('_ai'): continue
    a = cv2.imread(os.path.join(Y, 'images', 'train_aug', t['pair_id'] + '_s0.png'), cv2.IMREAD_UNCHANGED)[..., 1]
    b = cv2.imread(os.path.join(Y, 'images', 'train_aug', name + '.png'), cv2.IMREAD_UNCHANGED)[..., 1]
    for o, n in zip(read_yolo(os.path.join(Y, 'labels', 'train_aug', t['pair_id'] + '_s0.txt')),
                    read_yolo(os.path.join(Y, 'labels', 'train_aug', name + '.txt'))):
        o = [int(round(v)) for v in o[1:]]; n = [int(round(v)) for v in n[1:]]
        res['aug_box_checked'] += 1
        res['aug_box_fail'] += int(not np.array_equal(a[o[1]:o[3], o[0]:o[2]], b[n[1]:n[3], n[0]:n[2]]))
rep = dict(res); rep['max_coord_error_px'] = round(maxerr, 4)
json.dump(rep, open(os.path.join(OUT, 'reports', 'yolo_check.json'), 'w'), indent=1); print(json.dumps(rep, indent=1))
# (e) visual sheet: composite shown as-is (ref=blue, inspected=green, diff=red in OpenCV BGR order)
tiles = []
for name in ['20085000_s0', '20085000_ai', '20085000_ar']:
    im = cv2.imread(os.path.join(Y, 'images', 'train_aug', name + '.png'))
    for c, x1, y1, x2, y2 in read_yolo(os.path.join(Y, 'labels', 'train_aug', name + '.txt')):
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(im, str(c), (int(x1), int(y1) - 3), 0, 0.5, (255, 255, 255), 1)
    t = tf.get(name, {}); cv2.putText(im, f"{name} dx={t.get('dx', 0)} dy={t.get('dy', 0)} rot={t.get('rot_deg', 0)}", (8, 24), 0, 0.7, (0, 165, 255), 2)
    tiles.append(im)
cv2.imwrite(os.path.join(OUT, 'qa', 'yolo_check.png'), cv2.resize(np.hstack(tiles), None, fx=0.6, fy=0.6))
