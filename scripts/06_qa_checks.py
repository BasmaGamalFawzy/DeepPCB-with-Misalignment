"""Step 5 - Automatic label/geometry checks + visual check sheets.
(a) integer shifts are pixel-exact (interior compared with a numpy slice shift)
(b) for integer shifts the pixels under every moved box equal the pixels under the original box
(c) residual set: ECC-estimated transform (inspected vs its own s0 image) matches the recorded parameters
(d) every non-ignored box lies fully inside the valid region [MARGIN, 640-MARGIN]
(e) visual sheets: reference | s0 | shifted, boxes drawn, for random pairs of each condition"""
import os, sys, csv, json, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); import pcb_shift as T
OUT = os.environ.get('PCB_OUT'); S = os.path.join(OUT, 'sets'); m = T.MARGIN
imgs = list(csv.DictReader(open(os.path.join(OUT, 'index', 'images.csv'))))
labs = collections.defaultdict(list)
for l in csv.DictReader(open(os.path.join(OUT, 'index', 'labels.csv'))): labs[l['image_id']].append(l)
s0 = {x['pair_id']: x for x in imgs if x['condition'] == 's0'}
rd = lambda p: cv2.imread(os.path.join(S, p), cv2.IMREAD_GRAYSCALE)
fail = collections.Counter(); n = collections.Counter(); ecc_err = []
for x in imgs:
    if x['condition'] == 's0': continue
    a = rd(s0[x['pair_id']]['inspected_path']); b = rd(x['inspected_path'])
    if x['condition'] != 'r':
        dx, dy = int(float(x['dx'])), int(float(x['dy'])); H, W = a.shape
        A = a[m:H - m, m:W - m]; B = b[m + dy:H - m + dy, m + dx:W - m + dx]
        n['pixel_exact'] += 1; fail['pixel_exact'] += int(not np.array_equal(A, B))
        for l0, l1 in zip(labs[s0[x['pair_id']]['image_id']], labs[x['image_id']]):
            if l0['ignore'] == '1': continue
            x0, y0, x1_, y1_ = [int(float(l0[k])) for k in ('x1', 'y1', 'x2', 'y2')]
            u0, v0, u1, v1 = [int(round(float(l1[k]))) for k in ('x1', 'y1', 'x2', 'y2')]
            n['box_content'] += 1
            fail['box_content'] += int(not np.array_equal(a[y0:y1_, x0:x1_], b[v0:v1, u0:u1]))
    elif len(ecc_err) < 120:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(cv2.GaussianBlur(a, (5, 5), 1).astype(np.float32),
                                           cv2.GaussianBlur(b, (5, 5), 1).astype(np.float32), warp, cv2.MOTION_EUCLIDEAN,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
            Mtrue = T.affine(float(x['dx']), float(x['dy']), float(x['rot_deg']), a.shape)
            Mest = warp   # ECC(template=s0, input=shifted) returns W with shifted(W x) = s0(x), i.e. our M
            c = np.array([[0, 0, 1], [639, 0, 1], [0, 639, 1], [639, 639, 1], [320, 320, 1]], float)
            ecc_err.append(float(np.abs(c @ Mtrue.T - c @ Mest.T).max()))
        except cv2.error: fail['ecc_not_converged'] += 1
    for l in labs[x['image_id']]:
        if l['ignore'] == '1': continue
        n['in_valid_region'] += 1
        fail['in_valid_region'] += int(not (float(l['x1']) >= m and float(l['y1']) >= m and float(l['x2']) <= 640 - m and float(l['y2']) <= 640 - m))
rep = dict(checks={k: f"{n[k] - fail[k]}/{n[k]} pass" for k in n},
           residual_ecc_corner_error_px=dict(n=len(ecc_err), median=round(float(np.median(ecc_err)), 3),
                                             p95=round(float(np.percentile(ecc_err, 95)), 3), max=round(float(max(ecc_err)), 3)),
           ecc_not_converged=fail['ecc_not_converged'])
json.dump(rep, open(os.path.join(OUT, 'reports', 'qa_checks.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))

# visual check sheets
rng = np.random.default_rng(1); os.makedirs(os.path.join(OUT, 'qa'), exist_ok=True)
COL = {1: (0, 0, 255), 2: (0, 200, 0), 3: (255, 0, 0), 4: (0, 200, 255), 5: (255, 0, 255), 6: (255, 255, 0)}
def draw(img, ls):
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(v, (m, m), (639 - m, 639 - m), (128, 128, 128), 1)
    for l in ls:
        p = tuple(int(round(float(l[k]))) for k in ('x1', 'y1')); q = tuple(int(round(float(l[k]))) for k in ('x2', 'y2'))
        cv2.rectangle(v, p, q, (160, 160, 160) if l['ignore'] == '1' else COL[int(l['cls'])], 2)
    return v
for cond in ['s1x', 's2y', 's3x', 's5x', 's5y', 'r']:
    cand = [x for x in imgs if x['condition'] == cond and x['split'] == 'test']
    rowsv = []
    for x in rng.choice(cand, 3, replace=False):
        z = s0[x['pair_id']]
        ref = draw(rd(z['ref_path']), []); a = draw(rd(z['inspected_path']), labs[z['image_id']])
        b = draw(rd(x['inspected_path']), labs[x['image_id']])
        diff = cv2.cvtColor(cv2.absdiff(rd(z['ref_path']), rd(x['inspected_path'])), cv2.COLOR_GRAY2BGR)
        row = np.hstack([ref, a, b, diff])
        cv2.putText(row, f"{x['pair_id']} {cond} dx={x['dx']} dy={x['dy']} rot={x['rot_deg']} | ref | s0+boxes | shifted+boxes | |ref-shifted|",
                    (8, 24), 0, 0.7, (0, 140, 255), 2)
        rowsv.append(row)
    sheet = cv2.resize(np.vstack(rowsv), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, 'qa', f'check_{cond}.png'), sheet)
