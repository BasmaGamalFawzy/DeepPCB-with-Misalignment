"""Step 2b - (i) Integer-pixel alignment audit of every ORIGINAL pair: binarise template and inspected image,
find the (dx,dy) minimising XOR pixel count (search +-3, widened to +-8 when the optimum sits on the border).
A pair is flagged `native_misaligned` when the best offset is non-zero AND it lowers the XOR count by >=1.5x
versus (0,0) (so defects alone cannot explain it).
(ii) Confirm hash-flagged near-duplicate templates at pixel level: best +-8 px shift, then the minimum of the
Jaccard overlaps of the two phases (copper / background). Near-uniform templates (minority phase < 2%)
are reported separately because almost any two of them look 'identical'."""
import os, csv, json, collections
import numpy as np, cv2
ROOT = os.environ.get('PCB_ROOT'); OUT = os.environ.get('PCB_OUT')
pairs = {p['pair_id']: p for p in csv.DictReader(open(os.path.join(OUT, 'index', 'pairs.csv')))}
def binimg(p): return cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_GRAYSCALE) > 127
M = 8
def xor_at(a, b, dx, dy):
    A = a[M:-M, M:-M]; B = b[M + dy:b.shape[0] - M + dy, M + dx:b.shape[1] - M + dx]
    return np.count_nonzero(A ^ B), A, B
def search(a, b, r):
    best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x = xor_at(a, b, dx, dy)[0]
            if best is None or x < best[0]: best = (x, dx, dy)
    return best

rows = []
for pid, p in pairs.items():
    a, b = binimg(p['temp']), binimg(p['test'])
    x0 = xor_at(a, b, 0, 0)[0]; x, dx, dy = search(a, b, 3)
    if max(abs(dx), abs(dy)) == 3: x, dx, dy = search(a, b, 8)
    ratio = x0 / max(x, 1)
    rows.append(dict(pair_id=pid, group=p['group'], official_split=p['official_split'], native_dx=dx, native_dy=dy,
                     xor_frac_at_0=round(x0 / 624 ** 2, 5), xor_frac_best=round(x / 624 ** 2, 5), improvement=round(ratio, 3),
                     native_misaligned=int((dx, dy) != (0, 0) and ratio >= 1.5)))
with open(os.path.join(OUT, 'index', 'original_pair_alignment.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
mis = [r for r in rows if r['native_misaligned']]

dups = list(csv.DictReader(open(os.path.join(OUT, 'index', 'near_duplicates_template_dhash.csv'))))
conf = []; blank = {}
for pid, p in pairs.items():
    t = binimg(p['temp']); blank[pid] = float(min(t.mean(), 1 - t.mean())) < 0.02
for d in dups:
    if int(d['hamming']) > 12: continue
    a, b = binimg(pairs[d['a']]['temp']), binimg(pairs[d['b']]['temp'])
    x, dx, dy = search(a, b, 8); _, A, B = xor_at(a, b, dx, dy)
    j1 = np.count_nonzero(A & B) / max(np.count_nonzero(A | B), 1)
    j0 = np.count_nonzero(~A & ~B) / max(np.count_nonzero(~A | ~B), 1)
    d.update(best_dx=dx, best_dy=dy, jaccard_min=round(min(j0, j1), 4),
             near_blank=int(blank[d['a']] or blank[d['b']]),
             duplicate=int(min(j0, j1) >= 0.9 and not (blank[d['a']] or blank[d['b']])))
    conf.append(d)
with open(os.path.join(OUT, 'index', 'near_duplicates_confirmed.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(conf[0])); w.writeheader(); w.writerows(conf)
true_dup = [d for d in conf if d['duplicate']]
rep = dict(native_misaligned_n=len(mis),
           native_misaligned_by_split=dict(collections.Counter(r['official_split'] for r in mis)),
           native_misaligned_by_group=dict(collections.Counter(r['group'] for r in mis)),
           native_offset_max_abs=max([max(abs(r['native_dx']), abs(r['native_dy'])) for r in mis] or [0]),
           native_offset_hist=dict(collections.Counter(max(abs(r['native_dx']), abs(r['native_dy'])) for r in mis)),
           near_blank_templates=sum(blank.values()),
           hash_candidates_checked=len(conf), confirmed_duplicates=len(true_dup),
           confirmed_by_splits=dict(collections.Counter(f"{d['split_a']}-{d['split_b']}" for d in true_dup)),
           confirmed_cross_group=sum(d['group_a'] != d['group_b'] for d in true_dup))
json.dump(rep, open(os.path.join(OUT, 'reports', 'verify_dups_alignment.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))
for d in true_dup:
    if d['group_a'] != d['group_b'] or 'test' in (d['split_a'], d['split_b']): print(d)
