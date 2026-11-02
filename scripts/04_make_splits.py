"""Step 3 - Grouped train/validation split.
Unit of grouping = DeepPCB group folder (the only source-board identifier; all confirmed near-duplicate
templates fall inside one group, which supports this choice). Official 1000/500 split is kept.
We enumerate every subset of trainval groups whose size is 17-23% of trainval and choose the one whose
per-class box proportions are closest (L1) to the whole trainval set; ties -> more groups."""
import os, csv, json, itertools, collections
OUT = os.environ.get('PCB_OUT')
pairs = list(csv.DictReader(open(os.path.join(OUT, 'index', 'pairs.csv'))))
boxes = list(csv.DictReader(open(os.path.join(OUT, 'index', 'boxes.csv'))))
align = {r['pair_id']: r for r in csv.DictReader(open(os.path.join(OUT, 'index', 'original_pair_alignment.csv')))}
tv = [p for p in pairs if p['official_split'] == 'trainval']
gsize = collections.Counter(p['group'] for p in tv)
gcls = collections.defaultdict(collections.Counter)
for b in boxes:
    if b['official_split'] == 'trainval': gcls[b['group']][b['cls_name']] += 1
classes = sorted({b['cls_name'] for b in boxes})
def prop(c):
    n = sum(c.values()); return [c[k] / n for k in classes]
ref = prop(sum(gcls.values(), collections.Counter()))
cands = []
for r in range(1, len(gsize)):
    for S in itertools.combinations(sorted(gsize), r):
        n = sum(gsize[g] for g in S)
        if not 0.17 * len(tv) <= n <= 0.23 * len(tv): continue
        c = sum((gcls[g] for g in S), collections.Counter())
        l1 = sum(abs(a - b) for a, b in zip(prop(c), ref))
        cands.append(dict(groups=list(S), n_pairs=n, frac=round(n / len(tv), 3), l1_class_prop=round(l1, 4)))
cands.sort(key=lambda d: (round(d['l1_class_prop'], 3), -len(d['groups'])))
val_groups = set(cands[0]['groups'])
rows = []
for p in pairs:
    split = 'test' if p['official_split'] == 'test' else ('val' if p['group'] in val_groups else 'train')
    a = align[p['pair_id']]
    rows.append(dict(pair_id=p['pair_id'], group=p['group'], official_split=p['official_split'], split=split,
                     temp=p['temp'], test=p['test'], label=p['label'], n_boxes=p['n_boxes'],
                     native_dx=a['native_dx'], native_dy=a['native_dy'], native_misaligned=a['native_misaligned']))
# duplicate leakage check against the new partition
dups = [d for d in csv.DictReader(open(os.path.join(OUT, 'index', 'near_duplicates_confirmed.csv'))) if d['duplicate'] == '1']
sp = {r['pair_id']: r['split'] for r in rows}
cross = [dict(a=d['a'], b=d['b'], split_a=sp[d['a']], split_b=sp[d['b']]) for d in dups if sp[d['a']] != sp[d['b']]]
test_dup_ids = sorted({x[k] for x in cross for k in ('a', 'b') if sp[x[k]] == 'test'})
for r in rows: r['test_has_trainval_duplicate'] = int(r['pair_id'] in test_dup_ids)
with open(os.path.join(OUT, 'index', 'splits.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cnt = collections.Counter(r['split'] for r in rows)
bc = collections.defaultdict(collections.Counter)
for b in boxes: bc[sp[b['pair_id']]][b['cls_name']] += 1
rep = dict(val_groups=sorted(val_groups), split_sizes=dict(cnt), top5_candidates=cands[:5],
           class_counts={k: dict(v) for k, v in bc.items()},
           groups_per_split={s: sorted({r['group'] for r in rows if r['split'] == s}) for s in cnt},
           test_groups_unseen_in_trainval=sorted({r['group'] for r in rows if r['split'] == 'test'} - {r['group'] for r in rows if r['split'] != 'test'}),
           duplicates_crossing_partitions=cross, test_pairs_with_trainval_duplicate=test_dup_ids,
           native_misaligned_per_split=dict(collections.Counter(r['split'] for r in rows if r['native_misaligned'] == '1')))
json.dump(rep, open(os.path.join(OUT, 'reports', 'splits.json'), 'w'), indent=1)
print(json.dumps(rep, indent=1))
