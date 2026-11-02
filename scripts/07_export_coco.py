"""Step 6 - Export one COCO-style JSON per split/condition (index/coco/<split>_<condition>.json) so any
detector library can load the sets directly. Ignored (boundary-truncated) boxes get iscrowd=1, the COCO
convention for 'do not count as TP or FN'. Each image entry also carries ref_file for two-image input."""
import os, csv, json, collections
OUT = os.environ.get('PCB_OUT'); os.makedirs(os.path.join(OUT, 'index', 'coco'), exist_ok=True)
NAMES = {1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'copper', 6: 'pin-hole'}
imgs = list(csv.DictReader(open(os.path.join(OUT, 'index', 'images.csv'))))
labs = collections.defaultdict(list)
for l in csv.DictReader(open(os.path.join(OUT, 'index', 'labels.csv'))): labs[l['image_id']].append(l)
groups = collections.defaultdict(list)
for x in imgs: groups[(x['split'], x['condition'])].append(x)
for (sp, cond), xs in sorted(groups.items()):
    d = dict(info=dict(description=f'DeepPCB {sp} {cond}', note='file_name=inspected image, ref_file=reference; paths relative to sets/'),
             categories=[dict(id=k, name=v) for k, v in NAMES.items()], images=[], annotations=[])
    aid = 1
    for i, x in enumerate(xs, 1):
        d['images'].append(dict(id=i, file_name=x['inspected_path'], ref_file=x['ref_path'], width=640, height=640,
                                pair_id=x['pair_id'], group=x['group'], dx=float(x['dx']), dy=float(x['dy']), rot_deg=float(x['rot_deg'])))
        for l in labs[x['image_id']]:
            x1, y1, x2, y2 = (float(l[k]) for k in ('x1', 'y1', 'x2', 'y2'))
            d['annotations'].append(dict(id=aid, image_id=i, category_id=int(l['cls']), bbox=[x1, y1, x2 - x1, y2 - y1],
                                         area=(x2 - x1) * (y2 - y1), iscrowd=int(l['ignore']))); aid += 1
    json.dump(d, open(os.path.join(OUT, 'index', 'coco', f'{sp}_{cond}.json'), 'w'))
print(len(groups), 'COCO files written')
