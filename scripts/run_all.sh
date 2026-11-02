#!/usr/bin/env bash
# Rebuild everything from the raw DeepPCB folder. Usage:  bash scripts/run_all.sh
# PCB_ROOT = .../DeepPCB-master/DeepPCB-master/PCBData   PCB_OUT = this MyData folder
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export PCB_ROOT="${PCB_ROOT:-$HERE/../DeepPCB-master/DeepPCB-master/PCBData}"
export PCB_OUT="${PCB_OUT:-$HERE}"
for s in 01_inspect 02_duplicates_alignment 03_verify_dups_alignment 04_make_splits 05_make_shift_sets 06_qa_checks 07_export_coco; do
  echo "== $s"; python3 "$HERE/scripts/$s.py"
done
echo "== 08_make_yolo (4 parallel shards)"
for i in 0 1 2 3; do SHARD=$i NSHARD=4 python3 "$HERE/scripts/08_make_yolo.py" & done; wait
python3 - <<'PY'
import glob, csv, os
d = os.path.join(os.environ['PCB_OUT'], 'yolo'); rows = []
for f in sorted(glob.glob(os.path.join(d, 'train_aug_transforms_part*.csv'))): rows += list(csv.DictReader(open(f))); os.remove(f)
rows.sort(key=lambda r: r['image'])
w = csv.DictWriter(open(os.path.join(d, 'train_aug_transforms.csv'), 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
PY
echo "== 09_check_yolo"; python3 "$HERE/scripts/09_check_yolo.py"
