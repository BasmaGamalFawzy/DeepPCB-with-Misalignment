"""Step 10 - Report figures in the style of the original DeepPCB figures: white-background images with
class-coloured boxes. Difference panels draw pixels that differ between reference and inspected in red on
white, over a light-grey copy of the reference for context (no false-colour overlays).

Writes to report/fig/: native_misalignment_white.png and check_s5x_white.png (3 example pairs each) and
*_row.png (one row, sized for the two-page report). Every figure is also written as *_bw.png in pure black and
white: boxes are black/white outlines tagged with the class number, differing pixels are black.
Needs numpy + Pillow."""
import os, csv, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
OUT = os.environ.get('PCB_OUT', os.path.join(os.path.dirname(__file__), '..'))
S = os.path.join(OUT, 'sets'); FIG = os.path.join(OUT, 'report', 'fig'); os.makedirs(FIG, exist_ok=True)
NAMES = {1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'copper', 6: 'pin-hole'}
COLORS = {1: (230, 25, 75), 2: (0, 130, 200), 3: (245, 130, 48), 4: (60, 180, 75), 5: (145, 30, 180), 6: (240, 50, 230)}
SCALE, PAD, HEAD = 0.5, 10, 26          # panels are 320 px; PAD between panels; HEAD = column-title height
BW = False                              # set by make(); True = black-and-white variant

def font(size, bold=False):
    for f in (('arialbd.ttf' if bold else 'arial.ttf'), 'DejaVuSans.ttf'):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()

imgs = {r['image_id']: r for r in csv.DictReader(open(os.path.join(OUT, 'index', 'images.csv')))}
labs = collections.defaultdict(list)
for l in csv.DictReader(open(os.path.join(OUT, 'index', 'labels.csv'))): labs[l['image_id']].append(l)
splits = {r['pair_id']: r for r in csv.DictReader(open(os.path.join(OUT, 'index', 'splits.csv')))}

def load(rel): return np.asarray(Image.open(os.path.join(S, rel)).convert('L'))

def tag(d, x, y, k, f):
    """Black tag with the white class number; (x, y) = bottom-left corner."""
    w = d.textlength(str(k), font=f) + 4
    d.rectangle([x, y - 13, x + w, y], fill='black'); d.text((x + 2, y - 13), str(k), fill='white', font=f)

def photo(a, boxes=()):
    """Grey image as RGB with class boxes (box coords in original 640 px)."""
    im = Image.fromarray(a).convert('RGB')
    im = im.resize((int(a.shape[1] * SCALE), int(a.shape[0] * SCALE)), Image.NEAREST)
    d = ImageDraw.Draw(im); ft = font(11, True)
    for l in boxes:
        x1, y1, x2, y2 = (float(l[k]) * SCALE for k in ('x1', 'y1', 'x2', 'y2'))
        if BW:   # black outer + white inner outline stays visible on both copper and background
            d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline='black', width=2)
            d.rectangle([x1, y1, x2, y2], outline='white', width=1)
            tag(d, x1 - 2, y1 - 2, int(l['cls']), ft)
        else:
            d.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=COLORS[int(l['cls'])], width=2)
    return im

def diff(ref, ins):
    """White background; differing pixels in red over light-grey reference copper (BW: black only)."""
    out = np.full(ref.shape + (3,), 255, np.uint8)
    if not BW: out[ref < 128] = (215, 215, 215)
    out[(ref < 128) != (ins < 128)] = (0, 0, 0) if BW else (220, 0, 0)
    return Image.fromarray(out).resize((int(ref.shape[1] * SCALE), int(ref.shape[0] * SCALE)), Image.NEAREST)

def sheet(rows, titles, path, legend=True):
    """rows: list of (row_label, [panels]); titles: column titles."""
    path = path.replace('.png', '_bw.png') if BW else path
    pw = rows[0][1][0].width; ph = rows[0][1][0].height; nc = len(titles)
    lab_h, leg_h = (22 if any(lab for lab, _ in rows) else 0), (30 if legend else 0)
    W = nc * pw + (nc + 1) * PAD
    H = HEAD + len(rows) * (lab_h + ph + PAD) + leg_h
    canvas = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(canvas)
    ft, fl, fs, ftag = font(17, True), font(15), font(14), font(13, True)
    frame = 'black' if BW else (170, 170, 170)
    for c, t in enumerate(titles):
        x = PAD + c * (pw + PAD); d.text((x + pw / 2, 6), t, fill='black', font=ft, anchor='ma')
    y = HEAD
    for label, panels in rows:
        d.text((PAD, y + 3), label, fill='black' if BW else (60, 60, 60), font=fl); y += lab_h
        for c, p in enumerate(panels):
            x = PAD + c * (pw + PAD); canvas.paste(p, (x, y)); d.rectangle([x - 1, y - 1, x + pw, y + ph], outline=frame)
        y += ph + PAD
    x = PAD
    for k in (sorted(NAMES) if legend else []):
        if BW: tag(d, x, y + 23, k, ftag); x += 16
        else: d.rectangle([x, y + 8, x + 22, y + 22], outline=COLORS[k], width=3); x += 28
        d.text((x, y + 7), NAMES[k], fill='black', font=fs); x += d.textlength(NAMES[k], font=fs) + 22
    if legend:
        d.rectangle([x, y + 8, x + 22, y + 22], fill='black' if BW else (220, 0, 0)); x += 28
        d.text((x, y + 7), 'pixels that differ', fill='black', font=fs)
    canvas.save(path); print('wrote', path, canvas.size)

def make():
    # Figure A: natively misaligned pairs, as supplied (s0)
    rows = []
    for pid in ('13000015', '50600013', '12000085'):
        r = splits[pid]; i = imgs[f'{pid}_s0']
        ref, ins = load(i['ref_path']), load(i['inspected_path'])
        rows.append((f"pair {pid} ({r['split']}): best offset dx={r['native_dx']}, dy={r['native_dy']} px",
                     [photo(ref), photo(ins, labs[i['image_id']]), diff(ref, ins)]))
    sheet(rows, ['Template (reference)', 'Inspected + labels', 'Difference'], os.path.join(FIG, 'native_misalignment_white.png'))

    # Figure B: label check for the 5 px horizontal shift
    rows = []
    for pid in ('20085315', '12300244', '12300112'):
        i0, i5 = imgs[f'{pid}_s0'], imgs[f'{pid}_s5x']
        ref, ins0, ins5 = load(i0['ref_path']), load(i0['inspected_path']), load(i5['inspected_path'])
        rows.append((f"pair {pid}: shift dx={int(float(i5['dx']))}, dy={int(float(i5['dy']))} px",
                     [photo(ref), photo(ins0, labs[i0['image_id']]), photo(ins5, labs[i5['image_id']]), diff(ref, ins5)]))
    sheet(rows, ['Template (reference)', 'Inspected s0 + labels', 'Shifted s5x + labels', 'Difference'],
          os.path.join(FIG, 'check_s5x_white.png'))

    # One-row versions for the two-page report: two misaligned pairs side by side; one pair for the 5 px check
    rows = []
    for pid in ('50600013', '12000085'):
        r = splits[pid]; i = imgs[f'{pid}_s0']
        ref, ins = load(i['ref_path']), load(i['inspected_path'])
        rows.append((pid, r, [photo(ref), photo(ins, labs[i['image_id']]), diff(ref, ins)]))
    titles = []
    for pid, r, _ in rows:
        titles += [f'{pid}: template', 'inspected + labels', f"difference (dx={r['native_dx']}, dy={r['native_dy']})"]
    sheet([('', rows[0][2] + rows[1][2])], titles, os.path.join(FIG, 'native_misalignment_white_row.png'))
    pid = '12300244'; i0, i5 = imgs[f'{pid}_s0'], imgs[f'{pid}_s5x']
    ref, ins0, ins5 = load(i0['ref_path']), load(i0['inspected_path']), load(i5['inspected_path'])
    sheet([('', [photo(ref), photo(ins0, labs[i0['image_id']]), photo(ins5, labs[i5['image_id']]), diff(ref, ins5)])],
          [f'{pid}: template', 'inspected s0 + labels', f"shifted s5x (dx={int(float(i5['dx']))}) + labels", 'difference'],
          os.path.join(FIG, 'check_s5x_white_row.png'))

def labelled(a, boxes, extra=None):
    """Full-resolution panel in the DeepPCB figure style: green boxes, class name in a white tag above each box.
    extra: optional mask drawn first (pixels that differ from the template)."""
    rgb = np.stack([a] * 3, -1).copy()
    if extra is not None:
        rgb[(a < 128)] = (255, 255, 255) if BW else (215, 215, 215)      # copper shown light so differences stand out
        rgb[extra] = (0, 0, 0) if BW else (220, 0, 0)
    im = Image.fromarray(rgb.astype(np.uint8)); d = ImageDraw.Draw(im); f = font(17)
    B = [tuple(float(l[k]) for k in ('x1', 'y1', 'x2', 'y2')) for l in boxes]
    for x1, y1, x2, y2 in B:
        if BW:
            d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline='black', width=3)
            d.rectangle([x1, y1, x2, y2], outline='white', width=1)
        else:
            d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(0, 200, 0), width=3)
    H, W = a.shape
    hit = lambda r, s: not (r[2] < s[0] or s[2] < r[0] or r[3] < s[1] or s[3] < r[1])
    taken = [(x1 - 4, y1 - 4, x2 + 4, y2 + 4) for x1, y1, x2, y2 in B]
    for (x1, y1, x2, y2), l in zip(B, boxes):       # tag above the box; else below, left, right - first free spot
        name = NAMES[int(l['cls'])]; w = d.textlength(name, font=f) + 10; h = 23
        spots = [(x1 - 3, y1 - h - 6), (x1 - 3, y2 + 6), (x1 - w - 6, y1), (x2 + 6, y1)]
        spots = [(min(max(x, 0), W - w), min(max(y, 0), H - h)) for x, y in spots]
        own = (x1 - 4, y1 - 4, x2 + 4, y2 + 4)
        free = [(x, y) for x, y in spots if not any(hit((x, y, x + w, y + h), t) for t in taken if t != own)]
        tx, ty = (free or spots)[0]
        taken.append((tx, ty, tx + w, ty + h))
        d.rectangle([tx, ty, tx + w, ty + h], fill='white', outline='black'); d.text((tx + 5, ty + 2), name, fill='black', font=f)
    return im

def defect_shift_demo(pid='12300244'):
    """One pair: as supplied with its defects; the 5 px shifted copy; and its difference from the template."""
    i0, i5 = imgs[f'{pid}_s0'], imgs[f'{pid}_s5x']
    ref, ins0, ins5 = load(i0['ref_path']), load(i0['inspected_path']), load(i5['inspected_path'])
    dx = int(float(i5['dx']))
    panels = [labelled(ins0, labs[i0['image_id']]), labelled(ins5, labs[i5['image_id']]),
              labelled(ins5, labs[i5['image_id']], extra=(ref < 128) != (ins5 < 128))]
    titles = ['(a) inspected image and its defects', f'(b) same image shifted {abs(dx)} px',
              '(c) shifted image vs. template']
    pw, ph, head, gap = panels[0].width, panels[0].height, 34, 60
    canvas = Image.new('RGB', (3 * pw + 2 * gap + 20, ph + head + 10), 'white'); d = ImageDraw.Draw(canvas); ft = font(22, True)
    for c, (p, t) in enumerate(zip(panels, titles)):
        x = 10 + c * (pw + gap); canvas.paste(p, (x, head)); d.rectangle([x - 1, head - 1, x + pw, head + ph], outline='black')
        d.text((x + pw / 2, 4), t, fill='black', font=ft, anchor='ma')
    path = os.path.join(FIG, 'defect_shift_demo' + ('_bw' if BW else '') + '.png')
    canvas.save(path); print('wrote', path, canvas.size)

for BW in (False, True):
    make()
    defect_shift_demo()
