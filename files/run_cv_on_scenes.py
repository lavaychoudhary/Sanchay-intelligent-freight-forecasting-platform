#!/usr/bin/env python3
"""
Sanchay - run best.pt on port scenes and produce cv_observations.csv for the numeric pipeline.
NOTE: written against the ultralytics API but NOT executed here (no GPU/ultralytics in my sandbox) - test on 2-3 images first.

  pip install ultralytics pillow pandas numpy torch torchvision

1) CALIBRATE the count (do this first, on your labelled val split):
   python run_cv_on_scenes.py --weights best.pt --eval --images sanchay_32k/yolo_dataset/images/val --labels sanchay_32k/yolo_dataset/labels/val --limit 2000
   -> prints count MAE / bias for several confidence thresholds. Use the conf with the lowest MAE (or bias near 0) as --conf below.

2) PRODUCE observations (file names must be  <Port_Name>_<YYYY-MM-DD>.<ext>, e.g. New_Mangalore_2026-11-10.png):
   python run_cv_on_scenes.py --weights best.pt --scenes ./scenes --conf 0.30 --out cv_observations.csv

3) python sanchay_v3_pipeline.py --cv-obs cv_observations.csv
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
TILE, OVERLAP = 768, 0.2                     # best.pt was trained at imgsz 768

def tiles(w, h):
    step = int(TILE * (1 - OVERLAP)); xs = list(range(0, max(w - TILE, 0) + 1, step)); ys = list(range(0, max(h - TILE, 0) + 1, step))
    if xs[-1] + TILE < w: xs.append(w - TILE)
    if ys[-1] + TILE < h: ys.append(h - TILE)
    return [(max(x, 0), max(y, 0)) for y in ys for x in xs]

def detect(model, img, conf):
    """Tile a large scene, run YOLO, merge duplicate boxes across tile overlaps with NMS. Returns boxes xyxy, scores."""
    import torch, torchvision
    w, h = img.size; allb, alls = [], []
    for x, y in tiles(w, h):
        crop = img.crop((x, y, min(x + TILE, w), min(y + TILE, h))).convert("RGB")
        r = model.predict(crop, imgsz=TILE, conf=conf, iou=0.5, verbose=False)[0]
        if len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy().copy(); b[:, [0, 2]] += x; b[:, [1, 3]] += y
            allb.append(b); alls.append(r.boxes.conf.cpu().numpy())
    if not allb: return np.zeros((0, 4)), np.zeros(0)
    b, s = np.concatenate(allb), np.concatenate(alls)
    keep = torchvision.ops.nms(torch.tensor(b), torch.tensor(s), 0.5).numpy()
    return b[keep], s[keep]

def run_scenes(a):
    from ultralytics import YOLO
    model = YOLO(a.weights); rows = []
    for f in sorted(glob.glob(os.path.join(a.scenes, "*.*"))):
        m = re.match(r"(.+)_(\d{4}-\d{2}-\d{2})\.\w+$", os.path.basename(f))
        if not m: print("skip (bad name):", f); continue
        img = Image.open(f); b, s = detect(model, img, a.conf); w, h = img.size
        area = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / (w * h) if len(b) else np.array([0.0])
        rows.append(dict(date=m.group(2), port_name=m.group(1).replace("_", " "), cv_ship_count=len(b),
                         cv_mean_conf=round(float(s.mean()), 4) if len(s) else np.nan,
                         cv_mean_box_area=round(float(area.mean()), 6), cv_total_box_area=round(float(area.sum()), 6), scene_file=os.path.basename(f)))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out)

def run_eval(a):
    from ultralytics import YOLO
    model = YOLO(a.weights); imgs = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[: a.limit or None]
    confs = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]; pred = {c: [] for c in confs}; true = []
    for f in imgs:
        lf = os.path.join(a.labels, os.path.splitext(os.path.basename(f))[0] + ".txt")
        true.append(sum(1 for _ in open(lf)) if os.path.exists(lf) else 0)
        sc = model.predict(f, imgsz=TILE, conf=min(confs), iou=0.5, verbose=False)[0].boxes.conf.cpu().numpy()
        for c in confs: pred[c].append(int((sc >= c).sum()))
    true = np.array(true); print(f"{len(imgs)} images | mean true count {true.mean():.2f}")
    print("conf   MAE   bias(pred-true)   bias%")
    for c in confs:
        p = np.array(pred[c]); print(f"{c:.2f}  {np.abs(p - true).mean():.3f}  {(p - true).mean():+.3f}  {100 * (p - true).mean() / max(true.mean(), 1e-9):+.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="best.pt"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--images"); ap.add_argument("--labels"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenes"); ap.add_argument("--conf", type=float, default=0.30); ap.add_argument("--out", default="cv_observations.csv"); a = ap.parse_args()
    run_eval(a) if a.eval else run_scenes(a)
