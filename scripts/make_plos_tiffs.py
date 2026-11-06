#!/usr/bin/env python3
"""Produce the figure files PLOS ONE accepts, from the PNGs the figure
scripts emit plus the two hand-drawn diagrams kept from the submission.

PLOS requires: TIFF or EPS; RGB 8 bit/channel or grayscale, no alpha;
LZW compression; flattened, single page; width 789-2250 px and height at
most 2625 px at 300 dpi; 300-600 dpi; at most 10 MB per file. It also asks
for Arial, Times or Symbol at 8-12 pt in the final size; the charts set
Arial and are drawn at their printed width, so nothing is resampled and
the sizes they are authored at are the sizes that print. This script does
not check the text, only the container, so changing a figure's width in
make_figures.py means checking the labels again.

File names follow the order the figures appear in, so Fig5.tif is
Figure 5. Run from anywhere:  python3 scripts/make_plos_tiffs.py
"""
import os, sys
from PIL import Image

MIN_W, MAX_W, MAX_H, DPI, MAX_BYTES = 789, 2250, 2625, 300, 10 * 1024 * 1024
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = [os.path.join(ROOT, "figures", "out"), os.path.join(ROOT, "figures", "diagrams")]
OUT = os.path.join(ROOT, "figures", "plos_tiff")


def find(name):
    for d in SRC:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def convert(src, dst):
    im = Image.open(src)
    # Flatten any transparency onto white; PLOS forbids an alpha channel.
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    w, h = im.size
    if w > MAX_W:
        h = max(1, round(h * MAX_W / w)); w = MAX_W
    if h > MAX_H:
        w = max(1, round(w * MAX_H / h)); h = MAX_H
    if (w, h) != im.size:
        im = im.resize((w, h), Image.LANCZOS)
    im.save(dst, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    return im.size


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for i in range(1, 10):
        name = "Fig%d.png" % i
        src = find(name)
        if src is None:
            print("MISSING  %s (looked in %s)" % (name, ", ".join(SRC))); ok = False; continue
        dst = os.path.join(OUT, "Fig%d.tif" % i)
        w, h = convert(src, dst)
        size = os.path.getsize(dst)
        bad = []
        if w < MIN_W: bad.append("width %d below %d" % (w, MIN_W))
        if w > MAX_W: bad.append("width %d above %d" % (w, MAX_W))
        if h > MAX_H: bad.append("height %d above %d" % (h, MAX_H))
        if size > MAX_BYTES: bad.append("file %d bytes above limit" % size)
        print("Fig%d.tif  %4dx%-4d  %3d dpi  %5d KB  %s"
              % (i, w, h, DPI, size // 1024, "; ".join(bad) if bad else "ok"))
        if bad: ok = False
    print("\n%s -> %s" % ("all figures meet the requirements" if ok else "SOME FIGURES FAIL", OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
