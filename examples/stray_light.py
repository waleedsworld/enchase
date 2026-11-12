#!/usr/bin/env python3
"""Instrumental profile and stray light: fit, correction, and the paper figure.

Fits ONE instrumental profile to the disk-centre spectra of both windows
(Gaussian core plus a broad Gaussian holding a fraction of the light, see
docs/CALIBRATION.md section 8), removes the disk-centre stray light from a
cutout around a sunspot, and draws the figure of the paper: the spot in the
Halpha and Fe I line cores, observed above the diagonal and corrected below
it, with the intensity along one row underneath.

    python examples/stray_light.py RSM20230329T021231_0000_HA.fits RSM20230329T021231_0000_FE.fits

Options:
    --center Y X      cutout centre in pixels (default: the 2023-03-29 spot, 988 1352)
    --half N          cutout half width in pixels (default 55)
    --free            fit core, broad width and fraction freely (about 5 s)
                      instead of holding them at 0.100 A, 1.8 A and 14.5 %
    --atlas-file F    FTS atlas idlsave, if ISPy is not installed
    --out NAME        output stem (default fig_straylight), writes .png and .pdf

Needs the FTS atlas from ISPy (pip install 'enchase[atlas]') or --atlas-file.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import chase

T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1f s] %s" % (time.time() - T0, msg), flush=True)


def pixel_scale(header) -> float:
    """Arcsec per pixel from CDELT1/2, else from R_SUN (959.63 arcsec)."""
    for key in ("CDELT1", "CDELT2"):
        try:
            v = abs(float(header.get(key)))
        except (TypeError, ValueError):
            continue
        if 0.2 < v < 5.0:
            return v
    return 959.63 / float(header["R_SUN"])


def letter(ax, s):
    ax.text(0.04, 0.95, s, transform=ax.transAxes, va="top",
            bbox=dict(fc="w", ec="none", pad=2))


def main(ha_file, fe_file, center, half, free, atlas_file, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    files = {"HA": ha_file, "FE": fe_file}

    # ------------------------------------------------ fit at disk centre
    log("disk-centre spectra, 51 x 51 px")
    wav_dc, spec_dc = {}, {}
    for b in ("HA", "FE"):
        box, wav, _ = chase.extract_disk_center(files[b], size=51)
        wav_dc[b], spec_dc[b] = np.asarray(wav, float), np.asarray(box, float).mean(axis=(1, 2))

    kw = dict(atlas_file=atlas_file)
    if free:
        log("free fit of one profile to both windows")
        kw.update(fwhm=np.arange(0.080, 0.1201, 0.005),
                  broad_fwhm=np.arange(1.0, 3.01, 0.2),
                  frac=np.arange(0.08, 0.2001, 0.01),
                  shifts=np.arange(-0.04, 0.0401, 0.002), verbose=True)
    else:
        log("one profile for both windows, held at 0.100 A, 1.8 A and 14.5 %")
    fit = chase.fit_instrumental_profile([wav_dc["HA"], wav_dc["FE"]],
                                         [spec_dc["HA"], spec_dc["FE"]], **kw)
    win = dict(zip(("HA", "FE"), fit["windows"]))
    log("core %.3f A, broad term %.2f A with %.1f %% of the light"
        % (fit["fwhm"], fit["broad_fwhm"], 100 * fit["frac"]))
    for b in ("HA", "FE"):
        log("   %s rms %.4f of the continuum, offset %+.3f A" % (b, win[b]["rms"], win[b]["shift"]))

    # ------------------------------------------------ cutout and correction
    yc, xc = center
    region = (yc - half, yc + half + 1, xc - half, xc + half + 1)
    obs, cor, core = {}, {}, {}
    for b in ("HA", "FE"):
        log("reading the %s cutout" % b)
        cube, hdr, wav = chase.load_cube(files[b], region=region)
        wav = np.asarray(wav, float)
        trans = np.polyval(win[b]["poly"], wav - win[b]["wmean"])
        obs[b] = np.asarray(cube, float) * trans[:, None, None]
        cor[b] = chase.remove_stray_light(obs[b], wav, win[b]["profile"],
                                          frac=fit["frac"], broad_fwhm=fit["broad_fwhm"])
        core[b] = int(np.argmin(np.median(obs[b], axis=(1, 2))))
        log("   %s core plane %d at %.3f A" % (b, core[b], wav[core[b]] + win[b]["shift"]))
        if b == "FE":
            header = hdr

    # ------------------------------------------------ figure
    pix = pixel_scale(header)
    x0d, y0d = float(header["CRPIX1"]) - 1.0, float(header["CRPIX2"]) - 1.0
    ny, nx = obs["FE"].shape[1:]
    ext = [(region[2] - x0d) * pix, (region[3] - 1 - x0d) * pix,
           (region[0] - y0d) * pix, (region[1] - 1 - y0d) * pix]
    jrow = half
    xarc = ext[0] + np.arange(nx) * pix
    below = (np.arange(ny)[:, None] / (ny - 1.0)) < (np.arange(nx)[None, :] / (nx - 1.0))
    top = max(max(obs[b][core[b], jrow].max(), cor[b][core[b], jrow].max()) for b in obs)

    fig = plt.figure(figsize=(7.1, 6.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.72])
    for k, b in enumerate(("HA", "FE")):
        a = fig.add_subplot(gs[0, k])
        img = np.where(below, cor[b][core[b]], obs[b][core[b]])
        im = a.imshow(img, origin="lower", extent=ext, cmap="gray",
                      vmin=0.0, vmax=1.02 * img.max(), interpolation="nearest")
        a.plot([ext[0], ext[1]], [ext[2], ext[3]], color="k", lw=0.8)
        a.axhline((yc - y0d) * pix, color="k", lw=0.8, ls=":")
        a.set_xlim(ext[0], ext[1])
        a.set_ylim(ext[2], ext[3])
        a.set_ylabel("solar Y (arcsec)")
        plt.setp(a.get_xticklabels(), visible=False)
        letter(a, "%s)" % "ac"[k])
        cb = fig.colorbar(im, ax=a, location="top", fraction=0.05, pad=0.04)
        cb.set_label("$I/I_{c}$")

        q = fig.add_subplot(gs[1, k])
        q.plot(xarc, obs[b][core[b], jrow], color="C1", lw=1.2, label="observed")
        q.plot(xarc, cor[b][core[b], jrow], color="C0", lw=1.2, label="corrected")
        q.set_xlim(ext[0], ext[1])
        q.set_ylim(0.0, 1.28 * top)
        q.set_xlabel("solar X (arcsec)")
        q.set_ylabel("$I/I_{c}$")
        plt.setp(q.get_xticklabels(), rotation=30)
        letter(q, "%s)" % "bd"[k])
        if k == 0:
            q.legend(loc="upper right", frameon=False, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    log("wrote %s.png and %s.pdf" % (out, out))

    # ------------------------------------------------ numbers for the text
    edge = lambda a: np.median(np.r_[a[:10], a[-10:]])
    print()
    print("%-10s %22s %22s %22s" % ("", "umbra (min along row)", "surroundings", "umbra / surroundings"))
    for b in ("HA", "FE"):
        o, c = obs[b][core[b], jrow], cor[b][core[b], jrow]
        print("%-10s %10.4f -> %8.4f %10.4f -> %8.4f %10.3f -> %8.3f"
              % (b + " core", o.min(), c.min(), edge(o), edge(c),
                 o.min() / edge(o), c.min() / edge(c)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ha_file")
    ap.add_argument("fe_file")
    ap.add_argument("--center", type=int, nargs=2, default=[988, 1352], metavar=("Y", "X"))
    ap.add_argument("--half", type=int, default=55)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--atlas-file", default=None)
    ap.add_argument("--out", default="fig_straylight")
    a = ap.parse_args()
    main(a.ha_file, a.fe_file, tuple(a.center), a.half, a.free, a.atlas_file, a.out)
