"""Instrumental profile and stray light of CHASE/HIS.

The disk-center profile is compared with the FTS atlas seen through one
instrumental profile for both spectral windows, a narrow Gaussian core plus a
broad Gaussian that carries a fraction ``frac`` of the light,

    T(lambda) * DN  =  (1 - frac) * atlas (x) G(fwhm)  +  frac * atlas (x) G(broad_fwhm)

where (x) is a convolution and T a low-order polynomial for the prefilter
slope. This is the model of Hou et al. (2020, RAA 20, 045, Eq. 1), used for
CHASE by Hong et al. (2022, A&A 668, A9), with their constant stray-light
level replaced by the broad term. With ``broad_fwhm=None`` it is their model
exactly. A constant level needs different values in the two windows (about
10 percent in Fe I and 5 percent in H-alpha), which cannot be right for two
windows recorded in the same exposure on the same detector. The broad term
fits both with one set of numbers.

The stray light is then assumed to be the same over the field of view (Hong
et al. 2022), so it is taken from the disk-center profile and removed from
every pixel,

    I  =  (I_obs - frac * I_dc (x) G(broad_fwhm)) / (1 - frac).

This removes the broad wings of the profile. The narrow core is left in.

Limits. The level is set at disk center, so the correction takes away too
much close to and beyond the limb and must not be used on off-limb features.
Light scattered from the immediate surroundings of a pixel is not treated.

Functions
---------
load_fts_atlas      FTS disk-center atlas, I/Ic (ISPy or an idlsave file)
instrumental_profile_model       right-hand side of the first equation above
fit_instrumental_profile         fit of the profile at disk center, one or several windows
stray_light_veil    frac * I_dc (x) G(broad_fwhm)
remove_stray_light  the correction
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

__all__ = ["load_fts_atlas", "instrumental_profile_model", "fit_instrumental_profile",
           "stray_light_veil", "remove_stray_light",
           "DEFAULT_FWHM", "DEFAULT_BROAD_FWHM", "DEFAULT_FRAC", "CAL_MASK"]

DEFAULT_FWHM = 0.100        # A, Gaussian core (before the detector pixel)
DEFAULT_BROAD_FWHM = 1.8    # A, broad term
DEFAULT_FRAC = 0.145        # fraction of the light in the broad term

# Wavelengths left out of the fit (center, half width, in A). Telluric lines
# and weak blends that differ between the ground-based atlas and CHASE.
CAL_MASK = [(6559.56, 0.14), (6560.22, 0.12), (6561.10, 0.13), (6564.04, 0.12),
            (6564.18, 0.14), (6565.51, 0.12), (6565.88, 0.12), (6568.71, 0.10),
            (6568.83, 0.10)]

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def load_fts_atlas(w0: float, w1: float, atlas_file: Optional[str] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """FTS disk-center atlas between w0 and w1 (A), as (wavelength, I/Ic).

    Wavelengths are in the observer frame (no gravitational redshift taken
    out). Without ``atlas_file`` the atlas shipped with ISPy is used
    (https://github.com/ISP-SST/ISPy). ``atlas_file`` can point at an
    ``fts_disk_center*.idlsave`` file instead.
    """
    if atlas_file is None:
        try:
            from ISPy.spec import atlas as isp_atlas
        except ImportError as e:
            raise ImportError(
                "load_fts_atlas needs ISPy (pip install "
                "git+https://github.com/ISP-SST/ISPy) or atlas_file=...") from e
        fts = isp_atlas.atlas()
        wav, spec, cont = fts.wave, fts.spec, fts.cont
    else:
        from scipy.io import readsav
        sav = {k.lower(): v for k, v in readsav(atlas_file).items()}
        wav = sav["ftswav"]
        spec = sav.get("ftsint_si", sav.get("ftsint"))
        cont = sav.get("ftscnt_si", sav.get("ftscnt"))
    keep = (wav >= w0) & (wav <= w1)
    return (np.asarray(wav[keep], float),
            np.asarray(spec[keep], float) / np.asarray(cont[keep], float))


def _smeared_atlas(atlas_wav, atlas_int, fwhm, pixel):
    """Atlas convolved with a Gaussian of this FWHM and one detector pixel."""
    step = float(np.median(np.diff(atlas_wav)))
    out = gaussian_filter1d(atlas_int, fwhm * _FWHM_TO_SIGMA / step,
                            mode="reflect", truncate=4.0)
    npix = max(1, int(round(pixel / step)))
    return uniform_filter1d(out, npix, mode="reflect")


def instrumental_profile_model(wav, atlas_wav, atlas_int, fwhm: float = DEFAULT_FWHM,
                  frac: float = DEFAULT_FRAC,
                  broad_fwhm: Optional[float] = DEFAULT_BROAD_FWHM,
                  shift: float = 0.0, pixel: Optional[float] = None) -> np.ndarray:
    """Atlas seen through the instrumental profile, I/Ic on ``wav``.

    ``(1 - frac) * atlas (x) G(fwhm) + frac * atlas (x) G(broad_fwhm)``,
    evaluated at ``wav + shift``. ``broad_fwhm=None`` gives the constant
    stray-light level of Hou et al. (2020), ``(1 - frac) * ... + frac``.
    ``pixel`` is the detector pixel in A (default, the step of ``wav``).
    """
    wav = np.asarray(wav, float)
    pixel = float(np.median(np.diff(wav))) if pixel is None else float(pixel)
    core = np.interp(wav + shift, atlas_wav,
                     _smeared_atlas(atlas_wav, atlas_int, fwhm, pixel))
    if broad_fwhm is None:
        return (1.0 - frac) * core + frac
    broad = np.interp(wav + shift, atlas_wav,
                      _smeared_atlas(atlas_wav, atlas_int, broad_fwhm, pixel))
    return (1.0 - frac) * core + frac * broad


def _grid(value):
    return np.atleast_1d(np.asarray([np.nan] if value is None else value, float))


def fit_instrumental_profile(wav, spectrum, atlas: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                fwhm: Union[float, Sequence[float]] = DEFAULT_FWHM,
                frac: Union[float, Sequence[float]] = DEFAULT_FRAC,
                broad_fwhm: Union[None, float, Sequence[float]] = DEFAULT_BROAD_FWHM,
                poly_deg: Optional[int] = None,
                mask: Sequence[Tuple[float, float]] = CAL_MASK,
                shifts: Optional[Sequence[float]] = None,
                atlas_file: Optional[str] = None, verbose: bool = False) -> dict:
    """Fit the instrumental profile to a disk-center spectrum.

    Parameters
    ----------
    wav, spectrum : array (nwav,), or lists of such arrays
        Mean disk-center spectrum of one window, in any units (DN is fine).
        Pass lists, e.g. ``[wav_ha, wav_fe], [spec_ha, spec_fe]``, to fit
        several windows with ONE profile. Each window then gets its own
        wavelength offset and polynomial, and the windows have equal weight.
    atlas : (wavelength, I/Ic), optional
        Reference atlas. Default, :func:`load_fts_atlas` over the windows.
    fwhm, frac, broad_fwhm : float or sequence of float
        Core FWHM (A), fraction of the light in the broad term, and FWHM of
        the broad term (A). A single number holds that parameter fixed, a
        sequence is searched as a grid. ``broad_fwhm=None`` is the constant
        level of Hou et al. (2020), where ``frac`` is their alpha.
    poly_deg : int, optional
        Degree of the transmission polynomial. Default 3 for windows wider
        than 4 A (H-alpha) and 2 otherwise (Fe I).
    mask : sequence of (center, half width)
        Wavelengths left out of the fit.
    shifts : sequence of float, optional
        Wavelength offsets to try (A). Default -0.2 to 0.2 in steps of 0.002.
    verbose : bool
        Print one line per core width while a grid is searched.

    Returns
    -------
    dict with the best ``fwhm``, ``frac``, ``broad_fwhm``, the summed
    ``chi2`` (mean squared residual, added over windows) and ``windows``, a
    list with one dict per window holding ``shift``, ``poly`` and ``wmean``
    (transmission = polyval(poly, wav - wmean)), ``transmission``,
    ``profile`` (= transmission * spectrum, the observed I/Ic), ``model``,
    ``rms`` (of profile - model over the fitted points) and ``keep``. For a
    single window these entries are also at the top level.
    """
    single = np.ndim(wav[0]) == 0
    wavs = [np.asarray(wav, float)] if single else [np.asarray(w, float) for w in wav]
    specs = [np.asarray(spectrum, float)] if single else [np.asarray(s, float) for s in spectrum]
    shifts = np.arange(-0.200, 0.2001, 0.002) if shifts is None else np.asarray(shifts, float)
    fwhms, fracs, broads = _grid(fwhm), _grid(frac), _grid(broad_fwhm)
    flat = broad_fwhm is None

    if atlas is None:
        margin = 2.0 + (0.0 if flat else 2.5 * float(np.nanmax(broads)))
        atlas = load_fts_atlas(min(w[0] for w in wavs) - margin,
                               max(w[-1] for w in wavs) + margin, atlas_file)
    awav, aint = np.asarray(atlas[0], float), np.asarray(atlas[1], float)

    setup = []
    for w in wavs:
        keep = np.ones(w.size, bool)
        for lam, half in mask:
            keep &= np.abs(w - lam) > half
        deg = poly_deg if poly_deg is not None else (3 if w[-1] - w[0] > 4.0 else 2)
        setup.append((keep, deg, w - w.mean(), float(np.median(np.diff(w)))))

    best = None
    for i, fw in enumerate(fwhms):
        cores = [_smeared_atlas(awav, aint, fw, s[3]) for s in setup]
        for bw in broads:
            wings = [None if flat else _smeared_atlas(awav, aint, bw, s[3]) for s in setup]
            for fr in fracs:
                total, found = 0.0, []
                for w, sp, (keep, deg, x, _), core, wing in zip(wavs, specs, setup, cores, wings):
                    win = None
                    for sh in shifts:
                        mod = (1.0 - fr) * np.interp(w + sh, awav, core)
                        mod = mod + (fr if flat else fr * np.interp(w + sh, awav, wing))
                        poly = np.polyfit(x[keep], (mod / sp)[keep], deg)
                        ms = float(np.mean((sp * np.polyval(poly, x) - mod)[keep] ** 2))
                        if win is None or ms < win[0]:
                            win = (ms, sh, poly, mod)
                    total += win[0]
                    found.append(win)
                if best is None or total < best[0]:
                    best = (total, fw, fr, bw, found)
        if verbose and fwhms.size > 1:
            print("fit_instrumental_profile: core %.3f A (%d/%d), best so far %.3f A"
                  % (fw, i + 1, fwhms.size, best[1]), flush=True)

    total, fw, fr, bw, found = best
    out = dict(fwhm=float(fw), frac=float(fr),
               broad_fwhm=None if flat else float(bw), chi2=float(total), windows=[])
    for w, sp, (keep, deg, x, _), (ms, sh, poly, mod) in zip(wavs, specs, setup, found):
        trans = np.polyval(poly, x)
        out["windows"].append(dict(shift=float(sh), poly=poly, wmean=float(w.mean()),
                                   transmission=trans, profile=sp * trans, model=mod,
                                   rms=float(np.sqrt(ms)), keep=keep))
    if single:
        out.update(out["windows"][0])
    return out


def stray_light_veil(wav, dc_profile, frac: float = DEFAULT_FRAC,
                     broad_fwhm: Optional[float] = DEFAULT_BROAD_FWHM) -> np.ndarray:
    """The stray light, ``frac * dc_profile (x) G(broad_fwhm)``.

    ``dc_profile`` is the observed disk-center profile of one window on the
    regular grid ``wav``. The result has its units. The window is padded with
    its edge values for the convolution. ``broad_fwhm=None`` gives the
    constant level ``frac`` (for a profile normalized to the continuum).
    """
    dc_profile = np.asarray(dc_profile, float)
    if broad_fwhm is None:
        return np.full(dc_profile.shape, float(frac))
    step = float(np.median(np.diff(np.asarray(wav, float))))
    return frac * gaussian_filter1d(dc_profile, broad_fwhm * _FWHM_TO_SIGMA / step,
                                    mode="nearest", truncate=4.0)


def remove_stray_light(cube, wav, dc_profile, frac: float = DEFAULT_FRAC,
                       broad_fwhm: Optional[float] = DEFAULT_BROAD_FWHM) -> np.ndarray:
    """Remove the stray light from a cube, the same amount in every pixel.

    ``(cube - frac * dc_profile (x) G(broad_fwhm)) / (1 - frac)``

    Parameters
    ----------
    cube : array (nwav, ...)
        Observed intensities of one window, wavelength first.
    wav : array (nwav,)
        Its wavelengths (regular grid, A).
    dc_profile : array (nwav,)
        Observed mean profile at disk center, same wavelengths and SAME UNITS
        as ``cube`` (both multiplied by the transmission of
        :func:`fit_instrumental_profile`, or both calibrated with the same factor).
    frac, broad_fwhm : float
        As fitted or adopted in :func:`fit_instrumental_profile`.

    Notes
    -----
    At disk center this turns the observed profile into the one the fit
    describes with the narrow core alone. Do not use it on off-limb pixels.
    """
    cube = np.asarray(cube, float)
    veil = stray_light_veil(wav, dc_profile, frac, broad_fwhm)
    return (cube - veil.reshape((-1,) + (1,) * (cube.ndim - 1))) / (1.0 - frac)
