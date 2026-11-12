"""Instrumental profile and stray light (chase.calib.straylight).

Everything runs on a synthetic atlas, so neither ISPy nor real data is needed.
The synthetic "observation" is built with the same forward model the fit uses,
multiplied by a known transmission polynomial, so the fit must give back the
numbers that went in.
"""

import numpy as np
import pytest

from chase.calib import (
    fit_instrumental_profile,
    instrumental_profile_model,
    remove_stray_light,
    stray_light_veil,
)


def _atlas():
    """Continuum normalized atlas on a 0.0078 A grid: H-alpha-like and Fe I-like lines."""
    wav = np.arange(6550.0, 6580.0, 0.0078125)
    spec = np.ones_like(wav)
    for core, depth, width in ((6562.8, 0.84, 0.45), (6569.21, 0.65, 0.06),
                               (6560.58, 0.20, 0.05), (6564.20, 0.05, 0.03)):
        spec *= 1.0 - depth * np.exp(-0.5 * ((wav - core) / width) ** 2)
    return wav, spec


def _window(w0, w1, n):
    return np.linspace(w0, w1, n)


WAV_HA = _window(6559.4, 6565.1, 118)
WAV_FE = _window(6567.5, 6569.7, 46)


def _observe(wav, atlas, fwhm, frac, broad, shift, poly):
    """DN spectrum whose transmission-corrected profile is the model."""
    model = instrumental_profile_model(wav, *atlas, fwhm=fwhm, frac=frac,
                                       broad_fwhm=broad, shift=shift)
    trans = np.polyval(poly, wav - wav.mean())
    return model / trans


def test_constant_level_is_hou_model():
    """broad_fwhm=None is Hou et al. 2020 Eq. 1: (1 - a) atlas (x) G + a."""
    atlas = _atlas()
    core = instrumental_profile_model(WAV_FE, *atlas, fwhm=0.1, frac=0.0)
    flat = instrumental_profile_model(WAV_FE, *atlas, fwhm=0.1, frac=0.1, broad_fwhm=None)
    assert np.allclose(flat, 0.9 * core + 0.1)


def test_fit_recovers_one_profile_for_both_windows():
    atlas = _atlas()
    truth = dict(fwhm=0.105, frac=0.12, broad=2.4)
    dn_ha = _observe(WAV_HA, atlas, truth["fwhm"], truth["frac"], truth["broad"],
                     -0.004, [2e-6, -1e-5, 1e-3])
    dn_fe = _observe(WAV_FE, atlas, truth["fwhm"], truth["frac"], truth["broad"],
                     0.006, [5e-5, 2e-3])
    r = fit_instrumental_profile([WAV_HA, WAV_FE], [dn_ha, dn_fe], atlas=atlas, mask=[],
                                 fwhm=np.arange(0.09, 0.1201, 0.005),
                                 broad_fwhm=np.arange(2.0, 2.81, 0.2),
                                 frac=np.arange(0.10, 0.1401, 0.01),
                                 shifts=np.arange(-0.01, 0.0101, 0.002))
    assert r["fwhm"] == pytest.approx(truth["fwhm"], abs=1e-6)
    assert r["broad_fwhm"] == pytest.approx(truth["broad"], abs=1e-6)
    assert r["frac"] == pytest.approx(truth["frac"], abs=1e-6)
    assert [w["shift"] for w in r["windows"]] == pytest.approx([-0.004, 0.006], abs=1e-9)
    assert max(w["rms"] for w in r["windows"]) < 1e-4
    # the returned transmission puts the window back on the model
    for w in r["windows"]:
        assert np.allclose(w["profile"], w["model"], atol=1e-4)


def test_single_window_returns_flat_dict():
    atlas = _atlas()
    dn = _observe(WAV_FE, atlas, 0.1, 0.1, None, 0.0, [1e-3])
    r = fit_instrumental_profile(WAV_FE, dn, atlas=atlas, broad_fwhm=None, mask=[],
                                 fwhm=np.arange(0.08, 0.1201, 0.01),
                                 frac=np.arange(0.06, 0.1401, 0.02),
                                 shifts=[0.0])
    assert r["broad_fwhm"] is None
    assert r["fwhm"] == pytest.approx(0.1) and r["frac"] == pytest.approx(0.1)
    assert "rms" in r and len(r["windows"]) == 1


def test_correction_leaves_only_the_core_at_disk_center():
    """At disk center the corrected profile is the atlas seen through the core alone."""
    atlas = _atlas()
    dc = instrumental_profile_model(WAV_HA, *atlas, fwhm=0.1, frac=0.145, broad_fwhm=1.8)
    core = instrumental_profile_model(WAV_HA, *atlas, fwhm=0.1, frac=0.0)
    corrected = remove_stray_light(dc[:, None, None], WAV_HA, dc)[:, 0, 0]
    # the veil is built from the observed dc profile, not the atlas, so the match
    # holds to second order in frac
    assert np.max(np.abs(corrected - core)) < 0.01
    assert np.max(np.abs(corrected - core)) < 0.2 * np.max(np.abs(dc - core))


def test_same_veil_in_every_pixel():
    """Pixels that differ by a constant offset stay apart by offset / (1 - frac)."""
    atlas = _atlas()
    dc = instrumental_profile_model(WAV_FE, *atlas)
    cube = np.stack([dc, dc - 0.3, dc + 0.1], axis=1)
    out = remove_stray_light(cube, WAV_FE, dc, frac=0.145)
    assert np.allclose(out[:, 0] - out[:, 1], 0.3 / 0.855)
    assert np.allclose(out[:, 2] - out[:, 0], 0.1 / 0.855)


def test_veil_shapes_and_flat_option():
    atlas = _atlas()
    dc = instrumental_profile_model(WAV_HA, *atlas)
    veil = stray_light_veil(WAV_HA, dc, 0.145, 1.8)
    assert veil.shape == dc.shape
    # a smoothed spectrum: shallower than the profile, same mean to a few percent
    assert veil.min() / 0.145 > dc.min()
    assert abs(veil.mean() / 0.145 - dc.mean()) < 0.03
    assert np.allclose(stray_light_veil(WAV_HA, dc, 0.05, None), 0.05)
    cube4 = np.ones((WAV_HA.size, 2, 3, 4))
    assert remove_stray_light(cube4, WAV_HA, dc).shape == cube4.shape
