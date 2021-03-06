#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test Suite for Spectrum Class.

It is not perfect and can be definitely improved.
"""
from __future__ import division, print_function

import hypothesis.strategies as st
import numpy as np
import pytest
from astropy.io import fits
from hypothesis import example, given
from pkg_resources import resource_filename
from PyAstronomy import pyasl

from spectrum_overload import Spectrum, SpectrumError


@given(st.lists(st.floats(min_value=-1e5, max_value=1e5)), st.integers(), st.booleans())
def test_spectrum_assigns_hypothesis_data(y, x, z):
    """Test that data was assigned to the correct attributes."""
    # Use one hypothesis list they need to have the same length
    # multiply by a random int to mix it up a little.
    x = x * np.array(y)
    spec = Spectrum(flux=y, xaxis=x, calibrated=z)
    assert np.all(spec.flux == y)
    assert np.all(spec.xaxis == x)
    assert spec.calibrated == z


def test_spectrum_assigns_data():
    """Test a manual example.

    Lengths of x and y need to be the same.
    """
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 1, 0.9, 0.95, 1, 1]
    calib_val = 0

    spec = Spectrum(flux=y, xaxis=x, calibrated=calib_val)
    assert np.all(spec.flux == y)
    assert np.all(spec.xaxis == x)
    assert spec.calibrated == calib_val


def test_empty_call_is_nones():
    # Check empty Spectrum is implemented with Nones
    # and not Nones in an array (!=None)
    s = Spectrum()
    assert s.flux is None
    assert s.xaxis is None
    assert s.header == {}  # empty header dict
    assert s.calibrated is True

    s2 = Spectrum(calibrated=False)
    assert s2.calibrated is False


@pytest.mark.xfail
def test_setters_for_flux_and_xaxis():
    s = Spectrum()
    # Try set flux to None
    s.flux = None
    # Try set xaxis when flux is None
    s.xaxis = [1, 2, 3, 4]

    s.flux = [2, 2.1, 2.2, 2.1]
    # Spectrum(False, False)
    # Spectrum(False, None)
    # Spectrum(False, False)
    # Spectrum(None, False)
    assert False


def test_length_checking():
    s = Spectrum(flux=None, xaxis=None)
    # Length check when both None
    s.length_check()
    # Try set xaxis when flux is None
    s.xaxis = [1, 2, 3, 4]
    # Run length check when flux is None and xaxis is not
    s.length_check()
    s.flux = [2, 2.1, 2.2, 2.1]

    # Create new empty spectrum
    s = Spectrum(flux=None, xaxis=None)
    s.flux = [2, 2.1, 2.2, 2.1]
    # Run length check when xaxis is None and flux is not
    print(s.xaxis)
    print(s.flux)
    s.length_check()

    with pytest.raises(ValueError):
        # Wrong length should fail
        _ = Spectrum(flux=[1, 4, 5], xaxis=[2, 1])


def test_flux_and_xaxis_cannot_pass_stings():
    """Passing a string to flux or xaxis will raise a TypeError."""
    with pytest.raises(TypeError):
        _ = Spectrum(flux=[1, 2, 3], xaxis="bar")
    with pytest.raises(TypeError):
        _ = Spectrum(flux="foo", xaxis=[1.2, 3, 4, 5])
    with pytest.raises(TypeError):
        _ = Spectrum(flux="foo", xaxis="bar")
    spec = Spectrum(flux=[1, 1, .5, 1])
    with pytest.raises(TypeError):
        spec.flux = "foo"
    with pytest.raises(TypeError):
        spec.xaxis = "bar"


def test_auto_generation_of_xaxis_if_none():
    spec = Spectrum(flux=[1, 1, .5, 1])
    assert np.all(spec.xaxis == np.arange(4))
    spec2 = Spectrum(flux=[1, 1, .5, 1], xaxis=[100, 110, 160, 200])
    spec2.xaxis = None  # reset xaxis
    assert np.all(spec2.xaxis == np.arange(4))


@pytest.mark.parametrize(
    "xaxis, flux",
    [([1, 2, 3], [1, 2]), ([1, 2, 3], []), ([], [1, 2, 3]), ([1, 2], [1, 2, 3])],
)
def test_length_of_flux_and_xaxis_must_be_equal(xaxis, flux):
    """Try assign a mismatched xaxis it should raise a ValueError."""
    with pytest.raises(ValueError):
        _ = Spectrum(flux=flux, xaxis=xaxis)


def test_reassigning_unequal_length_fails():
    spec = Spectrum(flux=[1, 2, 3], xaxis=[1, 2, 3])
    with pytest.raises(ValueError):
        spec.xaxis = [1, 2]


@given(
    st.lists(st.floats(min_value=-1e5, max_value=1e5)),
    st.booleans(),
    st.floats(min_value=-1e5, max_value=1e5),
    st.floats(min_value=-1e5, max_value=1e5),
)
def test_wav_select(x, calib, wav_min, wav_max):
    """Test some properties of wavelength selection."""
    # Create spectrum
    y = np.copy(x)
    spec = Spectrum(flux=y, xaxis=x, calibrated=calib)
    # Select wavelength values
    spec.wav_select(wav_min, wav_max)

    # All values in selected spectrum should be less than the max and greater
    # than the min value.
    assert all(spec.xaxis >= wav_min)
    assert all(spec.xaxis <= wav_max)


def test_wav_select_example():
    """Manual test of a wavelength selection."""
    # Create spectrum
    y = 2 * np.random.random(20)
    x = np.arange(20)
    calib = False
    spec = Spectrum(flux=y, xaxis=x, calibrated=calib)
    # Select wavelength values

    spec.wav_select(5, 11)
    # All values in selected spectrum should be less than the max and greater
    # than the min value.
    assert all(spec.xaxis >= 5)
    assert all(spec.xaxis <= 11)
    assert all(spec.xaxis == np.arange(6, 11))
    assert all(spec.flux == y[np.arange(6, 11)])
    # Also need to test assignment!
    # spec2 = spec.wav_selector()


@given(
    st.lists(st.floats(min_value=1e-4, max_value=1e5), min_size=1),
    st.floats(min_value=-1e5, max_value=1e5),
    st.sampled_from((1, 1, 1, 1, 1, 1, 1, 0)),
    st.booleans(),
)
@example([1000, 2002, 2003, 2004], 1e-8, 1, 1)
def test_doppler_shift_with_hypothesis(x, rv, calib, rv_dir):
    """Test doppler shift properties.

    Need to check values against PyAstronomy separately.
    Calib is sampled with a 1/8 chance being uncalibrated.

    """
    # Added a min value to rv shift to avoid very small rv values (1e-300).
    # Have added a flag to change rv direction to explore negative values
    rvdir = 2 * rv_dir - 1  # True -> 1 , False -> -1
    rv = rv * rvdir
    x = np.asarray(x)
    y = np.random.random(len(x))

    spec = Spectrum(flux=y, xaxis=x, calibrated=calib)
    # Apply Doppler shift of rv km/s.
    spec.doppler_shift(rv)

    if not spec.calibrated:
        assert np.allclose(spec.xaxis, x)
    else:
        tolerance = 1e-6
        if abs(rv) < tolerance:
            assert np.allclose(spec.xaxis, x)
        elif np.isnan(rv) or np.isinf(rv):
            assert np.all(spec.xaxis == x)
        elif rv < 0:
            assert np.all(spec.xaxis < x)
        elif rv > 0:
            assert np.all(spec.xaxis > x)


def test_x_calibration_works():
    """Simple test to check that the calibration works."""
    # Setup the code
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = [float(x_i) for x_i in x]
    y = np.ones_like(x)
    spec = Spectrum(flux=y, xaxis=x, calibrated=False)

    # Easy test
    params = np.polyfit([1, 5, 10], [3, 15, 30], 1)

    spec.calibrate_with(params)

    assert spec.calibrated
    assert np.allclose(spec.xaxis, np.asarray(x) * 3)


def test_cant_calibrate_calibrated_spectrum():
    """Check that a calibrated spectra is not calibrated a second time."""
    s = Spectrum(flux=[1, 2, 3, 4], xaxis=[1, 2, 3, 4], calibrated=True)

    with pytest.raises(SpectrumError):
        s.calibrate_with([5, 3, 2])
    assert np.all(s.xaxis == np.array([1, 2, 3, 4]))


def test_calibration_wavelength_only_positive():
    """Not quite sure what is happening here."""
    # Can't have a wavelength of zero or negative.
    # So raise a SpectrumError before calibrating
    s = Spectrum(flux=[1, 2, 3, 4], xaxis=[-4, -3, -2, -1], calibrated=False)
    with pytest.raises(SpectrumError):
        s.calibrate_with([0, 1, 0])  # y = 0*x**2 + 1*x + 0
    assert s.calibrated is False  # Check values stay the same
    assert np.all(s.flux == np.array([1, 2, 3, 4]))
    assert np.all(s.xaxis == np.array([-4, -3, -2, -1]))

    s = Spectrum(flux=[1, 2, 3, 4], xaxis=[0, 2, 3, 4], calibrated=False)
    with pytest.raises(SpectrumError):
        s.calibrate_with([0, 1, 0])  # y = 0*x**2 + 1*x + 0
    assert s.calibrated is False  # Check values stay the same
    assert np.all(s.flux == np.array([1, 2, 3, 4]))
    assert np.all(s.xaxis == np.array([0, 2, 3, 4]))


def test_header_attribute():
    """Test header attribute is accessible as a dict."""
    header = {"Date": "20120601", "Exptime": 180}
    spec = Spectrum(header=header)
    # Some simple assignment tests
    assert spec.header["Exptime"] == 180
    assert spec.header["Date"] == "20120601"

    # Try with a Astropy header object
    test_file = resource_filename("spectrum_overload", "data/spec_1.fits")
    fitshdr = fits.getheader(test_file)
    spec2 = Spectrum(header=fitshdr)

    assert spec2.header["OBJECT"] == fitshdr["OBJECT"]
    assert spec2.header["EXPTIME"] == fitshdr["EXPTIME"]

    assert Spectrum().header == {}  # unassigned header is empty dict


def test_interpolation():
    """Test interpolation."""
    # Test the interpolation function some how
    # simple examples?
    # simple linear case
    x1 = [1., 2., 3., 4., 5.]
    y1 = [2., 4., 6., 8., 10]
    x2 = [1.5, 2, 3.5, 4]
    y2 = [1., 2., 1., 2.]
    s1 = Spectrum(flux=y1, xaxis=x1)
    s2 = Spectrum(flux=y2, xaxis=x2)
    s_lin = s1.copy()
    s_lin.interpolate1d_to(s2, kind="cubic")

    assert np.allclose(s_lin.flux, [3., 4., 7., 8.])
    # test linear interpolation matches numpy interp
    assert np.allclose(s_lin.flux, np.interp(x2, x1, y1))

    s_same = s1.copy()
    # Interpolation to itself should be the same
    s_same.interpolate1d_to(s1)
    assert np.allclose(s_same.flux, s1.flux)
    assert np.allclose(s_same.xaxis, s1.xaxis)

    # Need to test that if bounds error is True a ValueError is raised
    with pytest.raises(ValueError):
        s2.interpolate1d_to(s1, bounds_error=True)
    with pytest.raises(ValueError):
        s2.interpolate1d_to(s1, kind="cubic", bounds_error=True)
    with pytest.raises(TypeError):
        s2.interpolate1d_to(x1, bounds_error=True)
    with pytest.raises(ValueError):
        s2.interpolate1d_to(np.asarray(x1), bounds_error=True)
    with pytest.raises(TypeError):
        s2.interpolate1d_to([1, 2, 3, 4])
    with pytest.raises(TypeError):
        s2.interpolate1d_to("string")
    # Need to write better tests!


def test_interpolation_when_given_a_ndarray():
    """Test interpolation."""
    x1 = [1., 2., 3., 4., 5.]
    y1 = [2., 4., 6., 8., 10]
    x2 = [1.5, 2, 3.5, 4]
    # y2 = [1., 2., 1., 2.]
    s1 = Spectrum(flux=y1, xaxis=x1)
    # s2 = Spectrum(flux=y2, xaxis=x2)
    s_lin = s1.copy()
    s_lin.interpolate1d_to(np.asarray(x2), kind="linear")

    assert np.allclose(s_lin.flux, [3., 4., 7., 8.])
    # test linear interpolation matches numpy interp
    assert np.allclose(s_lin.flux, np.interp(x2, x1, y1))


def test_spline_interpolation():
    """Test spline interpolation."""
    # Test the interpolation function some how
    # simple examples?
    # simple linear case
    x1 = [1., 2., 3., 4., 5.]
    y1 = [2., 4., 6., 8., 10]
    x2 = [1.5, 2, 3.5, 4]
    y2 = [1., 2., 1., 2.]
    s1 = Spectrum(flux=y1, xaxis=x1)
    s2 = Spectrum(flux=y2, xaxis=x2)
    s_lin = s1.copy()
    s_lin.spline_interpolate_to(s2, k=1)

    assert np.allclose(s_lin.flux, [3., 4., 7., 8.])
    # test linear interpolation matches numpy interp
    assert np.allclose(s_lin.flux, np.interp(x2, x1, y1))

    s_same = s1.copy()
    # Interpolation to itself should be the same
    s_same.spline_interpolate_to(s1)
    assert np.allclose(s_same.flux, s1.flux)
    assert np.allclose(s_same.xaxis, s1.xaxis)

    # Need to test that if bounds error is True a ValueError is raised
    with pytest.raises(ValueError):
        s2.spline_interpolate_to(s1, bounds_error=True)
    with pytest.raises(ValueError):
        s2.spline_interpolate_to(s1, k=1, bounds_error=True)
    with pytest.raises(TypeError):
        s2.spline_interpolate_to(x1, bounds_error=True)
    with pytest.raises(ValueError):
        s2.spline_interpolate_to(np.asarray(x1), bounds_error=True)
    with pytest.raises(TypeError):
        s2.spline_interpolate_to([1, 2, 3, 4])
    with pytest.raises(TypeError):
        s2.spline_interpolate_to("string")
    # Need to write better tests!
    # These are a direct copy of other interpolation test


def test_spline_interpolation_when_given_a_ndarray():
    """Test spline interpolation."""
    x1 = [1., 2., 3., 4., 5.]
    y1 = [2., 4., 6., 8., 10]
    x2 = [1.5, 2, 3.5, 4]
    # y2 = [1., 2., 1., 2.]
    s1 = Spectrum(flux=y1, xaxis=x1)
    # s2 = Spectrum(flux=y2, xaxis=x2)
    s_lin = s1.copy()
    s_lin.spline_interpolate_to(np.asarray(x2), k=1)

    assert np.allclose(s_lin.flux, [3., 4., 7., 8.])
    # test linear interpolation matches numpy interp
    assert np.allclose(s_lin.flux, np.interp(x2, x1, y1))


def test_interp_method():
    """Test interpolation method attribute."""
    s = Spectrum()
    assert s.interp_method == "spline"

    s.interp_method = "linear"
    assert s.interp_method == "linear"


def test_bad_interp_method():
    s = Spectrum()
    with pytest.raises(ValueError):
        s.interp_method = "invalid_method"


@pytest.mark.parametrize("snr", [50, 100])
def test_add_noise(ones_spectrum, snr):
    """Test addition of noise."""
    np.random.seed(3)
    ones_spectrum.add_noise(snr)

    assert np.isclose(np.std(ones_spectrum.flux), 1. / snr, atol=1e-3)


def test_remove_nans():
    s = Spectrum(xaxis=np.arange(5), flux=[3, 2, np.nan, 4, np.nan])
    assert len(s.xaxis) == 5 and len(s.flux) == 5

    s = s.remove_nans()

    assert np.all(s.xaxis == np.array([0, 1, 3]))
    assert np.all(s.flux == np.array([3, 2, 4]))
    assert len(s.xaxis) == 3 and len(s.flux) == 3


def test_scalar_normalization():
    """Test normalization is close to unity."""
    x = np.arange(50)
    y = 5 * np.ones_like(x)
    s = Spectrum(xaxis=x, flux=y)
    sn = s.normalize(method="scalar")

    assert np.allclose(sn.flux, np.ones_like(x))


def test_exponential_normalization():
    """Test normalization is close to unity."""
    x = (np.arange(10, 500) * 0.01) + 2
    y = np.exp(x)
    s = Spectrum(xaxis=x, flux=y)
    sn = s.normalize(method="exponential")
    expected = np.ones_like(x)
    assert np.allclose(sn.flux, np.ones_like(x))


@pytest.mark.parametrize(
    "method, degree", [("scalar", 0), ("linear", 1), ("quadratic", 2), ("cubic", 3)]
)
def test_normalization_method_match_degree(method, degree):
    # TODO: Eventually call phoenix_spectrum to do this.
    x = np.arange(1, 1000)
    y = np.arange(1, 1000)
    s = Spectrum(xaxis=x, flux=y)
    named_method = s.normalize(method=method)
    named_method = named_method.remove_nans()  # hack for getting to run on < py34
    poly_deg = s.normalize(method="poly", degree=degree)
    poly_deg = poly_deg.remove_nans()  # hack for getting to pass on < py 34
    assert np.allclose(named_method.flux, poly_deg.flux)


@pytest.mark.parametrize(
    "R", [5, 10, 99, 500]
)  # Small resolutions for easy testing (measureable differences)
def test_instrument_broaden(phoenix_spectrum, R):
    """Test instrument_broadening same as pyastronomy."""
    spec = phoenix_spectrum

    new_flux = pyasl.instrBroadGaussFast(spec.xaxis, spec.flux, R)

    # There is a change due to broadening
    assert not np.allclose(spec.flux, new_flux)
    new_spec = spec.instrument_broaden(R)

    # There is a change due to broadening
    assert not np.allclose(new_spec.flux, spec.flux)
    # Spectrum result equals correct pyasl value.
    assert np.allclose(new_spec.flux, new_flux)


@pytest.mark.parametrize(
    "item",
    [
        [5],
        [-60],
        [1, 2, 5, 6, 7],
        [-1, 2, 5, 20, -6],
        slice(10, 100),
        slice(0, 8),
        slice(None),
    ],
)
def test_spectrum_slicing(phoenix_spectrum, item):
    sliced_spectrum = phoenix_spectrum[item]

    assert np.all(sliced_spectrum.xaxis == phoenix_spectrum.xaxis[item])
    assert np.all(sliced_spectrum.flux == phoenix_spectrum.flux[item])
    assert sliced_spectrum.header == phoenix_spectrum.header
    assert sliced_spectrum.calibrated == phoenix_spectrum.calibrated


def test_spectrum_slicing_with_colon(phoenix_spectrum):
    sliced_spectrum = phoenix_spectrum[:]
    assert np.all(sliced_spectrum.xaxis == phoenix_spectrum.xaxis[:])
    assert np.all(sliced_spectrum.flux == phoenix_spectrum.flux[:])

    sliced_spectrum2 = phoenix_spectrum[:50]
    assert np.all(sliced_spectrum2.xaxis == phoenix_spectrum.xaxis[:50])
    assert np.all(sliced_spectrum2.flux == phoenix_spectrum.flux[:50])

    sliced_spectrum3 = phoenix_spectrum[-200:]
    assert np.all(sliced_spectrum3.xaxis == phoenix_spectrum.xaxis[-200:])
    assert np.all(sliced_spectrum3.flux == phoenix_spectrum.flux[-200:])

    sliced_spectrum4 = phoenix_spectrum[50:2:80]
    assert np.all(sliced_spectrum4.xaxis == phoenix_spectrum.xaxis[50:2:80])
    assert np.all(sliced_spectrum4.flux == phoenix_spectrum.flux[50:2:80])


@pytest.mark.parametrize("item", [None, "hello", "", 7, 3.14, True, False, 0, 1.0])
def test_specturm_slicing_invalid_types(phoenix_spectrum, item):
    """Invalid scalars and other types."""
    with pytest.raises(ValueError):
        _ = phoenix_spectrum[item]
