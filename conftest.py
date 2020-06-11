import os

import numpy as np
import pytest
from astropy.io import fits
from hypothesis import Verbosity, settings
from pkg_resources import resource_filename

from spectrum_overload import Spectrum

settings.register_profile("ci", settings(max_examples=1000))
settings.register_profile("rpi", settings(max_examples=2))
settings.register_profile("dev", settings(max_examples=10))
settings.register_profile(
    "debug", settings(max_examples=10, verbosity=Verbosity.verbose)
)
settings.load_profile(os.getenv(u"HYPOTHESIS_PROFILE", "default"))


@pytest.fixture
def phoenix_spectrum():
    # Get a phoenix spectrum in test
    spec_1 = resource_filename("spectrum_overload", "data/spec_1.fits")
    flux = fits.getdata(spec_1)
    wave = np.arange(len(flux))
    header = fits.getheader(spec_1)
    return Spectrum(xaxis=wave, flux=flux, header=header)


@pytest.fixture
def ones_spectrum():
    x = np.linspace(2000, 2200, 1000)
    y = np.ones_like(x)
    return Spectrum(xaxis=x, flux=y)
