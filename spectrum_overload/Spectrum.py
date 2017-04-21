#!/usr/bin/python

from __future__ import print_function, division
import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from PyAstronomy import pyasl
"""Spectrum Class."""

# Begun August 2016
# Jason Neal


class Spectrum(object):
    """Spectrum class to represent and manipulate astronomical spectra.

    Attributes
    ----------
    xaxis (np.ndarray)
        The wavlength or pixel position values.
    flux (np.ndarray, array-like, list
        The extracted flux (measured intesity of light).
    calibrated : bool
        Flag to indicate calibration state. (Default = True.)
    header : astropy.Header, dict-like
        Header information of observation.

    Methods
    -------
    wav_select:
    calibrate_with:
    doopler_shift:
    length_check:
    interpolate1d_to:
    spline_interpolate_to:

    Notes
    -----
    pass

    """

    def __init__(self, flux=None, xaxis=None, calibrated=True, header=None):
        """Initalise a Spectrum object."""
        # Some checks before creating class
        if isinstance(flux, str):
            raise TypeError("Cannot assign {} to the flux attribute".format(
                type(flux)))
        elif isinstance(xaxis, str):
            raise TypeError("Cannot assign {} to the xaxis attribute".format(
                type(xaxis)))

        if flux is not None:
            self._flux = np.asarray(flux)
        else:
            self._flux = flux

        if xaxis is None:
            if flux is None:
                self._xaxis = None
            else:
                # Applying range to xaxis of equal length of flux
                try:
                    self._xaxis = np.arange(len(flux))
                except TypeError:
                    print("TypeError caught becasue flux has no length")
                    self._xaxis = None
        else:
            self._xaxis = np.asarray(xaxis)  # Setter not used - need asarray

        # Check assigned lenghts
        self.length_check()
        self.calibrated = calibrated
        self.header = header   # Access header with a dictionary call.

    @property
    def xaxis(self):
        """Getter for the xaxis attribute."""
        # print("Getting xaxis property")
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value):
        """Setter for xaxis attribute.

        Turns value intp a numpy array if it is not.

        Parameters
        ----------
        value : array-like or list or None
            The value to set the spectrum's xaxis attribute. If None is given
            and the flux is not None then xaxis is turned into a array
            representing pixel position using np.arange(len(flux)).

        """
        if isinstance(value, str):
            # Try to catch some bad assignments
            # Yes a list of strings will not be caught
            raise TypeError("Cannot assign {} to the xaxis attribute".format(
                type(value)))
        elif value is None:
            try:
                # Try to assign arange the length of flux
                self._xaxis = np.arange(len(self._flux))
            except TypeError:
                # if self._flux is None then it has no length.
                self._xaxis = None
            # print("assigning xaxis the same length of _flux")

        # Add any other checks in here if necessary
        elif self._flux is not None:
            if len(value) != len(self._flux):
                raise ValueError("Lenght of xaxis does not match flux length")
            else:
                self._xaxis = np.asarray(value)
        else:
            # If flux is None
            self._xaxis = value

    @property
    def flux(self):
        """Getter for the flux attribute."""
        return self._flux

    @flux.setter
    def flux(self, value):
        """Setter for the flux attribute.

        Preforms type checking at turn into a numpy array if it is not.

        Parameters
        ----------
        value: array-like or list
            The value to set the spectrum's flux attribute

        """
        if isinstance(value, str):
            # Try to catch some bad assignments
            # Yes a list of strings will not be caught
            raise TypeError("Cannot assign {} to the flux attribute".format(
                type(value)))

        if value is not None:
            # print("Turning flux input into np array")
            # Not checking to make sure it equals the xaxis
            # If changing flux and xaxis set the flux first
            self._flux = np.asarray(value)
        else:
            self._flux = value

    def length_check(self):
        """Check length of xaxis and flux are equal.

        If everyting is ok then there is no response/output.

        Raises
        ------
        ValueError:
            The length of xaxis and flux must be the same.

        """
        if (self._flux is None) and (self._xaxis is None):
            # Can't measure lenght of none
            pass
        elif (self._flux is None) or (self._xaxis is None):
            pass
        elif len(self._flux) != len(self._xaxis):
            raise ValueError("The length of xaxis and flux must be the same")

    def wav_select(self, wav_min, wav_max):
        """Select part of the spectrum between the given wavelength bounds.

        Parameters
        ----------
        wav_min : float
            Lower wavelength bound
        wav_max : float
            Upper wavelength bound

        Returns
        -------
        None:
            Acts on self

        Notes
        -----
        This might be better suited to return the new spectra
        instead of direct replacement.

        """
        x_org = self.xaxis
        flux_org = self.flux
        try:
            if len(self.xaxis) == 0:
                print("Warning! Spectrum has an empty xaxis to select"
                      " wavelengths from")
            else:
                mask = (self.xaxis > wav_min) & (self.xaxis < wav_max)
                self.flux = self.flux[mask]    # change flux first
                self.xaxis = self.xaxis[mask]
        except TypeError:
            print("Spectrum has no xaxis to select wavelength from")
            # Return to original values iscase were changed
            self.flux = flux_org           # Fix flux first
            self.xaxis = x_org
            raise

    def add_noise(self, snr):
        """Add noise level of snr to the flux of the spectrum."""
        sigma = self.flux / snr
        # Add normal distributed noise at the SNR level.
        self.flux += np.random.normal(0, sigma)

    def doppler_shift(self, RV):
        """Doopler shift wavelength by a given Radial Velocity.

        Apply doopler shift to the wavlength values of the spectrum
        using the radial velocity value provided and the relation
        RV/c = :math:`\Delta\lambda/\lambda`.

        Parameters
        ----------
        RV : float
            Radial Velocity to doopler shift by in km/s.

        Warnings
        --------
        Small RV :
            A lower limit of RV shift of 0.1 mm/s is set to prevent RV
            shifts much smaller than wavelength accuracy.
        Uncalibrated xaxis :
            When the xaxis is uncalibrated there is no wavelength to
            doopler shift. A message is printed and no shift is done.

        Notes
        -----
        The Doopler shift is calculated using the relation

        .. math::
            RV / c = \Delta\lambda / \lambda

        Where RV is the radial velocity (in km/s), :math:`\lambda_0`
        is the rest wavelength and :math:`\Delta\lambda` is the wavelength
        shift, :math:`(\lambda_{shift} - \lambda_0)`

        """
        if RV == 0:
            """Do nothing."""
            pass
        if abs(RV) < 1e-7:  # RV smaller then 0.1 mm/s
            print("Warning the RV value given is very small (<0.1 mm/s).\n "
                  "Not performing the doppler shift")

        elif np.isnan(RV) or np.isinf(RV):
            print("Warning RV is infinity or Nan."
                  "Not performing the doppler shift")

        elif self.calibrated:
            c = 299792.458
            lambda_shift = self.xaxis * (RV / c)
            self.xaxis = self.xaxis + lambda_shift
        else:
            print("Attribute xaxis is not wavelength calibrated."
                  " Cannot perform doppler shift")

    def crosscorrRV(self, spectrum, rvmin, rvmax, drv, **params):
        """Perform pyasl.crosscorrRV with another spectrum.

        Parameters
        spectrum: Spectrum
            Spectrum object to cross correlate with.
        rvmin: float
            Minimum radial velocity for which to calculate the cross-correlation
            function [km/s].
        rvmax: float
            Maximum radial velocity for which to calculate the cross-correlation
            function [km/s].
        drv: float
            The width of the radial-velocity steps to be applied in the calculation
            of the cross-correlation function [km/s].
        kwargs: dict
            Cross-correlation parameters.

        Returns
        dRV: array
            The RV axis of the cross-correlation function. The radial velocity refer
            to a shift of the template, i.e., positive values indicate that the
            template has been red-shifted and negative numbers indicate a blue-shift
            of the template. The numbers are given in km/s.
        CC: array
            The cross-correlation function.

        Notes
        Uses the PyAstronomy function pyasl.crosscorrRV
        http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyaslDoc/aslDoc/crosscorr.html
        """
        drv, cc = pyasl.crosscorrRV(self.xaxis, self.flux, spectrum.xaxis, spectrum.flux,
                                    rvmin, rvmax, drv, **params)
        return drv, cc

    def calibrate_with(self, wl_map):
        """Calibrate with a wavelength mapping polynomial.

        Parameters
        ----------
        wl_map :
            Polynomial cooeficients of the form expected by np.poylval().
            [p0, p1, p2 ...]

        Returns
        -------
        None :
            Replaces xaxis of self. Also self.calibrated is set to True.

        Notes
        -----
        The parameters can be generated by np.polyfit(x, y, order)

        """
        if self.calibrated:
            # To bypass this could set Spectrum.calibrated = False
            raise SpectrumError("Spectrum is already calibrated"
                                ", Not recalibrating.")
        else:
            wavelength = np.polyval(wl_map, self.xaxis)   # Polynomial params
            if np.any(wavelength <= 0):
                raise SpectrumError("Wavelength solution contains zero or "
                                    "negative values. But wavelength must "
                                    "be positive. This solution will not "
                                    "doppler shift correctly. Please check "
                                    "your calibrations")
            else:
                self.xaxis = wavelength
                self.calibrated = True  # Set calibrated Flag

    def interpolate1d_to(self, reference, kind="linear", bounds_error=False,
                         fill_value=np.nan):
        u"""Interpolate wavelength solution to the reference wavelength.

        This uses the scipy's interp1d interpolation. The optional
        parameters are passed to scipy's interp1d. This interpolates
        self to the given reference xaxis values. It overwrites the
        xaxis and flux of self with the new values.

        Parameters
        ----------
        reference : Spectrum or numpy.ndarray
            The reference xaxis values to interpolate to.
        kind : (str or int, optional)
            Specifies the kind of interpolation as a string (‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’ where
            ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is ‘linear’.
        bounds_error : bool, optional
            If True, a ValueError is raised any time interpolation is
            attempted on a value outside of the range of x
            (where extrapolation is necessary). If False, out
            of bounds values are assigned fill_value. By default, an error
            is raised unless fill_value=”extrapolate”.
        fill_value : array-like or “extrapolate”, optional (default = NaN)
            If a ndarray (or float), this value will be used to fill in for
            requested points outside of the data range. If not provided, then
            the default is NaN. The array-like must broadcast properly to the
            dimensions of the non-interpolation axes.
            If a two-element tuple, then the first element is used as a
            fill value for x_new < x[0] and the second element is used for
            x_new > x[-1]. Anything that is not a 2-element tuple (e.g.,
            list or ndarray, regardless of shape) is taken to be a single
            array-like argument meant to be used for both bounds as below,
            above = fill_value, fill_value.
            If “extrapolate”, then points outside the data range will be
            extrapolated. (“nearest” and “linear” kinds only.)

        Raises
        ------
        TypeError:
            Cannot interpolate with the given object of type <type>.

        References
        ----------
        scipy.interolate.interp1d
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d

        """
        if kind == 'cubic':
            print("Warning! Cubic spline interpolation with interp1d can cause"
                  " memory errors and crashes")
        # Create scipy interpolation function from self
        interp_function = interp1d(self.xaxis, self.flux, kind=kind,
                                   fill_value=fill_value,
                                   bounds_error=bounds_error)

        # Determine the flux at the new locations given by reference
        if isinstance(reference, Spectrum):      # Spectrum type
            new_flux = interp_function(reference.xaxis)
            self.flux = new_flux                 # Flux needs to change first
            self.xaxis = reference.xaxis
        elif isinstance(reference, np.ndarray):  # Numpy type
            new_flux = interp_function(reference)
            self.flux = new_flux                 # Flux needs to change first
            self.xaxis = reference
        else:
            # print("Interpolate was not give a valid type")
            raise TypeError("Cannot interpolate with the given object of type"
                            " {}".format(type(reference)))

    def spline_interpolate_to(self, reference, w=None, bbox=[None, None], k=3,
                              ext=0, check_finite=False, bounds_error=False):
        """Interpolate wavelength solution using scipy's
                InterpolatedUnivariateSpline.

        The optional parameters are for scipy's InterpolatedUnivariateSpline
        function.

        Documentation copied from Sicpy:

        One-dimensional interpolating spline for a given set of data points.

        Fits a spline y = spl(x) of degree k to the provided x, y data. Spline
        function passes through all provided points. Equivalent to
        UnivariateSpline with s=0.

        Parameters
        ----------
        x : (N,) array_like
            Input dimension of data points this much be must be in
            ascending order.
        y : (N,) array_like
            Input dimension of data points
        w : (N,) array_like, optional
            Weights for spline fitting.
            Must be positive. If None (default), weights are all equal.
        bbox : (2,) array_like, optional
            2-sequence specifying the boundary of the approximation
            interval. If None (default), bbox=[x[0], x[-1]].
        k : int, optional
            Degree of the smoothing spline. Must be 1 <= k <= 5.
        ext : int or str, optional
            Controls the extrapolation mode for elements not in the
            interval defined by the knot sequence.
                if ext=0 or ‘extrapolate’, return the extrapolated value.
                if ext=1 or ‘zeros’, return 0
                if ext=2 or ‘raise’, raise a ValueError
                if ext=3 of ‘const’, return the boundary value.
            Default value is 0.
        check_finite : bool, optional
            Whether to check that the input arrays contain only
            finite numbers. Disabling may give a performance gain,
            but may result in problems (crashes, non-termination
            or non-sensical results) if the inputs do contain
            infinities or NaNs.
            Default is False.

        Raises:
        -------
        TypeError:
            Cannot interpolate with the given object of type
        ValueError:
            A value in reference is outside the interpolation range."

        See also
        --------
            https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html#scipy.interpolate.InterpolatedUnivariateSpline

        """
        # Create scipy interpolation function from self
        interp_spline = InterpolatedUnivariateSpline(self.xaxis, self.flux,
                                                     w=w, bbox=bbox, k=k,
                                                     ext=ext,
                                                     check_finite=False)

        # interp_function = interp1d(self.xaxis, self.flux, kind=kind,
        #                           fill_value=fill_value,
        #                           bounds_error=bounds_error)

        # Determine the flux at the new locations given by reference
        if isinstance(reference, Spectrum):      # Spectrum type
            new_flux = interp_spline(reference.xaxis)
            self_mask = ((reference.xaxis < np.min(self.xaxis)) |
                         (reference.xaxis > np.max(self.xaxis)))
            if np.any(self_mask) & bounds_error:
                raise ValueError("A value in reference.xaxis is outside"
                                 "the interpolation range.")
            new_flux[self_mask] = np.nan
            self.flux = new_flux                 # Flux needs to change first
            self.xaxis = reference.xaxis
        elif isinstance(reference, np.ndarray):  # Numpy type
            new_flux = interp_spline(reference)
            self_mask = ((reference < np.min(self.xaxis)) |
                         (reference > np.max(self.xaxis)))
            if np.any(self_mask) & bounds_error:
                raise ValueError("A value in reference is outside the"
                                 "interpolation range.")
            new_flux[self_mask] = np.nan
            self.flux = new_flux                 # Flux needs to change first
            self.xaxis = reference
        else:
            # print("Interpolate was not give a valid type")
            raise TypeError("Cannot interpolate with the given object of type"
                            " {}".format(type(reference)))

    # ######################################################
    # Overloading Operators
    # ######################################################

    def __add__(self, other):
        """Overloaded addition method for Spectrum.

        If there is addition between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a + b) = 10 and len(b + a) = 15.

        This makes a + b != b + a

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        new_flux = self.flux + prepared_other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __radd__(self, other):
        """Right addition."""
        # E.g. for first Item in Sum  0  + Spectrum fails.
        new_flux = self.flux + other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __sub__(self, other):
        """Overloaded subtraction method for Spectrum.

        If there is subtraction between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum.

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a - b) = 10 and len(b - a) = 15.

        # This makes a - b != -b + a

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        new_flux = self.flux - prepared_other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __mul__(self, other):
        """Overloaded multiplication method for Spectrum.

        If there is multiplication between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum.

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a * b) = 10 and len(b * a) = 15.

        This makes a * b != b * a

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        new_flux = self.flux * prepared_other
        return Spectrum(flux=new_flux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __truediv__(self, other):
        """Overloaded truedivision (/) method for Spectrum.

        If there is truedivision between two Spectrum objects which have
        difference xaxis values then the second Spectrum is interpolated
        to the xaxis of the first Spectum.

        e.g. if len(a.xaxis) = 10 and len(b.xaxis = 15)
        then if len(a / b) = 10 and len(b / a) = 15.

        This makes (a / b) != (1/b) / (1/a).

        """
        # Checks for type errors and size. It interpolates other if needed.
        prepared_other = self._prepare_other(other)
        # Divide by zero only gives a runtime warning with numpy
        new_flux = self.flux / prepared_other
        # May want to change the inf to something else, nan, 0?...
        # new_flux[new_flux == np.inf] = np.nan
        return Spectrum(flux=new_flux, xaxis=self.xaxis,
                        calibrated=self.calibrated)

    def __pow__(self, other):
        """Exponetial magic method."""
        # Overlaod to use power to scale the flux of the spectra
        # if len(other) > 1 :
        #    raise ValueError("Spectrum can only be raised to the power of
        # one number not {}".format(len(other)))
        if isinstance(other, Spectrum):
            raise TypeError("Can not preform Spectrum ** Spectrum")
        elif isinstance(other, (int, float, np.ndarray)):
            try:
                new_flux = self.flux ** other
                return Spectrum(flux=new_flux, xaxis=self.xaxis,
                                header=self.header, calibrated=self.calibrated)
            except:
                # Type error or value error are likely
                raise
        else:
            raise TypeError("Unexpected type {} given for"
                            " __pow__".format(type(other)))

    def __len__(self):
        """Return length of flux Spectrum."""
        return len(self.flux)

    def __neg__(self):
        """Take negative flux."""
        negflux = -self.flux
        return Spectrum(flux=negflux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __pos__(self):
        """Take positive flux."""
        posflux = +self.flux
        return Spectrum(flux=posflux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def __abs__(self):
        """Take absolute flux."""
        absflux = abs(self.flux)
        return Spectrum(flux=absflux, xaxis=self.xaxis, header=self.header,
                        calibrated=self.calibrated)

    def _prepare_other(self, other):
        if isinstance(other, Spectrum):
            if self.calibrated != other.calibrated:
                # Checking the Spectra are of same calibration state
                raise SpectrumError("Spectra are not calibrated similarly.")
            if np.all(self.xaxis == other.xaxis):  # Only for equal xaxis
                # Easiest condition in which xaxis of both are the same
                return copy.copy(other.flux)
            else:  # Uneven length xaxis need to be interpolated
                no_overlap_lower = (np.min(self.xaxis) > np.max(other.xaxis))
                no_overlap_upper = (np.max(self.xaxis) < np.min(other.xaxis))
                if no_overlap_lower | no_overlap_upper:
                    raise ValueError("The xaxis do not overlap so cannot"
                                     " be interpolated")
                else:
                    other_copy = copy.copy(other)
                    # other_copy.interpolate_to(self)
                    other_copy.spline_interpolate_to(self)
                    return other_copy.flux
        elif isinstance(other, (int, float, np.ndarray)):
            return copy.copy(other)
        else:
            raise TypeError("Unexpected type {} given".format(type(other)))


class SpectrumError(Exception):
    """An errorclass for specturm errors."""
    pass
