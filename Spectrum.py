#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
# Spectrum Class

# Begun August 2016
# Jason Neal


class Spectrum:
    """ Spectrum class represents and manipulates astronomical spectra. """
    
    def __init__(self, flux=[], xaxis=[], calibrated=False):
        """ Create a empty spectra """
        self.xaxis = xaxis
        self.flux = flux
        self.calibrated = calibrated

    def wav_select(self, wav_min, wav_max):
        """ Fast Wavelength selector between wav_min and wav_max values 
        If passed lists it will return lists.
        If passed np arrays it will return arrays
    
        """
        if isinstance(self.xaxis, list): # If passed lists
            wav_sel = [wav_val for wav_val in self.xaxis if (wav_min < wav_val < wav_max)]
            flux_sel = [flux_val for wav_val, flux_val in zip(self.xaxis, self.flux) if (wav_min < wav_val < wav_max)]
        elif isinstance(self.xaxis, np.ndarray):
            # Super Fast masking with numpy
            mask = (self.xaxis > wav_min) & (self.xaxis < wav_max)
            wav_sel = self.xaxis[mask]
            flux_sel = self.flux[mask]
        
        else:
              raise TypeError("Unsupported input wav type of type ", type(self.xaxis))
        # Set new spectra
        self.xaxis = wav_sel
        self.flux = flux_sel

    def doppler_shift(self, RV):
        ''' Function to compute a wavelenght shift due to radial velocity
        using RV / c = delta_lambda/lambda
        RV - radial velocity (in km/s)
        lambda_rest - rest wavelenght of the spectral line
        delta_lambda - (lambda_final - lambda_rest)
        '''
        if abs(RV) < 1e-7:
            """ RV smaller then 0.1 mm/s"""
            print("Warning the RV value given is very small (<0.1 mm/s).\n " 
                "Not performing the doppler shift")
        elif self.calibrated:
            c = 299792.458
            lambdaShift = self.xaxis * (RV / c)
            self.xaxis = self.xaxis + lambdaShift
        else:
            print("Attribute xaxis is not wavelength calibrated. Cannot perform doppler shift")


    def calibrate_with(self, wl_map):
        """ Calibrate with polynomial with parameters wl_map.
        Input:
            wl_map - Polynomial cooeficients that take the form expected by np.poylval()
        Output:
            self.xaxis is replaced with the calibrated spectrum
            self.calibrated is set to True
        The parameters can be generated by np.polyfit(x, y, order)
        """
        if self.calibrated:
            print("Spectrum already calibrated, Not Calibrating again.")
        else:
            wavelength = np.polyval(wl_map, self.xaxis)   # Polynomail parameters
            self.xaxis = wavelength
            self.calibrated = True  # Set calibrated Flag 
        
        if np.any(self.xaxis <= 0):
            print("Warning! The wavlength solution contains a value of zero. "
                  "Please check your calibrations\nThis will not doppler "
                  "shift correctly. This may raise an error in the future.")


    def interpolate_to(self, spectrum):
        """Interpolate wavelength solution to wavelength of spectrum
        Think about weather this should be spectrum or sepctrum.xaxis (just the wavelength) 
        """
        pass



## TO DO !
#--------------------
# Overrideopperators such 
# e.g, plus, minus, subtract, divide

# Interpolation in wavelength (before subtraction)
