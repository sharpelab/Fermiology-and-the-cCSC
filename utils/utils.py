import json
import gzip
import csv
import os
from datetime import datetime, timezone

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, butter, sosfiltfilt
from itertools import product
from typing import Dict, List, Tuple
from lmfit.models import SplineModel

# Physical constants
h = 6.626E-34  # Planck constant (J·s)
e = 1.6002E-19  # Elementary charge (C)
phi0 = h / e    # Flux quantum (Wb)
Rq = h/e**2

# Device params (rmg19.json lives at the repo root, next to the figure folders)
_RMG19_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rmg19.json')
with open(_RMG19_JSON, 'r') as f:
    rmg19 = json.load(f)

dbg = rmg19['default']['dbg']
dtg = rmg19['default']['dtg']

# ==============================================================================
# Gate voltage and carrier density calculations
# ==============================================================================

def calculate_gate_voltages(n, D, dtg=None, dbg=None):
    # Calculate electron density n from nu
    # Use values from config if not provided
    if dtg is None:
        dtg = rmg19['default']['dtg']
    if dbg is None:
        dbg = rmg19['default']['dbg']
    
    ep = 3   
    conversion_factor = 5.52634936
    # Set up system of equations based on original expressions for n and D
    # Rearrange these to solve for Vbg and Vtg
    Vtg = (n / (ep * conversion_factor) + 2* D/ep) * dtg/2
    Vbg = dbg*(n / (ep * conversion_factor) - Vtg/dtg)

    return Vtg, Vbg


def calculate_n_D(Vtg, Vbg, dtg=None, dbg=None):
    # Use values from config if not provided
    if dtg is None:
        dtg = rmg19['default']['dtg']
    if dbg is None:
        dbg = rmg19['default']['dbg']
    
    ep = 3
    n = (Vbg/dbg + Vtg/dtg)*ep*5.52634936
    D = 0.5*ep*(Vtg/dtg - Vbg/dbg)
    return n, D

# ==============================================================================
# Various Physics Functions
# ==============================================================================

def streda(C, Bs, offset=0):
    return C*Bs/phi0/1E16+offset


# ==============================================================================
# Combinatorics
# ==============================================================================

def linear_combos_close_to_one(
    vals: List[float],
    bound: int,
    window: float,
) -> Dict[Tuple[int, ...], Tuple[float, float]]:
    """
    Returns:
        {
          (c1, c2, ..., cn): (total_sum, abs(total_sum - 1))
        }
        for all |sum - 1| <= window
    """
    n = len(vals)
    r = range(-bound, bound + 1)

    solutions: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    for coeffs in product(r, repeat=n):
        total = sum(c * v for c, v in zip(coeffs, vals))
        err = abs(total - 1.0)

        if err <= window:
            solutions[coeffs] = (total, err)

    return solutions

def linear_combos_close_to_one_restrained(
    vals: List[float],
    bound: int,
    window: float,
) -> Dict[Tuple[int, ...], Tuple[float, float]]:
    """
    Returns:
        {
          (c1, c2, ..., cn): (total_sum, abs(total_sum - 1))
        }
        for all |sum - 1| <= window
    """
    n = len(vals)
    r = range(-bound, bound + 1)

    solutions: Dict[Tuple[int, ...], Tuple[float, float]] = {}

    for coeffs in product(r, repeat=n):
        total = sum(c * v for c, v in zip(coeffs, vals))
        err = abs(total - 1.0)

        if err <= window:
            solutions[coeffs] = (total, err)

    return solutions

def linear_combos_specific_integers(
    vals: List[float],
    allowed_integers: List[int],
    scaling: float = 0.03,
) -> Dict[Tuple[int, ...], Tuple[float, float, float]]:
    """
    Search for linear combinations of vals that equal 1, using only specific
    allowed integers as coefficients, and allowing for scaling of all input values.
    
    Parameters
    ----------
    vals : List[float]
        Input values to combine
    allowed_integers : List[int]
        Specific integers allowed as coefficients (e.g., [-2, -1, 0, 1, 2])
    scaling : float, optional
        Allowed scaling factor for all values (default 0.03 for 3%)
        
    Returns
    -------
    solutions : Dict[Tuple[int, ...], Tuple[float, float, float]]
        {
          (c1, c2, ..., cn): (scaled_sum, scale_factor, error)
        }
        where scale_factor is the scaling applied to all vals
        and error is the absolute difference from 1.0
    """
    n = len(vals)
    solutions: Dict[Tuple[int, ...], Tuple[float, float, float]] = {}
    
    for coeffs in product(allowed_integers, repeat=n):
        # Skip all-zero coefficients
        if all(c == 0 for c in coeffs):
            continue
            
        base_sum = sum(c * v for c, v in zip(coeffs, vals))
        
        if abs(base_sum) < 1e-10:
            continue
            
        # Calculate required scale factor to reach 1.0
        scale_factor = 1.0 / base_sum
        
        # Check if scale factor is within allowed range
        if abs(scale_factor - 1.0) <= scaling:
            scaled_sum = base_sum * scale_factor
            error = abs(scaled_sum - 1.0)
            solutions[coeffs] = (scaled_sum, scale_factor, error)
    
    return solutions



# ==============================================================================
# Data interpolation and FFT analysis
# ==============================================================================

def interp_rxx(B, Rxx, nx=400, type="cubic"):
    """
    Interpolate Rxx onto evenly-spaced 1/B grid.
    
    Parameters
    ----------
    B : array
        Magnetic field values (T)
    Rxx : array
        Resistance values (Ω)
    nx : int
        Number of interpolation points
    
    Returns
    -------
    invBvec : array
        Evenly-spaced 1/B values (1/T)
    Rxx_interp : array
        Interpolated resistance values
    """
    invB = 1.0 / B
    order = np.argsort(invB)
    invB_sorted = invB[order]
    Rxx_sorted = Rxx[order, ...]
    
    invBvec = np.linspace(invB_sorted[0], invB_sorted[-1], nx)
    if type == "cubic":
        cs = scipy.interpolate.CubicSpline(invB_sorted, Rxx_sorted, axis=0)
        Rxx_interp = cs(invBvec)
    elif type == "linear":
        Rxx_interp = np.interp(invBvec, invB_sorted, Rxx_sorted)
    return invBvec, Rxx_interp


def interp_rxx_B(B, Rxx, nx=400, type="cubic"):
    """
    Interpolate Rxx onto an evenly-spaced B grid (NOT 1/B).

    Use this when checking for oscillations periodic in B itself rather than
    in 1/B (e.g. Aharonov–Bohm-like or apparent B-periodic artifacts).
    """
    order = np.argsort(B)
    B_sorted = B[order]
    Rxx_sorted = Rxx[order, ...]

    Bvec = np.linspace(B_sorted[0], B_sorted[-1], nx)
    if type == "cubic":
        cs = scipy.interpolate.CubicSpline(B_sorted, Rxx_sorted, axis=0)
        Rxx_interp = cs(Bvec)
    elif type == "linear":
        Rxx_interp = np.interp(Bvec, B_sorted, Rxx_sorted)
    return Bvec, Rxx_interp


def make_vec_fft(invB, Rxx, n, padding=0, normalize=True, return_complex=False):
    """
    Compute FFT of Rxx in 1/B space.
    
    Parameters
    ----------
    invB : array
        Evenly-spaced 1/B values (1/T)
    Rxx : array
        Resistance values (Ω)
    n : float
        Carrier density in units of 10^12 cm^-2
    padding : int
        Zero-padding factor for FFT
    
    Returns
    -------
    psd : array
        Normalized power spectral density
    freq : array
        Frequency in units of f/(hn/e)
    """
    # bstep = invB[1] - invB[0]
    bstep = np.abs(np.mean(np.diff(invB)))
    sy, = Rxx.shape

    hanning_window = scipy.signal.windows.hann(sy)
    windowed_data = np.pad(np.multiply(Rxx, hanning_window), (sy*padding, sy*padding))
    window_fft = scipy.fft.fft(windowed_data)
    
    freq_T = scipy.fft.fftfreq(windowed_data.size, d=bstep)
    f_expected = (phi0 * n * 1E16) / 4
    
    freq = freq_T / (4 * f_expected)

    if return_complex: 
        return window_fft, freq
    else:
        window_fft = np.abs(window_fft)

    psd = window_fft**2
    if normalize:
        psd = psd / np.amax(psd)

    return psd, freq


def make_mat_fft_welch(B, Rxx, ns, nx, padding=5, normalize=True):
    psds = []
    freqs = []
    
    for i, cut in enumerate(Rxx.T):
        ne = ns[i]
        interp_invB, interp_cut = interp_rxx(B, cut, nx=nx)
        freq, psd = scipy.signal.welch(
                interp_cut, 
                fs=1/(np.median(np.diff(interp_invB))),
                nfft = len(interp_cut)*(padding+1),
        )
        if normalize:
            psd = psd / np.amax(psd)
        
        f_expected = (phi0 * ne * 1E16) / 4
        freq = freq / (4 * f_expected)

        psds.append(psd)
        freqs.append(freq)
    
    psds = np.array(psds).T  # Transpose to shape (freq, density)
    freqs = np.array(freqs).T
    
    return psds, freqs


def make_vec_fft_ls(invB, Rxx, n, padding=0, normalize=True, return_complex=False):
    # Allows for uneven spacing in invB
    freq_conversion = phi0 * n * 1E16
    # freq_to_find = np.linspace(1e-14,4*freq_conversion , num=len(Rxx)*(padding+1)) 
    bstep = bstep = np.abs(np.mean(np.diff(invB)))
    freq_to_find = scipy.fft.rfftfreq(len(Rxx)*(2*padding+1), d=bstep) + 1e-14
    fft = scipy.signal.lombscargle(
        invB, 
        Rxx,
        freq_to_find * 2 * np.pi,
        normalize='amplitude',
        floating_mean=True,
    ) * np.sqrt(len(freq_to_find)) # Match fft's norm: 'backward'
    freq_out = freq_to_find/(freq_conversion)
    
    if return_complex: 
        return fft, freq_out

    psd = np.abs(fft)**2 
    if normalize:
        psd = psd / np.amax(psd)

    return psd, freq_out

def make_vec_fft_nonuniform_periodogram(invB, Rxx, n, padding=0):
    # Could I speed this up with some fft/convolution thm tricks? Probably. Have I done that here? No.
    freq_conversion = phi0 * n * 1E16
    # freq_to_find = np.linspace(1e-14,4*freq_conversion , num=len(Rxx)*(padding+1)) 
    bstep = bstep = np.abs(np.mean(np.diff(invB)))
    freq_to_find = scipy.fft.rfftfreq(len(Rxx)*(2*padding+1), d=bstep) + 1e-14
    # freq_to_find = np.linspace(0, 2*(phi0 * n * 1E16) , num=(padding+1)*len(xs) ) + 1e-14
    fft = np.zeros_like(freq_to_find, dtype=complex)
    for i, freq in enumerate(freq_to_find):
        fft[i] = np.sum(
            Rxx * np.exp(-2j * np.pi * invB * freq)
        ) 
    f_expected = (phi0 * n * 1E16) / 4
    freqs = freq_to_find / (4 * f_expected)
    return freqs, fft

def make_mat_fft_nonuniform_periodogram(B, Rxx, ns, padding=5):
    invB = 1/(B+1e-14)
    ffts = []
    freqs = []
    for i, cut in enumerate(Rxx.T):
        ne = ns[i]
        freq, fft = make_vec_fft_nonuniform_periodogram(invB, cut, ne, padding)

        ffts.append(fft)
        freqs.append(freq)
    
    ffts = np.array(ffts).T  # Transpose to shape (freq, density)
    freqs = np.array(freqs).T
    return ffts, freqs


def make_mat_fft_lombscargle(B, Rxx, ns, padding=5, return_complex=False):
    invB = 1/(B+1e-14)
    psds = []
    freqs = []
        
    for i, cut in enumerate(Rxx.T):
        ne = ns[i]
        f_expected = (phi0 * ne * 1E16) / 4
        freq_to_find = np.linspace(0, 2*(phi0 * np.max(ns) * 1E16) , num=len(cut)*(padding+1)) + 1e-14 # Add 1e-14 to avoid zero frequency
        # Currrently use max(ns) to make an actual grid... 
        # detrend_cut, bg, _ = detrend_savgol(invB, cut,  win=401, poly=3, max_win_frac=0.4) 
        psd = scipy.signal.lombscargle(
            invB, 
            # cut - np.mean(cut),
            cut,
            freq_to_find * 2 * np.pi,
            normalize='amplitude',
            floating_mean=True,
        )
        psds.append(psd)
        freqs.append(freq_to_find/(4 * f_expected))

    psds = np.array(psds).T  # Transpose to shape (freq, density)
    freqs = np.array(freqs).T

    if return_complex:
        return psds, freqs
    
    psds = np.abs(psds)**2

    return psds, freqs


def make_mat_fft(B, Rxx, ns, nx=400, win=401, padding=5, poly=3, max_win_frac=0.4, normalize=True):
    """
    Compute FFT of Rxx in 1/B space for multiple densities.
    
    Parameters
    ----------
    B : array
        Magnetic field values (T)
    Rxx : array (2D)
        Resistance values (Ω), shape (len(B), len(ns))
    ns : array
        Carrier densities in units of 10^12 cm^-2
    nx : int
        Number of interpolation points for 1/B
    win : int
        Savitzky-Golay filter window size
    padding : int
        Zero-padding factor for FFT
    poly : int
        Polynomial order for Savitzky-Golay filter
    max_win_frac : float
        Maximum window fraction for detrending
    
    Returns
    -------
    psds : array
        Power spectral density (positive frequencies only), shape (fft_length//2, len(ns))
    freqs : array
        Frequency in units of f/(hn/e) (positive only), shape (fft_length//2, len(ns))
    """
    psds = []
    freqs = []
    
    for i, cut in enumerate(Rxx.T):
        ne = ns[i]
        interp_invB, interp_cut = interp_rxx(B, cut, nx=nx)
        cut_detrended, bg = detrend_savgol(interp_invB, interp_cut, win=win, poly=poly, max_win_frac=max_win_frac)
        psd, freq = make_vec_fft(interp_invB, cut_detrended, ne, padding=padding, normalize=normalize)
        psds.append(psd[:len(freq)//2])
        freqs.append(freq[:len(freq)//2])
    
    psds = np.array(psds).T  # Transpose to shape (freq, density)
    freqs = np.array(freqs).T
    
    return psds, freqs

def make_mat_fft_bw(B, Rxx, ns, nx=400, cutoff=5, order=2, normalize=True, padding=5, interp_type='cubic'):
    """
    Compute FFT of Rxx in 1/B space for multiple densities.
    
    Parameters
    ----------
    B : array
        Magnetic field values (T)
    Rxx : array (2D)
        Resistance values (Ω), shape (len(B), len(ns))
    ns : array
        Carrier densities in units of 10^12 cm^-2
    nx : int
        Number of interpolation points for 1/B
    cutoff : int
        Savitzky-Golay filter window size
    order : int
        Polynomial order for Savitzky-Golay filter
    normalize : bool
        Whether to normalize the power spectral density
    
    Returns
    -------
    psds : array
        Power spectral density (positive frequencies only), shape (fft_length//2, len(ns))
    freqs : array
        Frequency in units of f/(hn/e) (positive only), shape (fft_length//2, len(ns))
    """
    psds = []
    freqs = []
    
    for i, cut in enumerate(Rxx.T):
        ne = ns[i]
        interp_invB, interp_cut = interp_rxx(B, cut, nx=nx, type=interp_type)
        cut_detrended, bg, = detrend_butterworth(interp_invB, interp_cut, cutoff=cutoff, order=order, mode="highpass")
        psd, freq = make_vec_fft(interp_invB, cut_detrended, ne, padding=padding, normalize=normalize)
        psds.append(psd[:len(freq)//2])
        freqs.append(freq[:len(freq)//2])
    
    psds = np.array(psds).T  # Transpose to shape (freq, density)
    freqs = np.array(freqs).T
    
    return psds, freqs


def find_1d_peaks(fv, fft, dist=10, prom=None):
    """
    Find peaks in 1D FFT spectrum.
    
    Parameters
    ----------
    fv : array
        Frequency values
    fft : array
        FFT magnitude values
    dist : int
        Minimum distance between peaks (in samples)
    prom : float, optional
        Minimum prominence for peak detection
    
    Returns
    -------
    peaks : zip iterator
        Iterator of (frequency, magnitude) tuples for each peak
    """
    peak_ind = scipy.signal.find_peaks(fft, distance=dist, prominence=prom)[0]
    peak_xval = fv[peak_ind]
    peak_yval = fft[peak_ind]

    return zip(peak_xval, peak_yval)


def find_dR_div_R(B, Rxx, normB=1, tol=0.001):
    """
    Calculate normalized resistance change ΔR/R.
    
    Takes in a Rxx of shape (M, N) and a B field vector of shape (M,). Returns
    deltaRxx / Rxx(B=normB). Delta is found by subtracting Rxx(B) - Rxx(B=normB).
    
    Parameters
    ----------
    B : array
        Magnetic field values (T), shape (M,)
    Rxx : array
        Resistance values (Ω), shape (M, N)
    normB : float
        Normalization field value (T)
    tol : float
        Tolerance for finding normB in the B array
    
    Returns
    -------
    dRxx : array
        Normalized resistance change ΔR/R, shape (M, N)
    invB : array
        1/B values (1/T)
    """
    normBind = np.where(abs(B - normB) < tol)[0][0]
    normR = Rxx[normBind, :]

    dRxx = np.zeros(Rxx.shape)
    for i in range(len(Rxx)):
        dRxx[i] = (Rxx[i, :] - normR) / normR

    invB = 1 / B

    return dRxx, invB


# ==============================================================================
# Background subtraction methods
# ==============================================================================

def detrend_spline(x, y, n_knots=6):
    B_lo = x.min()
    B_hi = x.max()
    knots_invB = np.linspace(1/B_hi, 1/B_lo, n_knots)[1:-1]  # interior only
    knots = 1/knots_invB[::-1]
    
    bkg = SplineModel(prefix='bkg_', xknots=knots)
    out = bkg.fit(y, x=x)
    y_proc = y - out.best_fit
    y_bkg = out.best_fit
    y_proc = scipy.signal.detrend(y_proc, type="linear")
    return y_proc, y_bkg


def detrend_savgol(invB, y, win='auto', poly=3, max_win_frac=0.2):
    """
    Subtract a smooth baseline using Savitzky-Golay filter.

    Parameters
    ----------
    invB : array
        1/B values (used for estimating oscillation period when win='auto')
    y : array
        Rxx values to detrend
    win : int or 'auto'
        Window length (odd integer). Should be longer than oscillation period in 1/B.
        If 'auto', adaptively sets window to max_win_frac of data length,
        capped at 401 points.
    poly : int
        Polynomial order for local fits.
    max_win_frac : float
        Maximum window size as fraction of data length (used when win='auto').
        Default 0.2 means window is at most 20% of the data.

    Returns
    -------
    out : array
        Detrended signal with zero mean.
    bg : array
        Extracted background.
    win_used : int
        Actual window size used (useful when win='auto').
    """
    y = np.asarray(y, float)
    n = len(y)

    # Adaptive window sizing
    if win == 'auto':
        win = min(401, int(n * max_win_frac))
    else:
        win = int(win)

    # Ensure odd
    if win % 2 == 0:
        win += 1

    # Ensure valid for data length
    max_allowed = n - (1 - n % 2)  # largest odd <= n-1
    win = min(win, max_allowed)

    # Ensure valid for polynomial order
    if win < poly + 2:
        out = y - np.nanmean(y)
        return out, np.full_like(y, np.nanmean(y)), win

    bg = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    out = y - bg
    out -= np.nanmean(out)
    return out, bg


def detrend_divide(invB, y, win='auto', poly=3, max_win_frac=0.4):
    """
    Normalize by a smooth baseline (divide instead of subtract).
    
    This preserves relative oscillation amplitude better than subtraction,
    and is less likely to remove real structure when you have few periods.

    Parameters
    ----------
    invB : array
        1/B values
    y : array
        Rxx values to detrend
    win : int or 'auto'
        Window length for Savitzky-Golay smoothing.
    poly : int
        Polynomial order for local fits.
    max_win_frac : float
        Maximum window size as fraction of data length.
        Default 0.4 (larger than subtract method to be gentler).

    Returns
    -------
    out : array
        Normalized signal: (R - R_smooth) / R_smooth = R/R_smooth - 1
    bg : array
        Extracted background (smoothed R).
    win_used : int
        Actual window size used.
    """
    y = np.asarray(y, float)
    n = len(y)

    # Adaptive window sizing
    if win == 'auto':
        win = min(201, int(n * max_win_frac))
    else:
        win = int(win)

    # Ensure odd
    if win % 2 == 0:
        win += 1

    # Ensure valid for data length
    max_allowed = n - (1 - n % 2)
    win = min(win, max_allowed)

    # Ensure valid for polynomial order
    if win < poly + 2:
        bg = np.full_like(y, np.nanmean(y))
        out = y / bg - 1
        return out, bg, win

    bg = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    
    # Avoid division by zero
    bg_safe = np.where(np.abs(bg) < 1e-10, 1e-10, bg)
    out = y / bg_safe - 1  # fractional deviation from background
    
    return out, bg, win


def detrend_linear(invB, y, degree=1):
    """
    Remove only a linear (or quadratic) trend - the gentlest option.
    
    Use this when you have very few oscillation periods and don't want
    to remove any curvature that might be real signal.

    Parameters
    ----------
    invB : array
        1/B values
    y : array
        Rxx values to detrend
    degree : int
        1 = linear (just slope), 2 = quadratic. Default 1.

    Returns
    -------
    out : array
        Detrended signal.
    bg : array
        Linear/quadratic background.
    """
    y = np.asarray(y, float)
    invB = np.asarray(invB, float)
    
    idx = np.isfinite(y) & np.isfinite(invB)
    coeff = np.polyfit(invB[idx], y[idx], degree)
    bg = np.polyval(coeff, invB)
    out = y - bg
    out -= np.nanmean(out)
    
    return out, bg


def detrend_endpoints(invB, y):
    """
    Subtract a line connecting the first and last points.
    
    The absolute gentlest option - only removes the overall tilt,
    preserves all curvature.

    Parameters
    ----------
    invB : array
        1/B values
    y : array
        Rxx values

    Returns
    -------
    out : array
        Detrended signal.
    bg : array
        Linear background.
    """
    y = np.asarray(y, float)
    invB = np.asarray(invB, float)
    
    # Line from first to last point
    slope = (y[-1] - y[0]) / (invB[-1] - invB[0])
    bg = y[0] + slope * (invB - invB[0])
    
    out = y - bg
    out -= np.nanmean(out)
    
    return out, bg


def detrend_butterworth(invB, y, cutoff=None, order=2, mode="highpass",
                        nan_policy="interpolate", uniform_tol=1e-3):
    """
    Butterworth detrending in uniformly sampled invB = 1/B space.

    Parameters
    ----------
    invB : array
        1/B values. Should be evenly spaced (uniform grid).
    y : array
        Signal (e.g., Rxx) to detrend.
    cutoff : float
        Cutoff frequency in the SAME UNITS as an FFT frequency axis for y(invB):
        cycles per (1/B) i.e. cycles per (T^-1), numerically "Tesla".
        Example: SdH peak at f=20 -> you might choose cutoff ~ 2-5.
    order : int
        Butterworth order (2-4 typical).
    mode : {"highpass","lowpass"}
        "highpass" returns oscillatory component as `out`.
        "lowpass" returns background component as `out`.
    nan_policy : {"interpolate","raise"}
        How to handle NaNs in y.
    uniform_tol : float
        Relative tolerance for uniform spacing check.

    Returns
    -------
    out : array
        Requested component (high-pass oscillations or low-pass background).
    other : array
        Complementary component (bg or oscillations), computed as y - out.
    """
    invB = np.asarray(invB, float)
    y = np.asarray(y, float)

    # Check uniform spacing
    d = np.diff(invB)
    d_med = np.median(d)
    if not np.allclose(d, d_med, rtol=uniform_tol, atol=0):
        raise ValueError("invB must be uniformly spaced (resample onto a uniform 1/B grid first).")

    # NaN handling
    if np.isnan(y).any():
        if nan_policy == "raise":
            raise ValueError("y contains NaNs; filter requires finite values.")
        elif nan_policy == "interpolate":
            x = np.arange(y.size)
            m = np.isfinite(y)
            y = np.interp(x, x[m], y[m])
        else:
            raise ValueError("nan_policy must be 'interpolate' or 'raise'.")

    # Sampling frequency and Nyquist
    fs = 1.0 / abs(d_med)       # samples per (T^-1)
    nyq = 0.5 * fs

    if cutoff is None:
        cutoff = 8/(np.max(invB) - np.min(invB)) 
        # raise ValueError("Provide cutoff explicitly (recommended) to avoid removing real low-f SdH content.")
        

    Wn = np.clip(cutoff / nyq, 1e-6, 0.999)

    btype = "highpass" if mode == "highpass" else "lowpass"
    sos = butter(order, Wn, btype=btype, output="sos")

    out = sosfiltfilt(sos, y)
    other = y - out

    # Optional: center oscillatory component
    if mode == "highpass":
        out -= np.mean(out)

    return out, other


def detrend_poly(B, y, degree=5):
    """
    Subtract a polynomial baseline (original method).

    Parameters
    ----------
    B : array
        Magnetic field values.
    y : array
        Rxx values.
    degree : int
        Polynomial degree.

    Returns
    -------
    out : array
        Detrended signal.
    bg : array
        Polynomial background.
    """
    y = np.asarray(y, float)
    idx = np.isfinite(y) & np.isfinite(B)
    coeff = np.polyfit(B[idx], y[idx], degree)
    bg = np.polyval(coeff, B)
    out = y - bg
    out -= np.nanmean(out)
    return out, bg


# ==============================================================================
# InfluxDB data retrieval
# ==============================================================================

def _get_file_time_range(file_number, data_path="./data"):
    """Read first and last unix timestamps from a data file."""
    path = os.path.join(data_path, str(file_number), "data.tsv.gz")
    first_t = last_t = None
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if first_t is None:
                first_t = float(row[0])
            last_t = float(row[0])
    return first_t, last_t


def _load_influxdb_token():
    """Load InfluxDB token from .env file next to this script."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("INFLUXDB_TOKEN="):
                return line.split("=", 1)[1]
    raise ValueError("INFLUXDB_TOKEN not found in .env")


def fetch_influxdb(file_number, measurement, metric, unit=None,
                   bucket="023_toploader", data_path="./data"):
    """Fetch a variable from InfluxDB for the time range of a data file.

    Reads the start/end timestamps from the file's data.tsv.gz and queries
    InfluxDB for the requested measurement/metric pair.

    Parameters
    ----------
    file_number : int
        Numbered data directory to look up.
    measurement : str
        InfluxDB measurement name (e.g. "avs_bridge").
    metric : str
        Metric tag to filter on (e.g. "mixing_ch_low").
    unit : str, optional
        Unit tag to filter on. Only applied if provided.
    bucket : str
        Bucket to query. Default "023_toploader".
    data_path : str
        Path to the data directory. Default "./data".

    Returns
    -------
    times : np.ndarray
        Unix epoch timestamps.
    values : np.ndarray
        Field values.
    """
    from influxdb_client import InfluxDBClient

    t_start, t_stop = _get_file_time_range(file_number, data_path)
    token = _load_influxdb_token()

    url = "https://metrics.aaronsharpe.science"
    org = "dgg"

    start_dt = datetime.fromtimestamp(t_start, tz=timezone.utc)
    stop_dt = datetime.fromtimestamp(t_stop, tz=timezone.utc)

    start_rfc = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_rfc = stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = (
        f'from(bucket: "{bucket}")'
        f'  |> range(start: {start_rfc}, stop: {stop_rfc})'
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
        f'  |> filter(fn: (r) => r["metric"] == "{metric}")'
        f'  |> filter(fn: (r) => r["_field"] == "value")'
    )
    if unit is not None:
        query += f'  |> filter(fn: (r) => r["unit"] == "{unit}")'

    with InfluxDBClient(url=url, token=token, org=org) as client:
        tables = client.query_api().query(query, org=org)

    times, values = [], []
    for table in tables:
        for record in table.records:
            times.append(record.get_time().timestamp())
            values.append(record.get_value())

    return np.array(times), np.array(values)


# rc-param presets live in external JSON so they are the single source of truth,
# shared by every notebook (no per-notebook inline copies).
_RC_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root (utils/ sits at <root>/utils) — home of the shared figs/ scratch dir.
_REPO_ROOT = os.path.dirname(_RC_DIR)


def _load_rc_params(name):
    with open(os.path.join(_RC_DIR, name)) as f:
        return json.load(f)


def update_rc_params(style='paper'):
    """Apply an rc-param preset to matplotlib.

    style : {'paper', 'talk'}
        'paper' — small fonts / scaled ticks for multipanel print figures.
        'talk'  — larger fonts, full-size ticks, no forced white frame; for
                  single-panel and diagnostic/presentation plots.

    Presets live in utils/rcparams_<style>.json (single source of truth)."""
    if style not in ('paper', 'talk'):
        raise ValueError("style must be 'paper' or 'talk', got %r" % (style,))
    plt.rcParams.update(_load_rc_params('rcparams_%s.json' % style))


def save_fig(fig, name, formats=('pdf', 'png'), dpi=600, transparent=False,
             pad_inches=0.02, dirpath=None):
    """Save `fig` into the shared repo-root figs/ scratch dir with the
    repo-standard settings (dpi=600, bbox_inches='tight', pad_inches=0.02,
    transparent=False).

    By default (`dirpath=None`) figures land in <repo root>/figs/ regardless of
    which notebook folder is the working directory; pass `dirpath` to override.
    The dir is created if missing and is git-ignored, so outputs never get
    committed. If `name` carries an extension ('fig1.pdf'), only that format is
    written; otherwise every format in `formats` (PDF + PNG by default) is
    written. Returns the list of output paths."""
    if dirpath is None:
        dirpath = os.path.join(_REPO_ROOT, 'figs')
    os.makedirs(dirpath, exist_ok=True)
    base, ext = os.path.splitext(name)
    exts = [ext.lstrip('.')] if ext else list(formats)
    paths = []
    for e in exts:
        path = os.path.join(dirpath, base + '.' + e)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches,
                    transparent=transparent)
        paths.append(path)
    return paths
