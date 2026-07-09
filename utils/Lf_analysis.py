from multiprocessing import Value
import os
import sys

module_path = os.path.abspath(os.path.join('../measureme'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sweep import sweep_load as sl
from sweep import raster
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import colors
from cycler import cycler
from matplotlib.colors import Normalize
import matplotlib.image as mpimg

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, SymmetricalLogLocator
import matplotlib.patches as mpatches

from functools import lru_cache
    
from utils import *
# import cmcrameri.cm as cmc

# Physical constants
from scipy.constants import h, hbar, e, m_e
from scipy.constants import k as k_B
# h = 6.626E-34  # Planck constant (J·s)
# e = 1.6002E-19  # Elementary charge (C)
phi0 = h / e    # Flux quantum (Wb)
Rq = h/e**2
from matplotlib.gridspec import GridSpecFromSubplotSpec
file_path = '../data/raw_sweeps' # raw sweep data (download from SDR; see README)



class Lf_analysis:
    def __init__(self, file, contact):
        data = sl.pload(file_path, file)
        self.B_raw = data['ys'][0]
        self.Rxx_raw = data[f'sr830_{contact}_X']/data['sr830_4_X']
        Vtgs = data['xs'][0]
        Vbgs = data['xs'][1]
        self.density, self.displacement = calculate_n_D(Vtgs, Vbgs, dtg=rmg19['default']['dtg'], dbg=rmg19['default']['dbg'])

    # @lru_cache(maxsize=1)
    def get_bg_subtracted_data(self, bg_method, bg_method_params, blims, return_bg=False):
        """ 
            Do background subtraction on data 
            bg_method is function that takes in a row in B, Rxx and subtracts it
            blims is (low, high) bounds for B
        """
        # Limit our B
        blo, bhi = blims
        bmask = (blo < self.B_raw) & (self.B_raw < bhi)
        bflat = np.abs(np.diff(self.B_raw[bmask])) > 0 # get rid of parts where b is flat
        bmask[bmask] = bmask[bmask] & np.append(bflat, 1) & np.insert(bflat, 0, 1)
        B = self.B_raw[bmask] 
        Rxx_raw = self.Rxx_raw[bmask]    
        # Do a background subtraction on each line
        Rxx_detrended = np.zeros_like(Rxx_raw)
        for i, cut in enumerate(Rxx_raw.T):
            Rxx_detrended[:,i], _ = bg_method(B, cut, **bg_method_params)
        if return_bg:
            return B, Rxx_detrended, Rxx_raw - Rxx_detrended
        return B, Rxx_detrended

    def get_fft_B(self, bg_method, bg_method_params, padding, blims):
        # Gets the FFT in B, rather than 1/B
        B, Rxx_detrended = self.get_bg_subtracted_data(bg_method, bg_method_params, blims)
        n_B = len(B)
        ffts = np.zeros((int(n_B*(2*padding+1)/2), len(self.density)), dtype=complex)
        freqs = np.zeros((int(n_B*(2*padding+1)/2), len(self.density)))
        for i, cut in enumerate(Rxx_detrended.T):
            fft, freq = make_vec_fft(B, cut, n=1, padding=padding, return_complex=True)
            ffts[:,i] = fft[:len(freq)//2] / len(freq) # Divide by N for units of Ohms
            freqs[:,i] = freq[:len(freq)//2] * (phi0 * 1E16) # Back to freq in 1/T

        return ffts, freqs

    def get_fft(self, bg_method, bg_method_params, padding, blims):
        B, Rxx_detrended = self.get_bg_subtracted_data(bg_method, bg_method_params, blims)
        invB_range = np.max(1/B) - np.min(1/B)
        num_invB_pts = int(np.ceil(invB_range / np.min(np.abs(np.diff(1/B)))))
        ffts = np.zeros((int(num_invB_pts*(2*padding+1)/2), len(self.density)), dtype=complex)
        freqs = np.zeros((int(num_invB_pts*(2*padding+1)/2), len(self.density)))
        for i, cut in enumerate(Rxx_detrended.T):
            interp_invB, interp_cut = interp_rxx(B, cut, nx=num_invB_pts, type='cubic')
            fft, freq = make_vec_fft(interp_invB, interp_cut, n=self.density[i], padding=padding, return_complex=True)
            ffts[:,i] = fft[:len(freq)//2] / len(freq)
            freqs[:,i] = freq[:len(freq)//2]

        return ffts, freqs
    

    def get_fft_subinvB(self, bg_method, bg_method_params, padding, blims):
        # Do the background subtraction in 1/B
        B, Rxx = self.get_bg_subtracted_data(lambda y, x: (x,0), {}, blims)
        invB_range = np.max(1/B) - np.min(1/B)
        num_invB_pts = int(np.ceil(invB_range / np.min(np.abs(np.diff(1/B)))))
        ffts = np.zeros((int(num_invB_pts*(2*padding+1)/2), len(self.density)), dtype=complex)
        freqs = np.zeros((int(num_invB_pts*(2*padding+1)/2), len(self.density)))
        for i, cut in enumerate(Rxx.T):
            interp_invB, interp_cut = interp_rxx(B, cut, nx=num_invB_pts, type='cubic')
            cut_dentrended, _ = bg_method(interp_invB, interp_cut, **bg_method_params)
            fft, freq = make_vec_fft(interp_invB, cut_dentrended, n=self.density[i], padding=padding, return_complex=True)
            ffts[:,i] = fft[:len(freq)//2] / len(freq)
            freqs[:,i] = freq[:len(freq)//2]

        return ffts, freqs
    
    def get_fft_pg(self, pg_method, bg_method, bg_method_params, padding, blims):
        # Get fft using periodogram rather than interpolation
        B, Rxx_detrended = self.get_bg_subtracted_data(bg_method, bg_method_params, blims)
        invB = 1/(B+1e-14)
        ffts = []
        freqs = []
        for i, cut in enumerate(Rxx_detrended.T):
            ne = self.density[i]
            if pg_method == 'nonuniform_brute':
                freq, fft = make_vec_fft_nonuniform_periodogram(invB, cut, ne, padding)
            elif pg_method == 'lombscargle':
                fft, freq = make_vec_fft_ls(invB, cut, ne, padding, return_complex=True)
            else:
                raise ValueError(f'Invalid Periodogram Method {pg_method}')

            ffts.append(fft)
            freqs.append(freq) / len(freq)
        
        ffts = np.array(ffts).T   # Transpose to shape (freq, density)
        freqs = np.array(freqs).T

        return ffts, freqs

    
    def get_fft_line(self, target_density, bg_method, bg_method_params, padding, blims):
        # As get fft, but only get one line in density
        density_idx = np.argmin(np.abs(self.density-target_density))
        blo, bhi = blims
        bmask = (blo < self.B_raw) & (self.B_raw < bhi)
        bflat = np.abs(np.diff(self.B_raw[bmask])) > 0 # get rid of parts where b is flat
        bmask[bmask] = bmask[bmask] & np.append(bflat, 1) & np.insert(bflat, 0, 1)
        B = self.B_raw[bmask] 
        Rxx_raw = self.Rxx_raw[bmask, density_idx]        
        Rxx_detrended, _ = bg_method(B, Rxx_raw, **bg_method_params)
        invB_range = np.max(1/B) - np.min(1/B)
        num_invB_pts = int(np.ceil(invB_range / np.min(np.abs(np.diff(1/B)))))
        interp_invB, interp_cut = interp_rxx(B, Rxx_detrended, nx=num_invB_pts, type='cubic')
        psd, freq = make_vec_fft(interp_invB, interp_cut, n=self.density[density_idx], padding=padding, normalize=False)
        psd = psd[:len(freq)//2] / len(freq)
        freq = freq[:len(freq)//2]
        return psd, freq

    @staticmethod
    def get_fft_peak(freq, fft, target_peak_val):
        # Given a cut in freq and peak, find peaks most closely matching target_peak_vals
        peaks, _ = find_peaks(np.abs(fft)**2, prominence=0.1)
        if len(peaks) < 1:
            return np.nan, np.nan
    
        peak_v = peaks[np.argmin(np.abs(freq[peaks] - target_peak_val))]
        
        return fft[peak_v], freq[peak_v]

    def plot_blim_effect(self, fig, ax, norm, bg_method, bg_method_params, padding, bhi_max, blo_min, target_density, target_freq, n_B_lims=60, plot_dimensionless=False):
        min_window_len = 0.2

        b_mins = np.linspace(blo_min, 2.0, num=n_B_lims)
        b_maxes = np.linspace(blo_min+min_window_len, bhi_max, num=n_B_lims)

        fft_coefs_out = np.zeros((n_B_lims, n_B_lims)) * np.nan
        freqs_out = np.zeros((n_B_lims, n_B_lims)) * np.nan

        density_val = self.density[np.argmin(np.abs(self.density-target_density))]        
        for j, blo in enumerate(b_mins):
            for k, bhi in enumerate(b_maxes):
                if bhi - blo < min_window_len:
                    # Skip windows that are too small
                    fft_coefs_out[j,k], freqs_out[j,k] = np.nan, np.nan
                    continue
                
                psds, freqs = self.get_fft_line(target_density, bg_method, bg_method_params, padding, blims=(blo, bhi))
                fft, freq = self.get_fft_peak(freqs, psds, target_freq)
                fft_coefs_out[j,k], freqs_out[j,k] = np.abs(fft)**2, freq
        
        ax.set_facecolor('k')
        if plot_dimensionless:
            plot = ax.pcolormesh(
                b_mins, b_maxes, freqs_out.T,  rasterized=True,
                norm=norm, cmap='RdBu_r'
            )
        else:
            plot = ax.pcolormesh(
                b_mins, b_maxes, freqs_out.T*density_val,  rasterized=True,
                norm=norm, cmap='RdBu_r'
            )
        
        ax.set_xlabel(r"Low $B$ cutoff (T)")
        ax.set_ylabel(r"High $B$ cutoff (T)")
        return plot
    

    def plot_bw_filter_effect(self, fig, ax, norm, padding, target_density, target_freq, n_filters=60, plot_dimensionless=False):
        min_window_len = 0.2

        filters = np.linspace(0, 30.0, num=n_filters)
        orders = np.arange(7, dtype=int)

        fft_coefs_out = np.zeros((n_filters, 7)) * np.nan
        freqs_out = np.zeros((n_filters, 7)) * np.nan

        density_val = self.density[np.argmin(np.abs(self.density-target_density))]        
        for j, filter_cutoff in enumerate(filters):
            for k, order in enumerate(orders):
              
                psds, freqs = self.get_fft_line(target_density, detrend_butterworth, {'cutoff': filter_cutoff, 'order': order}, padding, blims=(0.62, 4.98))
                fft_coefs_out[j,k], freqs_out[j,k] = self.get_fft_peak(freqs, psds, target_freq)
        
        ax.set_facecolor('k')
        if plot_dimensionless:
            plot = ax.pcolormesh(
                filters/ (phi0 * 1E16), orders, freqs_out.T,  rasterized=True,
                norm=norm, cmap='RdBu_r'
            )
        else:
            plot = ax.pcolormesh(
                filters/ (phi0 * 1E16), orders, freqs_out.T*density_val,  rasterized=True,
                norm=norm, cmap='RdBu_r'
            )
        
        ax.set_xlabel(r"Filter Cutoff ($n_{\mathrm{SdH}}$)")
        ax.set_ylabel("Filter Order")
        return plot


    def plot_lf(self, fig, ax, norm, bg_method, bg_method_params, blims=(0.51, 2.49), cmap='inferno', filter_invB=False):
        if filter_invB:
            B, Rxx = self.get_bg_subtracted_data(lambda y, x: (x,0), {}, blims)
            invB_range = np.max(1/B) - np.min(1/B)
            num_invB_pts = int(np.ceil(invB_range / np.min(np.abs(np.diff(1/B)))))
            Rxx_detrended = np.zeros((num_invB_pts, len(self.density)))
            for i, cut in enumerate(Rxx.T):
                interp_invB, interp_cut = interp_rxx(B, cut, nx=num_invB_pts, type='cubic')
                
                Rxx_detrended[:,i], _ = bg_method(interp_invB, interp_cut, **bg_method_params)
            B = 1/interp_invB

        else:
            B, Rxx_detrended = self.get_bg_subtracted_data(bg_method, bg_method_params, blims)

        plot = ax.pcolormesh(
            self.density, B, Rxx_detrended, norm=norm, cmap=cmap, rasterized=True, lw=0
        )
        ax.set_ylabel(r'$B$ (T)')
        ax.set_xlabel(r'$n$ $\left(10^{12} \mathrm{cm}^{-2}\right)$')
        return plot

    def plot_fft(self, fig, ax, norm, bg_method, bg_method_params, padding, blims=(0.51, 2.49), target_density=None):
        ffts, freqs = self.get_fft(bg_method, bg_method_params, padding, blims)
        psds = np.abs(ffts)**2 
        # psds = psds/np.max(psds)
        density_repeated = np.tile(self.density, (freqs.shape[0], 1))
        if norm is None:
            # Default to lognorm
            norm = colors.LogNorm(vmin=np.quantile(psds,0.2), vmax=np.quantile(psds,0.99))

        plot = ax.pcolormesh(density_repeated, freqs, psds, norm=norm, cmap='viridis', rasterized=True, lw=0)
        ax.set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylabel(r'$f/(nh/e)$')
        ax.set_ylim(0, 3)

        if target_density is not None:
            density_idx = np.argmin(np.abs(target_density-self.density))
            ax.vlines(self.density[density_idx], 0, 2, 'r', linestyle='dashed', alpha=0.8, lw=2)
            psd, freq = psds[:,density_idx], freqs[:,density_idx]
            target_peaks = np.array([1.155, 0.925])/self.density[density_idx]
            for peak in target_peaks:
                p, f = self.get_fft_peak(freq, psd, peak)
                print(f)

        return plot


    

    def plot_fft_B(self, fig, ax, norm, bg_method, bg_method_params, padding, blims=(0.51, 2.49)):
        ffts, freqs = self.get_fft_B(bg_method, bg_method_params, padding, blims)
        freqs = freqs*phi0*1e15 # Convert to nm2
        density_repeated = np.tile(self.density, (freqs.shape[0], 1))
        if norm is None:
            # Default to lognorm
            norm = colors.LogNorm(vmin=np.quantile(psds,0.2), vmax=np.quantile(psds,0.99))
        # Grey out freq cutoff
        psds = np.abs(ffts)**2
        # psds = psds/np.max(psds)
        plot = ax.pcolormesh(density_repeated, freqs, psds, norm=norm, cmap='cividis', rasterized=True, lw=0)
        ax.set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylabel(r'$f_B$ ($10^{3}$ nm$^2$)')
        return plot


    def plot_fft_invcm(self, fig, ax, norm, bg_method, bg_method_params, padding, blims=(0.51, 2.49), target_densities=None, return_data=False, filter_invB=False):
        if filter_invB:
            ffts, freqs = self.get_fft_subinvB(bg_method, bg_method_params, padding, blims)
        else:
            ffts, freqs = self.get_fft(bg_method, bg_method_params, padding, blims)
        
        density_repeated = np.tile(self.density, (freqs.shape[0], 1))
        if norm is None:
            # Default to lognorm
            norm = colors.LogNorm(vmin=np.quantile(psds,0.2), vmax=np.quantile(psds,0.99))
        # Grey out freq cutoff
        psds = np.abs(ffts)**2
        # psds = psds/np.max(psds)
        plot = ax.pcolormesh(density_repeated, freqs*density_repeated, psds, norm=norm, cmap='viridis', rasterized=True, lw=0)
        
        
        ax.set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylabel(r'$n_{\mathrm{SdH}}$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylim(0, 2)

        linecut_out = []

        if return_data:
            return plot, (ffts, freqs)

        if target_densities is not None:
            for target_density in target_densities:
                density_idx = np.argmin(np.abs(target_density-self.density))  
                n = self.density[density_idx]
                psd, freq =  psds[:,density_idx], freqs[:,density_idx]*n
                linecut_out.append((psd, freq))

            # density_idx = np.argmin(np.abs(target_density-self.density))
            # ax.vlines(self.density[density_idx], 0, 2, 'r', linestyle='dashed', alpha=0.8, lw=2)
            # psd, freq =  psds[:,density_idx], freqs[:,density_idx]
            # target_peaks = np.array([1.155, 0.925])/self.density[density_idx]
            # for peak in target_peaks:
            #     p, f = self.get_fft_peak(freq, psd, peak)
            #     print(f*self.density[density_idx])
            return plot, linecut_out
        return plot


    

    def plot_fft_periodogram(self, pg_method, fig, ax, norm, bg_method, bg_method_params, padding, blims=(0.51, 2.49), target_densities=None):
        ffts, freqs = self.get_fft_pg(pg_method, bg_method, bg_method_params, padding, blims)
        density_repeated = np.tile(self.density, (freqs.shape[0], 1))
        if norm is None:
            # Default to lognorm
            norm = colors.LogNorm(vmin=np.quantile(psds,0.2), vmax=np.quantile(psds,0.99))
        # Grey out freq cutoff
        psds = np.abs(ffts)**2
        # psds = psds/np.max(psds)
        plot = ax.pcolormesh(density_repeated, freqs*density_repeated, psds, norm=norm, cmap='viridis', rasterized=True, lw=0)
        
        
        ax.set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylabel(r'$n_{\mathrm{SdH}}$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylim(0, 2)

        linecut_out = []
        if target_densities is not None:
            for target_density in target_densities:
                density_idx = np.argmin(np.abs(target_density-self.density))  
                n = self.density[density_idx]
                psd, freq =  psds[:,density_idx], freqs[:,density_idx]*n
                linecut_out.append((psd, freq))
            return plot, linecut_out
        return plot



    def plot_fft_invcm_phase(self, fig, ax, norm, bg_method, bg_method_params, padding, blims=(0.51, 2.49), target_density=None):
        ffts, freqs = self.get_fft(bg_method, bg_method_params, padding, blims)
        angles = np.angle(ffts, deg=True) %180
        density_repeated = np.tile(self.density, (freqs.shape[0], 1))
        if norm is None:
            # Default to lognorm
            norm = colors.Normalize(vmin=0, vmax=180)

        plot = ax.pcolormesh(density_repeated, freqs*density_repeated, angles, norm=norm, cmap='twilight', rasterized=True, lw=0)
        ax.set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylabel(r'$n_{\mathrm{SdH}}$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylim(0, 2)

        if target_density is not None:
            density_idx = np.argmin(np.abs(target_density-self.density))
            ax.vlines(self.density[density_idx], 0, 2, 'r', linestyle='dashed', alpha=0.8, lw=2)
            psd, freq =  angles[:,density_idx], freqs[:,density_idx]
            target_peaks = np.array([1.155, 0.925])/self.density[density_idx]
            for peak in target_peaks:
                p, f = self.get_fft_peak(freq, psd, peak)
                print(f*self.density[density_idx])
         

        return plot



    def plot_fft_grad(self, fig, ax, norm, bg_method, bg_method_params, padding, blims=(0.51, 2.49)):
        ffts, freqs = self.get_fft(bg_method, bg_method_params, padding, blims)
        psds = np.abs(ffts)**2
        density_repeated = np.tile(self.density, (freqs.shape[0], 1))
        # def fft_deriv(linecut):
        #     # derivative down to a constant
        #     N = 6*len(linecut)
        #     fft = scipy.fft.rfft(linecut, n=N)
        #     freq = scipy.fft.rfftfreq(N)
        #     return scipy.fft.irfft(1j*freq*fft)[:len(linecut)]
            
        # deriv = np.apply_along_axis(fft_deriv, 0, psds)
        deriv = np.gradient(psds, axis=0)
        # deriv /= np.mean(np.abs(deriv), axis=0)
        plot = ax.pcolormesh(density_repeated, freqs*density_repeated, 1/np.abs(deriv), norm=norm, cmap='viridis', rasterized=True, lw=0)
        ax.set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylabel(r'$n_{\mathrm{SdH}}$ ($10^{12}$ cm$^{-2}$)')
        ax.set_ylim(0, 3)
        return plot, deriv

    # def get_peak_locs(self, fig, ax, bg_method, bg_method_params, padding, blims=(0.51, 2.49), density_lims=(0.35, 0.65)):
    #     nlo, nhi = density_lims
    #     density_mask = (nlo < self.density) & (self.density < nhi)
    #     target_peaks = np.array([1.155, 0.925])/self.density[density_mask,np.newaxis]

    #     ffts, freqs = self.get_fft(bg_method, bg_method_params, padding, blims)
    #     peaks_out = np.zeros_like(target_peaks)
    #     peaks_coefs = np.zeros_like(target_peaks, dtype=complex)
    #     for i, (density, peaks, fft, freq) in enumerate(zip(self.density[density_mask], target_peaks, ffts.T, freqs.T)):
    #         for j, peak in enumerate(peaks):
    #             ft, fr = self.get_fft_peak(freq, fft, peak)
    #             peaks_out[i,j] = fr*density # Output freq in 10^12 cm-2
    #             peaks_coefs[i,j] = ft
    #     ax.plot(self.density[density_mask], peaks_out[:,0], '.')
    #     ax.plot(self.density[density_mask], peaks_out[:,1], '.')
    #     return plot, peaks_out, peaks_coefs




