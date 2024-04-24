import mne
import yasa
import pandas as pd
import numpy as np
import importlib
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})

data_dir = '/path/to/openbci-session'

# git pull https://github.com/preraulab/multitaper_toolbox/
multitaper_dir = '/path/to/multitaper_toolbox' 

mt = module_from_file('mt', os.path.join(multitaper_dir, 'python/multitaper_spectrogram_python.py'))

# all recorded sessions
sessions_all = {
    '1.1': {'ref':'Pz','file': '2024-03-24_15-11-04-max-OBCI_F9.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 60, 'end': 60+300},
              {'type': 'REST', 'start': 640-300, 'end': 640}],
          'bads': []},
    '1.2': {'ref':'Pz','file': '2024-03-21_21-47-20-max-OBCI_F8.TXT.bdf',
          'periods': [
              {'type': 'NSDR', 'start': 82, 'end': 82+300},
              {'type': 'NSDR', 'start': 672-300, 'end': 672}],
          'bads': []},
    '2.1': {'ref':'Pz','file': '2024-03-24_21-21-43-max-OBCI_FA.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 60, 'end': 60+300},
              {'type': 'REST', 'start': 640-300, 'end': 640}],
          'bads': []},
    '2.2': {'ref':'Pz','file': '2024-03-24_21-40-11-max-OBCI_FB.TXT.bdf',
          'periods': [
              {'type': 'NSDR', 'start': 86, 'end': 86+300},
              {'type': 'NSDR', 'start': 640-300, 'end': 640}],
          'bads': []},
    '3.1': {'ref':'Oz','file': '2024-04-08_19-28-26-max-OBCI_FD.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 112, 'end': 112+300},
              {'type': 'REST', 'start': 680-300, 'end': 680}],
          'bads': []},
    '3.2': {'ref':'Oz','file': '2024-04-08_19-50-06-max-OBCI_FE.TXT.bdf',
          'periods': [
              {'type': 'NSDR', 'start': 60, 'end': 60+300},
              {'type': 'NSDR', 'start': 600-300, 'end': 600}]},
    '3.3': {'ref':'Oz','file': '2024-04-08_20-10-42-max-OBCI_FF.TXT.bdf',
          'periods': [
              {'type': 'CHANT', 'start': 90, 'end': 90+300},
              {'type': 'CHANT', 'start': 690-300, 'end': 690}],
          'bads': []},
    '4': {'ref':'M1','file': '2024-04-09_11-38-14-max-OBCI_01.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 50, 'end': 50+250},
              {'type': 'NSDR', 'start': 360, 'end': 360+300},
              {'type': 'NSDR', 'start': 860-300, 'end': 860}],
          'bads': []},
    '5': {'ref':'M1','file': '2024-04-18_20-04-38-max-OBCI_02.TXT.bdf',
          'periods': [
              {'type': 'CHANT', 'start': 131, 'end': 131+300},
              {'type': 'CHANT', 'start': 131+300, 'end': 131+300*2},
              {'type': 'CHANT', 'start': 915-300, 'end': 915}],
          'bads': []},
    '6': {'ref':'M1','file': '2024-04-18_21-08-06-max-OBCI_03.TXT.bdf',
          'periods': [
              {'type': 'DANBR', 'start': 172, 'end': 172+300},
              {'type': 'DANBR', 'start': 846-300, 'end': 846}],
          'bads': []},
    '7.1': {'ref':'M1','file': '2024-04-21_22-33-04-max-OBCI_F2.TXT.bdf',
          'periods': [
              {'type': 'REST-EYE', 'start': 90, 'end': 205},
              {'type': 'REST', 'start': 425, 'end': 600}],
          'bads': []},
    '7.2': {'ref':'M1','file': '2024-04-21_23-07-57-max-OBCI_F3.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 65, 'end': 150},
              {'type': 'REST-EYE', 'start': 155, 'end': 225},
              # {'type': 'OPENM', 'start': 240, 'end': 480},
              {'type': 'DANBR', 'start': 530, 'end': 530+300},
              {'type': 'DANBR', 'start': 530+300, 'end': 530+300*2},
              {'type': 'DANBR', 'start': 1380-300, 'end': 1380}],
          'bads': [[940,980],[1280,1330]]},
    '8': {'ref':'T5','file': '2024-04-24_08-50-35-max-OBCI_EC.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 60, 'end': 180},
              {'type': 'REST-EYE', 'start': 191, 'end': 360},
              {'type': 'DANBR', 'start': 405, 'end': 1260}
              ],
          'bads': []},
    # '': {'ref':'','file': '',
    #       'periods': [
    #           {'type': 'REST', 'start': 60, 'end': 60+300},
    #           {'type': 'REST', 'start': 640-300, 'end': 640}],
    #       'bads': []},
    
    }

# https://www.nature.com/articles/s41598-023-27528-0
bpf = [.5,None] # band pass filter
# nf = [50,1]
nf = None # notch filter
nj = 10 # n_jobs for multiprocessing
epoch_time = 20; epoch_overlap = 10; overlap_factor = 1 / (epoch_time / (epoch_time - epoch_overlap))
sessions_analyse = ['8'] # selected sessions to analyse
ref_types = ['ORI','LM', 'AR', 'REST','LT', 'Fpz']
ref_type = ref_types[5]
plots = ['Spectrum','Topomap','Frequency', 'Bands']
pipelines = ['epoch_reject','raw'] # raw does not reject and no epoch splitting
pipeline = pipelines[0]

# select sessions
sessions = {key: sessions_all[key] for key in sessions_analyse}
n_sessions = len(sessions)
n_periods = 0

for key in sessions:
    n_periods = n_periods + len(sessions[key]['periods'])

session_colors = {'NSDR':'red', 'CHANT':'violet', 'DANBR': 'blue', 'REST':'grey'}
reject_alpha = .3
reject_color = 'grey'

# bands list for topoplots
bp_bands = [
    (1, 4, "Delta"),
    (4, 6, "Theta 1"),
    (6, 8, "Theta 2"),
    (8, 10, "Alpha 1"),
    (10, 12, "Alpha 2"),
    (12, 16, "Beta 1"),
    (15, 25, "Beta 2"),
    (25, 30, "Beta 3"),
    (30, 40, "Gamma 1"),
    (40, 48, "Gamma 2"),
]
bp_bands_dict = dict()
for b in range(len(bp_bands)):
    bp_bands_dict[bp_bands[b][2]] = (bp_bands[b][0], bp_bands[b][1])

unit_l = 'dB(µV²/Hz)'; unit_p = 'µV²'; unit_pdb = 'dB(µV²)'; unit_a = 'µV'

# process raw sessions files
raws = []; epochs = []; epochs_bad = []
for index, key in enumerate(sessions):
    ref = sessions[key]['ref']
    raw = mne.io.read_raw_bdf(os.path.join(data_dir, sessions[key]['file']), preload=True, verbose=True)
    raw.add_reference_channels(ref)
    
    # remove '-ref' postfix from OpenBCI channel names and accelerometer channels
    ch = raw.ch_names.copy()
    ch = [x.replace('-'+ref, '') for x in ch]
    raw.rename_channels(dict(zip(raw.ch_names, ch)))
    ch.remove('ACC_X'); ch.remove('ACC_Y'); ch.remove('ACC_Z')
    
    # pick EEG channels
    raw.pick(ch)
    
    # apply notch and bandpass filters
    if nf is not None:
        raw.notch_filter(freqs=nf[0], notch_widths=nf[1], n_jobs=nj)
    raw.filter(bpf[0], bpf[1], n_jobs=nj)
    
    # apply montage
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage , match_case=False, on_missing="ignore")
    
    # Re-referencing: 
    # REST
    if ref_type == 'REST':
        sphere = mne.make_sphere_model("auto", "auto", raw.info)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=5.0)
        forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
        raw_rest = raw.copy().set_eeg_reference("REST", forward=forward)
        raw = raw_rest

    # digitally linked mastoids
    elif (ref == 'M1') and ('M2' in ch) and (ref_type == 'LM'):
        raw.set_eeg_reference(ref_channels=['M1','M2'])

    # digitally linked T5 and T6 if no mastoids as approx
    elif (ref == 'T5') and ('T6' in ch) and (ref_type == 'LT'):
        raw.set_eeg_reference(ref_channels=['T5','T6'])
        
    # average reference
    elif ref_type == 'AR':
        raw.set_eeg_reference(ref_channels = 'average')

    # original reference channel, no modification
    elif ref_type == 'ORI': 
        raw.drop_channels(ref)

    # specific channel for re-reference
    else:
        raw.set_eeg_reference(ref_channels = [ref_type])
        
    sessions[key]['ref_l'] = ref_type
    raws.append(raw)
    
    # epoch data and reject bad epochs, incuding manually set in sessions
    if pipeline == 'epoch_reject':
        epochs_raw = mne.make_fixed_length_epochs(raw, duration=epoch_time, preload=True, overlap=epoch_overlap)
        epochs.append(epochs_raw)
        from autoreject import AutoReject
        ar = AutoReject(n_jobs=10)
        epochs_clean = ar.fit_transform(epochs_raw)
        rejected_log = ar.get_reject_log(epochs_raw)
        rej_bad = rejected_log.bad_epochs
        # rejected_log.plot_epochs(epochs_raw)
        manu_bad = np.array([])
        for be in range(len(sessions[key]['bads'])):
            bad_start = sessions[key]['bads'][be][0]
            bad_end = sessions[key]['bads'][be][1]
            bad_epoch_start = round((1/overlap_factor) * bad_start / epoch_time)
            bad_epoch_end = round((1/overlap_factor) * bad_end / epoch_time)
            bad_range = np.arange(bad_epoch_start, bad_epoch_end, 1)
            manu_bad = np.append(manu_bad, bad_range)
        epochs_bad.append(np.unique(np.append(np.where(rej_bad)[0], manu_bad)))

if 'Spectrum' in plots:
    # Multitaper spectrogram from Prerau Labs Multitaper Toolbox
    # https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    frequency_range = [1, 48]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = 6 # Set time-half bandwidth
    num_tapers = time_bandwidth*2 - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [16, 4]  # Window size is 4s with step size of 1s
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = False  # use multiprocessing
    cpus = 3  # use 3 cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = False  # plot spectrogram
    return_fig = False # do not return plotted spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = False  # print extra info
    xyflip = False  # do not transpose spect output matrix
    spect_vlim = [-10,35]
    
    for index, key in enumerate(sessions):
        ref = sessions[key]['ref']
        ref_l = sessions[key]['ref_l']
        title = key
        raw = raws[index]
        ch = raw.ch_names
        fig, axes = plt.subplots(round(len(ch)/2+0.5),2, 
                 figsize=(16, len(ch)*2))
        fig.suptitle(f'Session #{key} Multitaper spectrogram')
        axes = axes.flatten()
    
        for e in range(len(ch)):
            spect, stimes, sfreqs = mt.multitaper_spectrogram(
                raw.get_data(units='uV')[e], raw.info['sfreq'], frequency_range, time_bandwidth, num_tapers, window_params, min_nfft, detrend_opt, multiprocess, cpus,
                weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
            spect_data = mt.nanpow2db(spect)
            
            start_time = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
            times = [start_time + timedelta(seconds=int(s)) for s in stimes]
            
            dtx = times[1] - times[0]
            dy = sfreqs[1] - sfreqs[0]
            x_s = mdates.date2num(times[0]-dtx)
            x_e = mdates.date2num(times[-1]+dtx)
            extent = [x_s, x_e, sfreqs[-1]+dy, sfreqs[0]-dy]
    
            ax = axes[e]
            im = ax.imshow(
                spect_data, extent=extent, aspect='auto', 
                cmap=plt.get_cmap('jet'), 
                vmin = spect_vlim[0], vmax = spect_vlim[1],
                )
            ax.xaxis_date()  # Interpret x-axis values as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))  # Format time as HH:MM:SS
            fig.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)
            
            # highlights periods on graph with vertical lines and labels
            for p_index in range(len(sessions[key]['periods'])):
                p_label = sessions[key]['periods'][p_index]['type']
                p_start = sessions[key]['periods'][p_index]['start']
                p_end = sessions[key]['periods'][p_index]['end']
                ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_start))), color='w', linestyle='--', linewidth = 1)
                ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_end))), color='w', linestyle='--', linewidth = 1)
                ax.text(mdates.date2num(start_time + timedelta(seconds=int(p_start + (p_end - p_start)/2))), frequency_range[1]-5,  p_label, color='white', ha='center', va='center', fontsize=12, bbox=dict(facecolor='black', alpha=0.35))
            ax.invert_yaxis()
            
            # plot bad epochs as grey rectangles
            if pipeline == 'epoch_reject':
                bads = epochs_bad[index]
                for b in range(len(bads)):
                    x1_num = mdates.date2num(start_time + timedelta(seconds=int((bads[b]*overlap_factor)*epoch_time)))
                    x2_num = mdates.date2num(start_time + timedelta(seconds=int((bads[b]*overlap_factor)*epoch_time + epoch_time)))
                    
                    # Calculate lower-left corner and dimensions
                    lower_left_x = min(x1_num, x2_num)
                    width = abs(x2_num - x1_num)
                    lower_left_y = min(spect_vlim[0], spect_vlim[1])
                    height = abs(spect_vlim[1] - spect_vlim[0])
                    rect = patches.Rectangle((lower_left_x, lower_left_y), width, height, 
                     linewidth=1, edgecolor=reject_color, facecolor=reject_color, alpha=reject_alpha)
                    ax.add_patch(rect)

            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f'{ch[e]}-{ref_l}')
            tick_intervals = np.linspace(x_s, x_e, 11)  # 11 points include 0% to 100%
            ax.set_xticks(tick_intervals)
            
            # Scale colormap
            if clim_scale:
                clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
                im.set_clim(clim)  # actually change colorbar scale
        plt.tight_layout()

# Build topomaps with Amplitude from yasa.bandpower
if 'Topomap' in plots:
    fig, axes = plt.subplots(n_periods,len(bp_bands_dict), 
             figsize=(len(bp_bands)*2, n_periods*2))
    fig.suptitle(f'Ampitude ({pipeline})')
    per_i = 0
    for index, key in enumerate(sessions):
        ref = sessions[key]['ref']
        ref_l = sessions[key]['ref_l']
        title = key
        raw = raws[index]
        ch = raw.ch_names
        for p_index in range(len(sessions[key]['periods'])):
            p_label = sessions[key]['periods'][p_index]['type']
            p_start = sessions[key]['periods'][p_index]['start']
            p_end = sessions[key]['periods'][p_index]['end']
            raw_p = raw.copy()
            raw_p.crop(tmin=p_start, tmax=p_end)
            axs = axes[per_i]; per_i = per_i + 1
            for b in range(len(bp_bands)):
                ax = axs[b]
                band = bp_bands[b]
                bps = []
                if pipeline == 'epoch_reject':
                    epochs_list = epochs[index].get_data(units='uV')
                    for e in range(len(epochs_list)):
                        if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                            bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=[band], bandpass=False, relative=False)
                            bps.append(np.array(bpy[band[2]]))
                    bps = np.array(bps).mean(axis=0)
                else:
                    bps = yasa.bandpower(raw_p, raw.info["sfreq"], bands=[band], bandpass=False, relative=False)
                    bps = np.array(bps[band[2]])

                bp = np.sqrt(bps)
                p_max = np.max(bp)
                p_min = np.min(bp)
                p_max = p_max if p_max > 4 else 4
                vlim = (p_min, p_max)
                im, _ = mne.viz.plot_topomap(
                    bp, 
                    raw.info,
                    cmap=cm.jet,
                    axes=ax,
                    vlim=vlim,
                    show=False)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, format='%0.1f', ticks = [vlim[0], vlim[1]], aspect=10)
                cbar.ax.set_position([0.85, 0.1, 0.05, 0.8])
                cbar.set_label(unit_a)
                bl = f'{band[2]} ({band[0]} - {band[1]} Hz)'
                if b == 0:
                    ax.set_title(f'#{title} {p_label}-{p_index+1} {bl}')
                elif b == 1:
                    ax.set_title(f'#{bl}, ref: {ref_l}')
                else:
                    ax.set_title(f'{bl}')
    plt.tight_layout()

# Power frequency plot with mne PSD computation
if 'Frequency' in plots:
    fig, axes = plt.subplots(round(n_periods/2+.5),2,
             figsize=(16, n_periods*3))
    fig.suptitle(f'PSD ({pipeline})')
    axes = axes.flatten()
    per_i = 0
    for index, key in enumerate(sessions):
        ref = sessions[key]['ref']
        ref_l = sessions[key]['ref_l']
        title = key
        raw = raws[index]
        ch = raw.ch_names
        for p_index in range(len(sessions[key]['periods'])):
            p_label = sessions[key]['periods'][p_index]['type']
            p_start = sessions[key]['periods'][p_index]['start']
            p_end = sessions[key]['periods'][p_index]['end']
            raw_p = raw.copy()
            raw_p.crop(tmin=p_start, tmax=p_end)
    
            # exclude bad epochs
            if pipeline == 'epoch_reject':
                epochs_list = epochs[index].copy()
                epoch_data = epochs_list.get_data()
                min_e = round((1/overlap_factor) * p_start / epoch_time + 0.5)
                max_e = round((1/overlap_factor) * p_end / epoch_time - 0.5)
                to_drop = epochs_bad[index]
                to_drop = np.append(to_drop, np.arange(0,min_e,1))
                to_drop = np.append(to_drop, np.arange(max_e,len(epoch_data),1))
                epochs_list.drop(np.unique(to_drop[to_drop < len(epochs_list)]))
                spectrum = epochs_list.compute_psd(fmin=1.0, fmax=48, n_jobs=10, 
                   method='welch')
                spectrum = spectrum.average()
            else:
                spectrum = raw_p.compute_psd(fmin=1.0, fmax=48, n_jobs=10)                
            psds_ori, freqs = spectrum.get_data(return_freqs=True)
            
            # https://github.com/mne-tools/mne-python/issues/9868
            # https://mne.tools/stable/auto_tutorials/time-freq/10_spectrum_class.html
        
            psds = psds_ori * (1e6 ** 2) # convert to power uV^2 / Hz
            psds = 10 * np.log10(psds) # convert to dB
            
            ax = axes[per_i]; per_i = per_i + 1
            for c in range(len(ch)):
                ax.plot(freqs, psds[c], label=f'{ch[c]}', linewidth=1)
            ax.legend()
            ax.set(title='PSD', xlabel='Frequency (Hz)', ylabel=unit_l)
            ax.set_title(f'#{title} {p_label}-{p_index+1}')
    plt.tight_layout()

# Power Bands over time, simplified bands list with yasa.bandpower
if 'Bands' in plots:
    bp_bands_simple = [
        (1, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 30, "Beta"),
        (30, 48, "Gamma"),
    ]
    
    for index, key in enumerate(sessions):
        ref = sessions[key]['ref']
        ref_l = sessions[key]['ref_l']
        title = key
        raw = raws[index]
        ch = raw.ch_names
        
        b_window = 30; b_step = 5
        fig, axes = plt.subplots(round(len(ch)/2+.5),2, figsize=(14,len(ch)*2))
        fig.suptitle(f'#{title} Band Power by channel  ({pipeline})')
        axes = axes.flatten()
        b_max = 35
        x_ts = None
        bp_a = [[] for _ in range(len(bp_bands_simple))]
        for c in range(len(ch)):
            t, eeg_2d = yasa.sliding_window(raw.get_data(units='uV')[c], raw.info["sfreq"], window=b_window, step=b_step)
            bp = yasa.bandpower(eeg_2d, raw.info["sfreq"], bands=bp_bands_simple, bandpass=False, relative=False)
            ts = np.arange(1,len(bp)+1,1) * b_step
            if x_ts is None:
                start_time = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
                x_ts = [start_time + timedelta(seconds=int(s)) for s in ts]
                x_s = mdates.date2num(x_ts[0])
                x_e = mdates.date2num(x_ts[-1])
                total_seconds = (x_ts[-1] - x_ts[0]).total_seconds()
                tick_interval_seconds = total_seconds / 10
                tick_locator = mdates.SecondLocator(interval=int(tick_interval_seconds))
            ax = axes[c]
            for b in range(len(bp_bands_simple)):
                band = 10 * np.log10(bp[bp_bands_simple[b][2]])
                bp_a[b].append(np.array(band))
                bl = f'{bp_bands_simple[b][2]} ({bp_bands_simple[b][0]} - {bp_bands_simple[b][1]} Hz)'
                ax.plot(x_ts, band, label=bl, linewidth=1)
                ax.set(title=f'{ch[c]}-{ref_l}', xlabel='Time', ylabel=unit_pdb)
                
            # highlights periods on graph with vertical lines and labels
            for p_index in range(len(sessions[key]['periods'])):
                p_label = sessions[key]['periods'][p_index]['type']
                p_start = sessions[key]['periods'][p_index]['start']
                p_end = sessions[key]['periods'][p_index]['end']
                ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_start))), color='black', linestyle='--', linewidth = 1)
                ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_end))), color='black', linestyle='--', linewidth = 1)
                ax.text(mdates.date2num(start_time + timedelta(seconds=int(p_start + (p_end - p_start)/2))), 
                        b_max - 5,  p_label, color='black', ha='center', va='center', fontsize=12, bbox=dict(facecolor='grey', alpha=0.35))
            
            # plot bad epochs as grey rectangles
            if pipeline == 'epoch_reject':
                bads = epochs_bad[index]
                for b in range(len(bads)):
                    x1_num = mdates.date2num(start_time + timedelta(seconds=int(bads[b]*overlap_factor*epoch_time)))
                    x2_num = mdates.date2num(start_time + timedelta(seconds=int(bads[b]*overlap_factor*epoch_time + epoch_time)))
                    
                    # Calculate lower-left corner and dimensions
                    lower_left_x = min(x1_num, x2_num)
                    width = abs(x2_num - x1_num)
                    lower_left_y = min(spect_vlim[0], spect_vlim[1])
                    height = abs(spect_vlim[1] - spect_vlim[0])
                    rect = patches.Rectangle((lower_left_x, lower_left_y), width, height, 
                     linewidth=1, edgecolor=reject_color, facecolor=reject_color, alpha=reject_alpha)
                    ax.add_patch(rect)

            ax.set_ylim(0,b_max)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
            ax.xaxis.set_major_locator(tick_locator)
        plt.tight_layout()
    
        bp_a = np.array(bp_a)
        bp_mean = bp_a.mean(axis=1)
        bp_std = bp_a.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.suptitle(f'#{title} Band Power ({pipeline})')
    
        for b in range(len(bp_bands_simple)):
            bl = f'{bp_bands_simple[b][2]} ({bp_bands_simple[b][0]} - {bp_bands_simple[b][1]} Hz)'
            ax.plot(x_ts, bp_mean[b], linewidth=1, label=bl)
            ax.fill_between(x_ts, bp_mean[b] - bp_std[b], bp_mean[b] + bp_std[b], alpha=0.15, label=bl)
            ax.set(xlabel='Time',ylabel=unit_pdb)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
            ax.xaxis.set_major_locator(tick_locator)
            ax.set_ylim(0,b_max)
        
        # highlights periods on graph with vertical lines and labels
        for p_index in range(len(sessions[key]['periods'])):
            p_label = sessions[key]['periods'][p_index]['type']
            p_start = sessions[key]['periods'][p_index]['start']
            p_end = sessions[key]['periods'][p_index]['end']
            ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_start))), color='black', linestyle='--', linewidth = 1)
            ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_end))), color='black', linestyle='--', linewidth = 1)
            ax.text(mdates.date2num(start_time + timedelta(seconds=int(p_start + (p_end - p_start)/2))), 
                    b_max-5,  p_label, color='black', ha='center', va='center', fontsize=12, bbox=dict(facecolor='grey', alpha=0.35))
        
        # plot bad epochs as grey rectangles
        if pipeline == 'epoch_reject':
            bads = epochs_bad[index]
            for b in range(len(bads)):
                x1_num = mdates.date2num(start_time + timedelta(seconds=int(bads[b]*overlap_factor*epoch_time)))
                x2_num = mdates.date2num(start_time + timedelta(seconds=int(bads[b]*overlap_factor*epoch_time + epoch_time)))
                
                # Calculate lower-left corner and dimensions
                lower_left_x = min(x1_num, x2_num)
                width = abs(x2_num - x1_num)
                lower_left_y = min(spect_vlim[0], spect_vlim[1])
                height = abs(spect_vlim[1] - spect_vlim[0])
                rect = patches.Rectangle((lower_left_x, lower_left_y), width, height, 
                 linewidth=1, edgecolor=reject_color, facecolor=reject_color, alpha=reject_alpha)
                ax.add_patch(rect)
        plt.tight_layout()
    plt.rcParams.update({"font.size": old_fontsize})

