import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import mne
import os
import pandas as pd
import sys
import yaml
import yasa

from autoreject import AutoReject
from datetime import datetime, timedelta
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import simps

# config, if relative path not working then use explicit path to working dir (repo dir with scripts and yml) or modify working directory in IDE/GUI settings
# working_dir = '/path/to/openbci-session'
working_dir = os.getcwd()
cfg_file = os.path.join(working_dir, "session_analyse.yml")

# rename sleep_analysis.yml.sample to sleep_analysis.yml and set directories
with open(cfg_file, "r") as yamlfile:
    cfg_base = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg = cfg_base['default']

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
nj = 15 # n_jobs for multiprocessing, usually n_cpu - 1

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})

# git pull https://github.com/preraulab/multitaper_toolbox/
os.chdir(cfg['multitaper_dir'])
from multitaper_spectrogram_python import multitaper_spectrogram, nanpow2db
os.chdir(working_dir)

# all recorded sessions
sessions_all = {
    '1.1': {'to_ref': 'REST', 'ref':'Pz',
            'file': '2024-03-24_15-11-04-max-OBCI_F9.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 60, 'end': 60+300},
              {'type': 'REST', 'start': 640-300, 'end': 640}],
          'bads': []},
    '1.2': {'to_ref': 'REST', 'ref':'Pz',
            'file': '2024-03-21_21-47-20-max-OBCI_F8.TXT.bdf',
          'periods': [
              {'type': 'NSDR', 'start': 82, 'end': 82+300},
              {'type': 'NSDR', 'start': 672-300, 'end': 672}],
          'bads': []},
    '2.1': {'to_ref': 'REST', 'ref':'Pz',
            'file': '2024-03-24_21-21-43-max-OBCI_FA.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 60, 'end': 60+300},
              {'type': 'REST', 'start': 640-300, 'end': 640}],
          'bads': []},
    '2.2': {'to_ref': 'REST', 'ref':'Pz',
            'file': '2024-03-24_21-40-11-max-OBCI_FB.TXT.bdf',
          'periods': [
              {'type': 'NSDR', 'start': 86, 'end': 86+300},
              {'type': 'NSDR', 'start': 640-300, 'end': 640}],
          'bads': []},
    '3.1': {'to_ref': 'REST', 'ref':'Oz',
            'file': '2024-04-08_19-28-26-max-OBCI_FD.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 112, 'end': 112+300},
              {'type': 'REST', 'start': 680-300, 'end': 680}],
          'bads': []},
    '3.2': {'to_ref': 'REST', 'ref':'Oz',
            'file': '2024-04-08_19-50-06-max-OBCI_FE.TXT.bdf',
          'periods': [
              {'type': 'NSDR', 'start': 60, 'end': 60+300},
              {'type': 'NSDR', 'start': 600-300, 'end': 600}]},
    '3.3': {'ref':'Oz','file': '2024-04-08_20-10-42-max-OBCI_FF.TXT.bdf',
          'periods': [
              {'type': 'CHANT', 'start': 90, 'end': 90+300},
              {'type': 'CHANT', 'start': 690-300, 'end': 690}],
          'bads': []},
    '4': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-04-09_11-38-14-max-OBCI_01.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 50, 'end': 50+250},
              {'type': 'NSDR', 'start': 360, 'end': 360+300},
              {'type': 'NSDR', 'start': 860-300, 'end': 860}],
          'bads': []},
    '5': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-04-18_20-04-38-max-OBCI_02.TXT.bdf',
          'periods': [
              {'type': 'CHANT', 'start': 131, 'end': 131+300},
              {'type': 'CHANT', 'start': 131+300, 'end': 131+300*2},
              {'type': 'CHANT', 'start': 915-300, 'end': 915}],
          'bads': []},
    '6': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-04-18_21-08-06-max-OBCI_03.TXT.bdf',
          'periods': [
              {'type': 'DANBR', 'start': 172, 'end': 172+300},
              {'type': 'DANBR', 'start': 846-300, 'end': 846}],
          'bads': []},
    '7.1': {'to_ref': 'REST', 'ref':'M1',
            'file': '2024-04-21_22-33-04-max-OBCI_F2.TXT.bdf',
          'periods': [
              # {'type': 'REST-EYE', 'start': 90, 'end': 205},
              {'type': 'REST', 'start': 425, 'end': 600}],
          'bads': []},
    '7.2': {'to_ref': 'REST', 'ref':'M1',
            'file': '2024-04-21_23-07-57-max-OBCI_F3.TXT.bdf',
          'periods': [
               {'type': 'REST', 'start': 65, 'end': 150},
              #  {'type': 'REST-EYE', 'start': 155, 'end': 225},
               # {'type': 'OPENM', 'start': 240, 'end': 480},
                # {'type': 'DANBR', 'start': 530, 'end': 530+300},
                # {'type': 'DANBR', 'start': 530+300, 'end': 530+300*2},
              # {'type': 'DANBR', 'start': 1380-300, 'end': 1380},
              {'type': 'DANBR', 'start': 530, 'end': 1380}
              ],
          'bads': [
                    [1120,1150],
                   ]},
    '8': {'to_ref': 'REST', 'ref':'T5',
          'file': '2024-04-24_08-50-35-max-OBCI_EC.TXT.bdf',
          'periods': [
              {'type': 'REST', 'start': 60, 'end': 180},
              # {'type': 'REST-EYE', 'start': 191, 'end': 360},
              # {'type': 'DANBR', 'start': 405, 'end': 1260}
              ],
          'bads': []},
    '9': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-04-25_22-29-48-max-OBCI_F4.TXT.bdf',
          'periods': [
                {'type': 'REST', 'start': 231, 'end': 370},
                # {'type': 'DANBR', 'start': 500, 'end': 1350},
                # {'type': 'DANBR', 'start': 500, 'end': 500+300},
                # {'type': 'DANBR', 'start': 500+300, 'end': 500+300*2},
                # {'type': 'DANBR', 'start': 1350-300, 'end': 1350}
              ],
           'bads': [
               # [231,261],[500,530],[590,630],[640,670],[680,710],
               # [720,780],[800,950],[930,1010],[1060,1130],
               # [1200,1230],[1330,1360]
               ]},
    '10': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-04-29_22-41-34-max-OBCI_F5.TXT.bdf',
          'periods': [
                {'type': 'REST', 'start': 80, 'end': 195, 'notes': 'seated'},
                {'type': 'NSDR', 'start': 270, 'end': 800, 'notes': 'seated'},
                {'type': 'REST', 'start': 820, 'end': 941, 'notes': 'seated'},
              ],
           'bads': [[170,180],[320,340],[355,370],[570,585],[760,770]
               ]},
    '12': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-05-01_18-41-13-max-OBCI_F6.TXT.bdf',
          'periods': [
                {'type': 'REST', 'start': 100, 'end': 392, 'notes': 'seated, eye-closed'},
                {'type': 'DANBR', 'start': 432, 'end': 1238, 'notes': 'seated, eye-closed'},
                {'type': 'REST', 'start': 1285, 'end': 1574, 'notes': 'seated, eye-closed'},
                {'type': 'VIPS', 'start': 1620, 'end': 2748, 'notes': 'seated, eye-closed'},
                {'type': 'REST', 'start': 2803, 'end': 3068, 'notes': 'seated, eye-closed'},
              ],
           'bads': [
               ]},
    '13': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-05-09_21-11-51-max-OBCI_05.TXT.bdf',
          'periods': [
                {'type': 'REST', 'start': 40, 'end': 240, 'notes': 'seated, eye-closed'},
                {'type': 'REST', 'start': 40, 'end': 240, 'notes': 'seated, eye-closed'},
                # {'type': 'DANBR', 'start': 264, 'end': 294, 'notes': 'seated, eye-closed'},
              ],
           'bads': [
               ]},
    '14': {'to_ref': 'REST', 'ref':'M1',
          'file': '2024-05-11_22-19-42-max-OBCI_07.TXT.bdf',
          'periods': [
                {'type': 'REST', 'start': 30, 'end': 229, 'notes': 'seated, eye-closed'},
                {'type': 'DANBR', 'start': 255, 'end': 690, 'notes': 'seated, eye-closed'},
                # {'type': 'DANBR', 'start': 264, 'end': 294, 'notes': 'seated, eye-closed'},
              ],
           'bads': [], 'bads_ch': ['']},
    # '': {'ref':'','file': '',
    #       'periods': [
    #           {'type': 'REST', 'start': 60, 'end': 60+300},
    #           {'tytspe': 'REST', 'start': 640-300, 'end': 640}],
    #       'bads': []},
    
    }
units = {'psd_dB': 'dB(µV²/Hz)', 'amp': 'µV', 'p': 'µV²', 'p_dB': 'dB(µV²)', 'rel': '%'}

load = True
# load = False
# https://www.nature.com/articles/s41598-023-27528-0
bpf = [.1,None] # band pass filter
bpf = [.5,48] # band pass filter
nf = None # notch filter, None / [50,1]
nf = [50,1] # notch filter, None / [50,1]
epoch_time = 12; epoch_overlap = 6; overlap_factor = 1 / (epoch_time / (epoch_time - epoch_overlap))
sessions_analyse = ['14'] # selected sessions to analyse
plots = ['Spectrum','Topomap','Frequency', 'Bands']
# plots = ['Frequency', 'Frequency_Epoch']
sf_to = 200

pipeline = 'epoch_reject' # 'epoch_reject' / 'epoch' / 'raw'
freq_method = 'mne_psd_welch' # 'mne_psd_welch' / 'mne_trf_morlet' / 'mne_psd_multitaper' / 'mne_tfr_multitaper'
topo_method = 'yasa_band_amp' # 'yasa_band_power' / 'mne_trf_morlet' / 'mne_fft_welch'
w_fft = 4; m_bandwidth = 1; m_freq_bandwidth = 2; tfr_time_bandwidth = 4; 

# multitaper spectrograms settings
spect_vlim = [-10,35]
spect_lim = [1,48]
time_bandwidth = 3 # Set time-half bandwidth
num_tapers = time_bandwidth*2 - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = [4, 1]  # Window size is Xs with step size of Ys

sig_specs = f'sf={sf_to}Hz, notch={nf}, bandpass={bpf}, epoch={epoch_time}s ({round(100*epoch_overlap/epoch_time)}% overlap)'
# select sessions
sessions = {key: sessions_all[key] for key in sessions_analyse}
n_sessions = len(sessions)
n_periods = 0

for key in sessions:
    n_periods = n_periods + len(sessions[key]['periods'])

session_colors = {'NSDR':'red', 'CHANT':'violet', 'DANBR': 'blue', 'REST':'grey'}
reject_alpha = .5
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

# bp_bands = [
#     (1, 4, "Delta"),
#     (4, 8, "Theta"),
#     (8, 12, "Alpha"),
#     (12, 30, "Beta"),
#     (30, 48, "Gamma"),
# ]

bp_bands_dict = dict()
for b in range(len(bp_bands)):
    bp_bands_dict[bp_bands[b][2]] = (bp_bands[b][0], bp_bands[b][1])

# process raw sessions files
if load:
    raws = []; epochs = []; epochs_bad = []
    for index, key in enumerate(sessions):
        raw = mne.io.read_raw_bdf(os.path.join(cfg['data_dir'], sessions[key]['file']), preload=True, verbose=True)
        raw.add_reference_channels(sessions[key]['ref'])
        
        # remove '-ref' postfix from OpenBCI channel names and accelerometer channels
        ch = raw.ch_names.copy()
        ch = [x.replace('-'+sessions[key]['ref'], '') for x in ch]
        raw.rename_channels(dict(zip(raw.ch_names, ch)))
        ch.remove('ACC_X'); ch.remove('ACC_Y'); ch.remove('ACC_Z')
        
        # pick EEG channels
        raw.pick(ch)
        
        # apply notch and bandpass filters
        if nf is not None:
            raw.notch_filter(freqs=nf[0], notch_widths=nf[1], n_jobs=nj)
        raw.filter(bpf[0], bpf[1], n_jobs=nj)
        
        if sf_to != raw.info['sfreq']:
            raw.resample(sfreq=sf_to)
        
        # apply montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(ten_twenty_montage , match_case=False, on_missing="ignore")
        
        # Re-referencing: 
        # REST
        if sessions[key]['to_ref'] == 'REST':
            sphere = mne.make_sphere_model("auto", "auto", raw.info)
            src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=5.0)
            forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
            raw.set_eeg_reference("REST", forward=forward)
    
        # digitally linked mastoids
        elif ('M1' in ch) and ('M2' in ch) and (sessions[key]['to_ref'] == 'LM'):
            raw.set_eeg_reference(ref_channels=['M1','M2'])
            raw.drop_channels(['M1','M2'])
    
        # digitally linked T5 and T6 if no mastoids as approx
        elif ('T5' in ch) and ('T6' in ch) and (sessions[key]['to_ref'] == 'LT'):
            raw.set_eeg_reference(ref_channels=['T5','T6'])
            raw.drop_channels(['T5','T6'])
            
        # average reference
        elif sessions[key]['to_ref'] == 'AR':
            raw.set_eeg_reference(ref_channels = 'average')
    
        # original reference channel, no modification
        elif sessions[key]['to_ref'] == 'ORI': 
            raw.drop_channels([sessions[key]['ref']])
    
        # specific channel for re-reference
        else:
            raw.set_eeg_reference(ref_channels = [sessions[key]['to_ref']])
            raw.drop_channels([sessions[key]['ref']])
            
        raws.append(raw)
        
        # epoch data and reject bad epochs, incuding manually set in sessions
        if pipeline == 'epoch_reject':
            epochs_raw = mne.make_fixed_length_epochs(raw, duration=epoch_time, preload=True, overlap=epoch_overlap)
            epochs.append(epochs_raw)
            ar = AutoReject(n_jobs=10, cv=5, consensus= [.2,.4,.6,.8], random_state=42)
            epochs_clean, rejected_log = ar.fit_transform(epochs_raw, return_log=True)
            rej_bad = ar.get_reject_log(epochs_raw).bad_epochs
            epochs_bad.append(np.unique(np.where(rej_bad)[0]))
        elif pipeline == 'epoch':
            epochs_raw = mne.make_fixed_length_epochs(raw, duration=epoch_time, preload=True, overlap=epoch_overlap)
            epochs.append(epochs_raw)

for index, key in enumerate(sessions):
    manu_bad = np.array([])
    for be in range(len(sessions[key]['bads'])):
        bad_start = sessions[key]['bads'][be][0]
        bad_end = sessions[key]['bads'][be][1]
        bad_epoch_start = int((1/overlap_factor) * bad_start / epoch_time)
        bad_epoch_end = round((1/overlap_factor) * bad_end / epoch_time + 0.5)
        bad_range = np.arange(bad_epoch_start, bad_epoch_end, 1)
        manu_bad = np.append(manu_bad, bad_range)
    if pipeline == 'epoch_reject':
        epochs_bad[index] = np.unique(np.append(epochs_bad[index], manu_bad))
    else:
        epochs_bad.append(manu_bad)
    

if pipeline == 'epoch': pipeline = 'epoch_reject'

if 'Spectrum' in plots:
    excl_spect_plot = False
    spect_time = False
    # Multitaper spectrogram from Prerau Labs Multitaper Toolbox
    # https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    frequency_range = [spect_lim[0], spect_lim[1]]  # Limit frequencies from 0 to 25 Hz
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

    
    for index, key in enumerate(sessions):
        raw = raws[index]
        ch = raw.ch_names
        if not excl_spect_plot:
            fig, axes = plt.subplots(round(len(ch)/2+0.5),2, 
                      figsize=(16, len(ch)*2))
            fig.suptitle(f'Session #{key} Multitaper spectrogram')
            axes = axes.flatten()
    
        for e in range(len(ch)):
            spect, stimes, sfreqs = multitaper_spectrogram(
                raw.get_data(units='uV')[e], raw.info['sfreq'], 
                frequency_range, time_bandwidth, num_tapers, 
                window_params, min_nfft, detrend_opt, multiprocess, cpus,
                weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
            spect_data = nanpow2db(spect)
            
            dy = sfreqs[1] - sfreqs[0]
            if spect_time:
                start_time = datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
                times = [start_time + timedelta(seconds=int(s)) for s in stimes]
                dtx = times[1] - times[0]
                x_s = mdates.date2num(times[0]-dtx)
                x_e = mdates.date2num(times[-1]+dtx)
            else:
                x_s = stimes[0]
                x_e =stimes[-1]
            extent = [x_s, x_e, sfreqs[-1]+dy, sfreqs[0]-dy]
    
            if excl_spect_plot:
                fig, ax = plt.subplots(figsize=(16,9))
                fig.suptitle(f'Session #{key} Multitaper spectrogram')
            else:
                ax = axes[e]
                axes = axes.flatten()
            
            im = ax.imshow(
                spect_data, extent=extent, aspect='auto', 
                cmap=plt.get_cmap('jet'), 
                vmin = spect_vlim[0], vmax = spect_vlim[1],
                )
            if spect_time:
                ax.xaxis_date()  # Interpret x-axis values as dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))  # Format time as HH:MM:SS
            fig.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)
            
            # highlights periods on graph with vertical lines and labels
            for p_index in range(len(sessions[key]['periods'])):
                p_label = sessions[key]['periods'][p_index]['type']
                p_start = sessions[key]['periods'][p_index]['start']
                p_end = sessions[key]['periods'][p_index]['end']
                if spect_time:
                    ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_start))), color='w', linestyle='--', linewidth = 1)
                    ax.axvline(x=mdates.date2num(start_time + timedelta(seconds=int(p_end))), color='w', linestyle='--', linewidth = 1)
                    ax.text(mdates.date2num(start_time + timedelta(seconds=int(p_start + (p_end - p_start)/2))), frequency_range[1]-5,  p_label, color='white', ha='center', va='center', fontsize=12, bbox=dict(facecolor='black', alpha=0.35))
                else:
                    ax.axvline(x=p_start, color='w', linestyle='--', linewidth = 1)
                    ax.axvline(x=p_end, color='w', linestyle='--', linewidth = 1)
                    ax.text(p_start + (p_end - p_start)/2, frequency_range[1]-5,  p_label, color='white', ha='center', va='center', fontsize=12, bbox=dict(facecolor='black', alpha=0.35))
            ax.invert_yaxis()
            
            # plot bad epochs as grey rectangles
            if pipeline == 'epoch_reject':
                bads = epochs_bad[index]
                for b in range(len(bads)):
                    if spect_time:
                        x1_num = mdates.date2num(start_time + timedelta(seconds=int((bads[b]*overlap_factor)*epoch_time)))
                        x2_num = mdates.date2num(start_time + timedelta(seconds=int((bads[b]*overlap_factor)*epoch_time + epoch_time)))
                    else:
                        x1_num = (bads[b]*overlap_factor)*epoch_time
                        x2_num = (bads[b]*overlap_factor)*epoch_time + epoch_time
                    
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
            ax.set_title(f'{ch[e]}-{sessions[key]["to_ref"]}')
            n_ticks = 16
            if excl_spect_plot:
                n_ticks = int((x_e - x_s) / (epoch_time * 2))
            tick_intervals = np.linspace(x_s, x_e, n_ticks)  # 11 points include 0% to 100%
            ax.set_xticks(tick_intervals)
            
            if excl_spect_plot:
                def custom_round(x, pos):
                    return f'{x:.0f}'  # Change '.0f' to '.1f' for one decimal place, etc.
                
                # Apply the formatter
                plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_round))

            
            # Scale colormap
            if clim_scale:
                clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
                im.set_clim(clim)  # actually change colorbar scale
        plt.tight_layout()

# Build topomaps Amplitude / PSD
if 'Topomap' in plots:
    per_i = 0; plot_type = 'Amplitude'; plot_params = ''
    psds_full_l = []
    for index, key in enumerate(sessions):
        raw = raws[index]
        ch = raw.ch_names
        psds_full_p = []
        for p_index in range(len(sessions[key]['periods'])):
            p_label = sessions[key]['periods'][p_index]['type']
            p_start = sessions[key]['periods'][p_index]['start']
            p_end = sessions[key]['periods'][p_index]['end']
            raw_p = raw.copy()
            raw_p.crop(tmin=p_start, tmax=p_end)
            
            epochs_list = epochs[index].get_data(units='uV')
            epochs_good = epochs[index].copy()
            min_e = round((1/overlap_factor) * p_start / epoch_time + 0.5)
            max_e = round((1/overlap_factor) * p_end / epoch_time - 0.5)
            epoch_bad = epochs_bad[index]
            to_drop = np.append(epoch_bad[np.logical_and(epoch_bad > min_e, epoch_bad < max_e)], np.arange(0,min_e,1))
            to_drop = np.append(to_drop, np.arange(max_e,len(epochs_good),1))
            epochs_good.drop(np.unique(to_drop), verbose=False)

            if topo_method == 'mne_psd_welch':
                # https://github.com/mne-tools/mne-python/issues/9868
                # https://mne.tools/stable/auto_tutorials/time-freq/10_spectrum_class.html
                n_fft = int(w_fft * raw.info['sfreq'])
                psds_full, s_freqs = mne.time_frequency.psd_array_welch(
                    epochs_good.get_data(units='uV'), raw.info['sfreq'],
                    fmin=spect_lim[0], fmax=spect_lim[1],
                    n_fft= n_fft, output = 'power', n_jobs = nj)
            elif topo_method == 'mne_psd_multitaper':
                psds_full, s_freqs = mne.time_frequency.psd_array_multitaper(
                    epochs_good.get_data(units='uV'), raw.info['sfreq'],
                    fmin=spect_lim[0], fmax=spect_lim[1], 
                    bandwidth=m_freq_bandwidth, 
                    output = 'power', n_jobs = nj)
            elif topo_method == 'mne_tfr_multitaper':
                s_freqs = np.arange(spect_lim[0], spect_lim[1], .5)
                n_cycles = int(epoch_time / 2)
                psds_full = mne.time_frequency.tfr_array_multitaper(
                    epochs_good.get_data(units='uV'), raw.info['sfreq'],
                    freqs=s_freqs, n_cycles=n_cycles, 
                    time_bandwidth=tfr_time_bandwidth, 
                    output = 'power', n_jobs = nj)
                psds_full = np.mean(psds_full, axis=3)
            elif topo_method == 'mne_tfr_morlet':
                s_freqs = np.arange(spect_lim[0], spect_lim[1], .5)
                n_cycles = int(epoch_time / 2)
                psds_full = mne.time_frequency.tfr_array_morlet(
                    epochs_good.get_data(units='uV'), raw.info['sfreq'],
                    s_freqs, n_cycles=n_cycles,
                    zero_mean = True, output = 'power', n_jobs = nj)
                psds_full = np.mean(psds_full, axis=3)
                psds_full_p.append(psds_full)
            else:
                psds_bp = []                
                for b in range(len(bp_bands)):
                    band = bp_bands[b]
                    if pipeline == 'epoch_reject':
                        if topo_method == 'yasa_band_amp':
                            bps = []
                            for e in range(len(epochs_list)):
                                if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                                    bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=[band], bandpass=False, relative=False)
                                    bps.append(np.array(bpy[band[2]]))
                            psds = np.array(bps).mean(axis=0)
                            psds = np.sqrt(psds) # convert to amp uV / sqrt(Hz)
                            plot_unit = units['amp']; plot_type = 'Amplitude'
                            plot_params = 'default'
                        elif topo_method == 'yasa_band_power':
                            bps = []
                            for e in range(len(epochs_list)):
                                if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                                    bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=[band], bandpass=False, relative=False)
                                    bps.append(np.array(bpy[band[2]]))
                            psds = np.array(bps).mean(axis=0)
                            plot_unit = units['p']; plot_type = 'Power'
                            plot_params = 'default'
                        elif topo_method == 'yasa_band_power_rel':
                            bps = []
                            for e in range(len(epochs_list)):
                                if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                                    bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=bp_bands, bandpass=False, relative=True)
                                    # bpy = bpy.reset_index()
                                    bps.append(np.array(bpy[bp_bands[b][2]]))
                            psds = np.array(bps).mean(axis=0)
                            # psds = np.sqrt(psds) # convert to amp uV / sqrt(Hz)
                            plot_unit = units['rel']; plot_type = 'Relative Band Power'
                            plot_params = 'relative=True'
                    psds_bp.append(psds)
                psds_full_p.append(psds_bp)
        psds_full_l.append(psds_full_p)

    fig, axes = plt.subplots(n_periods,len(bp_bands_dict), 
             figsize=(len(bp_bands)*2, n_periods*2))
    for index, key in enumerate(sessions):
        raw = raws[index]
        ch = raw.ch_names
        for p_index in range(len(sessions[key]['periods'])):
            p_label = sessions[key]['periods'][p_index]['type']
            p_start = sessions[key]['periods'][p_index]['start']
            p_end = sessions[key]['periods'][p_index]['end']
            raw_p = raw.copy()
            raw_p.crop(tmin=p_start, tmax=p_end)
            axs = axes[per_i]; per_i = per_i + 1
            
            epochs_list = epochs[index].get_data(units='uV')
            epochs_good = epochs[index].copy()
            min_e = round((1/overlap_factor) * p_start / epoch_time + 0.5)
            max_e = round((1/overlap_factor) * p_end / epoch_time - 0.5)
            epoch_bad = epochs_bad[index]
            to_drop = np.append(epoch_bad[np.logical_and(epoch_bad > min_e, epoch_bad < max_e)], np.arange(0,min_e,1))
            to_drop = np.append(to_drop, np.arange(max_e,len(epochs_good),1))
            epochs_good.drop(np.unique(to_drop), verbose=False)

            for b in range(len(bp_bands)):
                ax = axs[b]
                band = bp_bands[b]
                if pipeline == 'epoch_reject':
                    if topo_method == 'yasa_band_amp':
                        # bps = []
                        # for e in range(len(epochs_list)):
                        #     if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                        #         bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=[band], bandpass=False, relative=False)
                        #         bps.append(np.array(bpy[band[2]]))
                        # psds = np.array(bps).mean(axis=0)
                        # psds = np.sqrt(psds) # convert to amp uV / sqrt(Hz)
                        psds = psds_full_l[index][p_index][b]
                        plot_unit = units['amp']; plot_type = 'Amplitude'
                        plot_params = 'default'
                    elif topo_method == 'yasa_band_power':
                        bps = []
                        for e in range(len(epochs_list)):
                            if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                                bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=[band], bandpass=False, relative=False)
                                bps.append(np.array(bpy[band[2]]))
                        psds = np.array(bps).mean(axis=0)
                        plot_unit = units['p']; plot_type = 'Power'
                        plot_params = 'default'
                    elif topo_method == 'yasa_band_power_rel':
                        bps = []
                        for e in range(len(epochs_list)):
                            if (e not in epochs_bad[index]) and (e * overlap_factor * epoch_time >= p_start) and ((e * overlap_factor * epoch_time + epoch_time) <= p_end):
                                bpy = yasa.bandpower(epochs_list[e], raw.info["sfreq"], bands=bp_bands, bandpass=False, relative=True)
                                # bpy = bpy.reset_index()
                                bps.append(np.array(bpy[bp_bands[b][2]]))
                        psds = np.array(bps).mean(axis=0)
                        # psds = np.sqrt(psds) # convert to amp uV / sqrt(Hz)
                        plot_unit = units['rel']; plot_type = 'Relative Band Power'
                        plot_params = 'relative=True'
                    elif topo_method == 'mne_psd_welch':
                        psds_full = psds_full_l[index]
                        f_idx = np.logical_and(s_freqs >= band[0], s_freqs < band[1])
                        epoch_bps = []
                        for e in range(len(psds_full)):
                            # integrate PSD to get Power in uV
                            epoch_bps.append(np.array([simps(psds_ch[f_idx], s_freqs[f_idx]) for psds_ch in psds_full[e]]))
                        psds = np.sqrt(np.array(epoch_bps).mean(axis=0)) # convert to Amplitude
                        plot_unit = units['amp']; plot_type = 'Amplitude'
                        plot_params = f'fft_window={w_fft}s'
                    elif topo_method == 'mne_psd_multitaper':
                        f_idx = np.logical_and(s_freqs >= band[0], s_freqs < band[1])
                        psds = psds_full[...,f_idx].mean(axis=2).mean(axis=0)
                        psds = 10 * np.log10(psds) # convert to dB
                        plot_unit = units['psd_dB']; plot_type = 'PSD'
                        plot_params = f'bandwidth={m_bandwidth}'
                    elif topo_method == 'mne_tfr_multitaper':
                        f_idx = np.logical_and(s_freqs >= band[0], s_freqs < band[1])
                        psds = psds_full[...,f_idx].mean(axis=2).mean(axis=0)
                        psds = 10 * np.log10(psds) # convert to dB
                        # https://mne.discourse.group/t/units-for-tfr-multitaper-output/7125
                        plot_unit = units['psd_dB']; plot_type = 'TFR PSD'
                        plot_params = f'n_cycles={n_cycles}, time_bandwidth={tfr_time_bandwidth}'
                    elif topo_method == 'mne_tfr_morlet':
                        f_idx = np.logical_and(s_freqs >= band[0], s_freqs < band[1])
                        psds = psds_full[...,f_idx].mean(axis=2).mean(axis=0)
                        psds = 10 * np.log10(psds) # convert to dB
                        plot_unit = units['psd_dB']; plot_type = 'TFR PSD'
                        plot_params = f'n_cycles={n_cycles}'
                
                # p_max = np.max(psds)
                # p_min = np.min(psds)
                p_mins = []; p_maxs = []
                for ps in range(len(psds_full_l)):
                    psds_ps = np.array(psds_full_l[ps])
                    p_mins.append(psds_ps.min(axis=(0,2))[b])
                    p_maxs.append(psds_ps.max(axis=(0,2))[b])
                    
                p_min = min(p_mins)
                p_max = max(p_maxs)
                
                vlim = (p_min, p_max)
                if len(psds) > 1:
                    im, _ = mne.viz.plot_topomap(
                        psds, 
                        raw.info,
                        cmap=cm.jet,
                        axes=ax,
                        vlim=vlim,
                        show=False)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, format='%0.1f', ticks = [vlim[0], vlim[1]], aspect=10)
                    cbar.ax.set_position([0.85, 0.1, 0.05, 0.8])
                    cbar.set_label(plot_unit)
                bl = f'{band[2]} ({band[0]} - {band[1]} Hz)'
                if b == 0:
                    ax.set_title(f'#{key} {p_label}-{p_index+1} {bl}')
                elif b == 1:
                    ax.set_title(f'#{bl}, ref: {sessions[key]["to_ref"]}')
                else:
                    ax.set_title(f'{bl}')
    fig.suptitle(f'{plot_type} ({pipeline}, {sig_specs}, {topo_method}=[{plot_params}]')
    plt.tight_layout()

# Power frequency plot with mne PSD computation
if 'Frequency' in plots:
    fig, axes = plt.subplots(round(n_periods/2+.5),2,
             figsize=(16, n_periods*3))
    fig.suptitle(f'PSD ({pipeline})')
    axes = axes.flatten()
    per_i = 0
    for index, key in enumerate(sessions):
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
                epochs_good = epochs[index].copy()
                min_e = round((1/overlap_factor) * p_start / epoch_time + 0.5)
                max_e = round((1/overlap_factor) * p_end / epoch_time)
                epoch_bad = epochs_bad[index]
                to_drop = np.append(epoch_bad[np.logical_and(epoch_bad >= min_e, epoch_bad <= max_e)], np.arange(0,min_e,1))
                to_drop = np.array(np.append(to_drop, np.arange(max_e,len(epochs_good),1)), dtype=int)
                e_idx = np.setdiff1d(np.arange(0, len(epochs_good),1), to_drop)
                epochs_good.drop(np.unique(to_drop), verbose=False)
                if freq_method == 'mne_psd_welch':
                    n_fft = int(w_fft * raw.info['sfreq'])
                    psds, freqs = mne.time_frequency.psd_array_welch(
                        epochs_good.get_data(units='uV'), raw.info['sfreq'],
                        fmin=spect_lim[0], fmax=spect_lim[1],
                        n_fft= n_fft, output = 'power', n_jobs = nj)
                    psds_epoch = psds
                    psds = psds.mean(axis=0)
                    psds = 10 * np.log10(psds) # convert to dB
                    plot_unit = units['psd_dB']; plot_type = 'PSD'
                    plot_params = f'fft_window={w_fft}s'
                elif freq_method == 'mne_psd_multitaper':
                    psds, freqs = mne.time_frequency.psd_array_multitaper(
                        epochs_good.get_data(units='uV'), raw.info['sfreq'],
                        fmin=spect_lim[0], fmax=spect_lim[1], 
                        bandwidth=m_freq_bandwidth, 
                        output = 'power', n_jobs = nj)
                    psds_epoch = psds
                    psds = psds.mean(axis=0)
                    psds = 10 * np.log10(psds) # convert to dB
                    plot_unit = units['psd_dB']; plot_type = 'PSD'
                    plot_params = f'bandwidth={m_freq_bandwidth}'
                elif freq_method == 'mne_tfr_multitaper':
                    s_freqs = np.arange(spect_lim[0], spect_lim[1], .5)
                    n_cycles = int(epoch_time / 2)
                    psds_ori = mne.time_frequency.tfr_array_multitaper(
                        epochs_good.get_data(units='uV'), raw.info['sfreq'],
                        s_freqs, n_cycles=n_cycles, time_bandwidth=tfr_time_bandwidth, 
                        zero_mean = True, output = 'power', n_jobs = nj)
                    freqs = s_freqs; psds_epoch = psds_ori.mean(axis=3)
                    psds_ch_freq = psds_epoch.mean(axis=0)
                    psds_db = 10 * np.log10(psds_ch_freq) # convert to dB
                    psds = psds_db
                    plot_unit = units['psd_dB']; plot_type = 'TFR PSD'
                    plot_params = f'n_cycles={n_cycles}, time_bandwidth={tfr_time_bandwidth}'
                elif freq_method == 'mne_tfr_morlet':
                    s_freqs = np.arange(spect_lim[0], spect_lim[1], .5)
                    n_cycles = int(epoch_time / 2)
                    psds_ori = mne.time_frequency.tfr_array_morlet(
                        epochs_good.get_data(units='uV'), raw.info['sfreq'],
                        s_freqs, n_cycles=n_cycles,
                        zero_mean = True, output = 'power', n_jobs = nj)
                    freqs = s_freqs; psds_epoch = psds_ori.mean(axis=3)
                    psds_ch_freq = psds_epoch.mean(axis=0)
                    psds_db = 10 * np.log10(psds_ch_freq) # convert to dB
                    psds = psds_db
                    plot_unit = units['psd_dB']; plot_type = 'TFR PSD'
                    plot_params = f'n_cycles={n_cycles}'

            # https://github.com/mne-tools/mne-python/issues/9868
            # https://mne.tools/stable/auto_tutorials/time-freq/10_spectrum_class.html
        
            ax = axes[per_i]; per_i = per_i + 1
            for c in range(len(ch)):
                ax.plot(freqs, psds[c], label=f'{ch[c]}', linewidth=1)

            if 'Frequency_Epoch' in plots:
                for c in range(len(ch)):
                    fig2, ax2 = plt.subplots(figsize=(12,9))
                    for ep in range(len(epochs_good)):
                        ax2.plot(freqs, psds_epoch[ep][c], label=f'{round(p_start + (e_idx[ep]-min_e)*overlap_factor*epoch_time)} - {p_start + round((e_idx[ep]-min_e)*overlap_factor*epoch_time + epoch_time)} - [{e_idx[ep]},{e_idx[ep]-min_e}]', linewidth=1)
                    ax2.legend()
                    ax2.set(title=f'#{key} {p_label}-{p_index+1} {ch[c]}', xlabel='Frequency (Hz)', ylabel=plot_unit)
            ax.legend()
            ax.set(title=f'#{key} {p_label}-{p_index+1}', xlabel='Frequency (Hz)', ylabel=plot_unit)
    fig.suptitle(f'{plot_type} ({pipeline}, {sig_specs}, {freq_method}=[{plot_params}]')
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
        raw = raws[index].copy()
        ch = raw.ch_names
        
        b_window = 30; b_step = 5
        fig, axes = plt.subplots(round(len(ch)/2+.5),2, figsize=(14,len(ch)*2))
        fig.suptitle(f'#{key} Band Power by channel  ({pipeline})')
        axes = axes.flatten()
        b_max = 35
        x_ts = None
        bp_a = [[] for _ in range(len(bp_bands_simple))]
        t, eeg_2d = yasa.sliding_window(raw.get_data(units='uV')[c], raw.info["sfreq"], window=b_window, step=b_step)
        bp = yasa.bandpower(eeg_2d, raw.info["sfreq"], bands=bp_bands_simple, bandpass=False, relative=False)

        for c in range(len(ch)):

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
                ax.set(title=f'{ch[c]}-{sessions[key]["to_ref"]}', xlabel='Time', ylabel=units['p_dB'])
                
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
        fig.suptitle(f'#{key} Band Power ({pipeline})')
    
        for b in range(len(bp_bands_simple)):
            bl = f'{bp_bands_simple[b][2]} ({bp_bands_simple[b][0]} - {bp_bands_simple[b][1]} Hz)'
            ax.plot(x_ts, bp_mean[b], linewidth=1, label=bl)
            ax.fill_between(x_ts, bp_mean[b] - bp_std[b], bp_mean[b] + bp_std[b], alpha=0.15, label=bl)
            ax.set(xlabel='Time',ylabel=units['p_dB'])
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

