import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import mne
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import yaml
import yasa

from datetime import datetime, timedelta
from time import sleep

# config, if relative path not working then use explicit path to working dir (repo dir with scripts and yml) or modify working directory in IDE/GUI settings
# working_dir = '/path/to/openbci-session'
working_dir = os.getcwd()

# rename sleep_analysis.yml.sample to sleep_analysis.yml and set directories
cfg_file = os.path.join(working_dir, "sleep_analysis.yml")
with open(cfg_file, "r") as yamlfile:
    cfg_base = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg = cfg_base['default']

# git pull https://github.com/preraulab/multitaper_toolbox/
os.chdir(cfg['multitaper_dir'])
from multitaper_spectrogram_python import multitaper_spectrogram, nanpow2db

# import custom sleep functions
os.chdir(working_dir)
# https://github.com/roflecopter/qskit # install qskit if you want to process ECG to HR/HRV
from sleep_functions import acc_process, plot_radar, psd_plot, plot_multitaper_spect_ch, plot_multitaper_spect_all, create_spect, topomap_plot, process_bp, process_ecg, process_hypno, raw_preprocess, electrode_side, m2h, plot_average, plot_rolling_spindle_density, plot_hypnogram, sleep_stats, spindle_metrics, spindles_slow_fast, sws_metrics

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('future.no_silent_downcasting', True)

nj = cfg['n_jobs'] # n_jobs for multiprocessing, usually n_cpu - 1

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})

# used for cache/image file names
device = cfg['device']
user = cfg['user']
sleeps = cfg_base['sleeps']
if False: sleeps = {1: sleeps[1]}
settings = cfg_base['settings']

# ecg_invert: flips ecg signal, in case electrodes were placed inverse by mistake
# re_ref: set to True if you have recorded EEG with a single Ref channel and want to re-reference each channel to opposite hemisphere refs, e.g. F7-T3,F8-T3 will be changed to F7-T4,F8-T3. Always set to False if you have multiple refs
# sp_ch - channels used for spindle density analysis. same for sw_ch
# sp_slow_ch & sp_fast_ch - channels used for slow vs fast spindles analysis

# overwrite image files if exists (HRV, Hypno in PNG)
image_overwrite = settings['image_overwrite']

# signal filtering
bpf = settings['bpf'] # band pass filter, [0.1, None] or [.35, 45]
nf = settings['nf'] # notch filter, set to 50 or 60 Hz powerline noise freq depending on your country
eog_bpf = settings['eog_bpf']; emg_bpf = settings['emg_bpf'] # filter for EOG data
sf_to = settings['sf_to'] # sampling rate to resample for fast processing

plots = settings['plots']
smooth_arousal = settings['smooth_arousal'] # set True to smooth hypno by replace single awake epochs with previous epoch stage

# Channel types naming, everything not included threated as EEG. 
# Put unused channels to misc_ch
# Append ecg_ch if you have ECG channel with custom name
misc_ch = settings['misc_ch']; acc_ch = settings['acc_ch']
eog_ch = settings['eog_ch']; emg_ch = settings['emg_ch']
ecg_ch = settings['ecg_ch']; n_acc = settings['n_acc'] # number accelerometer channels, 3 for OpenBCI

# methods for calculating band power and topoplots
freq_method = settings['freq_method'] # 'mne_psd_welch' / 'mne_trf_morlet' / 'mne_psd_multitaper' / 'mne_tfr_multitaper'
topo_method = settings['topo_method'] # 'yasa_band_power' / 'mne_trf_morlet' / 'mne_fft_welch'
topo_ref = settings['topo_ref'] # 'AR' rereference type, 'REST' is not supported yet

w_fft = settings['w_fft']; m_bandwidth = settings['m_bandwidth']
m_freq_bandwidth = settings['m_freq_bandwidth']; tfr_time_bandwidth = settings['tfr_time_bandwidth']
bp_relative = settings['bp_relative'] # bandpass is relative or abs for topomap

# multitaper spectrograms settings, can leave as is if not sure what is it
spect_vlim = settings['spect_vlim']; spect_lim = settings['spect_lim'] 
freq_lim = settings['freq_lim']; time_bandwidth = settings['time_bandwidth'] # Set time-half bandwidth
num_tapers = time_bandwidth*2 - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = settings['window_params']  # Window size is Xs with step size of Ys

# units for labels
units = {'psd_dB': 'dB(µV²/Hz)', 'amp': 'µV', 'p': 'µV²', 'p_dB': 'dB(µV²)', 'rel': '%'}
sig_specs = f'sf={sf_to}Hz, notch={nf}, bandpass={bpf}'
spect_specs = f'num_tapers={num_tapers}, window={window_params}'

stages = {'W': 0,'N1': 1,'N2': 2,'N3': 3,'R': 4}
stages_plot = [0,4,2,3]

# list of bands for band power
bp_bands = [
    (1, 4, "Delta"),
    (4, 6, "Theta 1"),
    (6, 8, "Theta 2"),
    (8, 10, "Alpha 1"),
    (10, 12, "Alpha 2"),
    (12, 30, "Beta"),
    (30, 48, "Gamma"),
]

bp_bands_dict = dict()
for b in range(len(bp_bands)):
    bp_bands_dict[bp_bands[b][2]] = (bp_bands[b][0], bp_bands[b][1])

# set everything to True for full pipeline execution
load_spect = True
load_data = True
load_sp_sw = True
load_ecg = True
load_hypno = True
load_bp = True

# Pre-loading BDFs: read and format channel names, resample, filter, reoder, make raws array
if load_data:
    sessions = []
    #raws = []; refs = []; refs_ch = []; sessions =[]; accs = []; eegs = []; ecgs = []; eogs = []; miscs = []; raws_ori = []; dts = []
    for index, key in enumerate(sleeps):
        if sleeps[key]['file'].endswith('edf') or sleeps[key]['file'].endswith('EDF'):
            raw = mne.io.read_raw_edf(os.path.join(cfg['data_dir'], sleeps[key]['file']), preload=True, verbose=True)
        else:
            raw = mne.io.read_raw_bdf(os.path.join(cfg['data_dir'], sleeps[key]['file']), preload=True, verbose=True)
        
        sleeps[key]['dts'] = raw.info['meas_date']
        sleeps[key]['raw'], sleeps[key]['raw_ori'], sleeps[key]['eeg'], sleeps[key]['ref'], sleeps[key]['ref_ch'], sleeps[key]['acc'], sleeps[key]['ecg'], sleeps[key]['eog'], sleeps[key]['emg'], sleeps[key]['misc'] = raw_preprocess(raw, eog_ch, emg_ch, ecg_ch, acc_ch, misc_ch, sleeps[key]['re_ref'], nf, bpf, emg_bpf, eog_bpf, sf_to, nj=nj)

        if 'Spectrum_YASA' in plots:
            for spect_ch in raw.ch_names:
                if not (sleeps[key]['re_ref'] and sleeps[key]['ref'] == spect_ch):
                    sig = raw.get_data(spect_ch)
                    png_file = f"{raw.info['meas_date'].strftime(cfg['file_dt_format'])} yasa spect {spect_ch}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
                    fig = yasa.plot_spectrogram(sig[0], raw.info['sfreq'], None, trimperc=2.5)
                    plt.title(f'#{raw.info["meas_date"]} {spect_ch} YASA Spectrogram (processed raw)')
                    if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
        sessions.append(sleeps[key])
        
for index, session in enumerate(sessions):
    raw  = session['raw'].copy()

    if len(session['acc']) == n_acc:
        session['acc_agg'] = acc_process(raw, session['acc'], session['dts'])
    else:
        session['acc_agg'] = None

    raw.pick(session['eeg'])
    if load_hypno:
        session['hypnos'] = []; session['probs'] = []
        for ch_index, ch in enumerate(session['eeg']):
            sls = yasa.SleepStaging(raw, eeg_name=ch)
            prob = sls.predict_proba()
            session['probs'].append(prob)
            session['hypnos'].append(sls.predict())
        session['hypnos_max'], session['hypnos_adj'] = process_hypno(raw, session['probs'], smooth_arousal=True)
        
        # Save CSV file for possible import into EDFBrowser
        # Sleep Staging > How do I edit the predicted hypnogram in https://raphaelvallat.com/yasa/faq.html#sleep-staging  for more details
        hyp_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} probs_adj_consensus.csv"; hyp_filename = os.path.join(cfg['cache_dir'], hyp_file)
        hypno_export = pd.DataFrame({"onset": np.arange(len(session['hypnos_adj'])) * 30, "label": yasa.hypno_int_to_str(session['hypnos_adj']), "duration": 30})
        hypno_export.to_csv(hyp_filename, index=False)
        
        # make hypno array for future merge during ECG processing
        session['hypno_df'] = pd.DataFrame({'h': session['hypnos_adj'], 'dt': [session['dts'] + timedelta(seconds=30*(i+1)) for i in range(len(session['hypnos_adj']))]})
        session['hypno_df']['dtr'] = session['hypno_df']['dt'].dt.round('30s')
        session['hypno_df']['cumtime'] = (session['hypno_df']['dt']-session['dts']).dt.total_seconds()
        session['sleep_stats_info'] = sleep_stats(yasa.hypno_int_to_str(session['hypnos_adj']))

    if load_sp_sw:
        sp = yasa.spindles_detect(raw, include=(2), hypno=yasa.hypno_upsample_to_data(session['hypnos_adj'], sf_hypno=1/30, data=raw))
        if not (1 == np.sum(np.isin(session['sp_ch'], session['eeg']))/len(session['sp_ch'])):
            sp_ch = [session['eeg'][0]]
        sp_metric = spindle_metrics(sp, session['sleep_stats_info']['SOL_ADJ'], session['hypnos_adj'], sp_ch=session['sp_ch'], stages=[2], period=4.5*3600)

        sw = yasa.sw_detect(raw, include=(2,3), hypno=yasa.hypno_upsample_to_data(session['hypnos_adj'], sf_hypno=1/30, data=raw))                
        if not (1 == np.sum(np.isin(session['sw_ch'], session['eeg']))/len(session['sw_ch'])):
            sw_ch = [session['eeg'][0]]
        sw_metric = sws_metrics(sw, session['sleep_stats_info']['SOL_ADJ'], session['hypnos_adj'], sw_ch=session['sw_ch'], stages=[2,3], period=4.5*3600)
        
        session['sp'] = sp
        session['sp_metric'] = sp_metric
        session['sw'] = sw
        session['sw_metric'] = sw_metric
    sessions[index] = session

if 'Hypno' in plots:
    for index, session in enumerate(sessions):
        raw  = session['raw'].copy().pick(session['eeg'])

        fig, axes = plt.subplots(round(len(session['eeg']))+2, 
                  figsize=(8, 4+len(session['eeg'])*2))
        fig.suptitle(f"{session['dts'].strftime(cfg['plot_dt_format'])} Hypnograms, {spect_specs}")
        hyp = yasa.hypno_int_to_str(session['hypnos_max']); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = axes[0])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Max Probs)')
        hyp = yasa.hypno_int_to_str(session['hypnos_adj']); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = axes[1])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Adj Probs)')
        for ch_index, ch in enumerate(session['eeg']):
            hyp = session['hypnos'][ch_index]; hyp_stats = sleep_stats(hyp)
            ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = axes[ch_index+2])
            ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} ({ch})')

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} hypno channels {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
            
        fig, ax = plt.subplots(figsize=(5, 2))
        hyp = yasa.hypno_int_to_str(session['hypnos_adj']); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = ax, hl_lw=5)
        ax.set_title(f'{session["dts"].strftime(cfg["plot_dt_format"])} Hypno Adj Probs\n{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} hypno {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'HRV' in plots: # Process ECG
    if load_ecg:
        for index, session in enumerate(sessions):
            if len(session['ecg']) > 0:
                raw  = session['raw'].copy().pick(session['ecg'])
                ecg_invert = -1 if session['ecg_invert'] else 1
                fig, session['ecg_stats'], hrv_stages, major_acc_epoch = process_ecg(raw, session['ecg'], session['dts'], session['hypno_df'], session['acc_agg'], session['acc'], session['sleep_stats_info'], cfg, user, device, ecg_invert)
                png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} hrv {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
                sessions[index] = session

if load_bp:
    for index, session in enumerate(sessions):  
        session['raw_bp'], session['bps_s'], session['bp_stages'] = process_bp(session['raw'], session['eeg'], session['ref'], topo_ref, session['hypnos_adj'], stages, session['re_ref'], bp_bands, bp_relative)
        sessions[index] = session

if 'Topomap' in plots:
    for index, session in enumerate(sessions):
        if len(session['eeg']) > 2:
            fig = topomap_plot(session['dts'], session['raw_bp'], session['bps_s'], bp_relative, topo_ref, sig_specs, topo_method, session['hypnos_adj'], stages, stages_plot, session['bp_stages'], bp_bands, units, cfg)
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} topomap {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'Spectrum' in plots:
    # Multitaper spectrogram from Prerau Labs Multitaper Toolbox
    # https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    frequency_range = [spect_lim[0], spect_lim[1]]  # Limit frequencies from 0 to 25 Hz
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = False  # use multiprocessing
    cpus = nj # use 3 cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = False  # plot spectrogram
    return_fig = False # do not return plotted spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = False  # print extra info
    xyflip = False  # do not transpose spect output matrix
    
    if load_spect:
        for index, session in enumerate(sessions):              
            session['spects'], session['stimes'], session['sfreqs'] = create_spect(session['raw'], session['eeg'], multitaper_spectrogram, nanpow2db, spect_lim, frequency_range, time_bandwidth, num_tapers, 
                window_params, min_nfft, detrend_opt, multiprocess, cpus,
                weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
            sessions[index] = session

    for index, session in enumerate(sessions): 
        fig = plot_multitaper_spect_all(session['raw'], session['dts'], session['eeg'], session['spects'], session['stimes'], session['sfreqs'], session['hypno_df'], spect_specs, cfg, nanpow2db, spect_vlim, clim_scale, sig_specs)
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} merged spectrum {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

        figs = plot_multitaper_spect_ch(session['raw'], session['dts'], session['eeg'], session['ref_ch'], session['spects'], session['stimes'], session['sfreqs'], session['hypno_df'], spect_specs, cfg, nanpow2db, spect_vlim, clim_scale, sig_specs)
        for cy, fig in enumerate(figs):                
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} spect {cy} {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'Spindles' in plots:
    for index, session in enumerate(sessions):
        if (1 == np.sum(np.isin(session['sp_ch'], session['eeg']))/len(session['sp_ch'])):
            fig = plot_rolling_spindle_density(session['sp'].summary(), session['dts'], cfg, channels=session['sp_ch'], window_minutes=10,stage_filter=[2],type_label='spindles', verbose=False)
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} SPD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
    
if 'SlowWaves' in plots:
    for index, session in enumerate(sessions):
        if (1 == np.sum(np.isin(session['sw_ch'], session['eeg']))/len(session['sw_ch'])):
            fig = plot_rolling_spindle_density(session['sw'].summary(), session['dts'], cfg, channels=session['sw_ch'], window_minutes=10,stage_filter=[2,3],type_label='slow wave',verbose=False)    
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} SWD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'SpindlesFreq':
    for index, session in enumerate(sessions):
        if (1 == np.sum(np.isin(session['sp_slow_ch'], session['eeg']))/len(session['sp_slow_ch'])) and (1 == np.sum(np.isin(session['sp_fast_ch'], session['eeg']))/len(session['sp_fast_ch'])):
            fig = spindles_slow_fast(session['sp'].summary(), yasa.hypno_upsample_to_data(session['hypnos_adj'], sf_hypno=1/30, data=session['raw']), session['raw'], session['eeg'], slow_ch=session['sp_slow_ch'], fast_ch=session['sp_fast_ch'])
            if fig is not None:
                png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} Spindles Freqs {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

# Power frequency plot with mne PSD computation
if 'Features' in plots:
    for index, session in enumerate(sessions):
        fig, amp, max_amp = psd_plot(session['dts'], session['raw_ori'], session['eeg'], session['ref_ch'], session['sp'], session['sp_ch'], session['sp_metric'], session['sw'], session['sw_ch'], session['sw_metric'], freq_method, w_fft, freq_lim, units, sig_specs, cfg, nj)
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} PSD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'Radar' in plots:
    for index, session in enumerate(sessions):
        if 'ecg_stats' in session.keys():
            fig = plot_radar(session['dts'], session['sleep_stats_info'], session['ecg_stats'], session['acc'], cfg, n3_goal = 90, rem_goal = 105, awk_goal = 30, hr_goal = 45, hrv_goal = 35, mh_goal = 4.3)
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} Radar {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
