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
from sleep_functions import acc_process, plot_multitaper_spect_ch, plot_multitaper_spect_all, create_spect, topomap_plot, process_bp, process_ecg, process_hypno, raw_preprocess, electrode_side, m2h, plot_average, plot_rolling_spindle_density, plot_hypnogram, sleep_stats, spindle_metrics, spindles_slow_fast, sws_metrics

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('future.no_silent_downcasting', True)

nj = 15 # n_jobs for multiprocessing, usually n_cpu - 1

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})

# used for cache/image file names
device = 'openbci'
user = 'user'

# make array with bdf files recorded with session_start.py
f_name = os.path.join(cfg['data_dir'], 'yasa_example_night_young.edf')
sleeps = {'1': {'file': f_name, 'ecg_invert': False}} # ecg_invert flips ecg signal, in case electrodes were placed inverse by mistake

# overwrite image files if exists (HRV, Hypno in PNG)
image_overwrite = True

# signal filtering
bpf = [.35, 45] # band pass filter, [0.1, None] or [.35, 45]
nf = [50,1] # notch filter, set to 50 or 60 Hz powerline noise freq depending on your country
eog_bpf = [.5,8]; emg_bpf = [10,70] # filter for EOG data
sf_to = 256 # sampling rate to resample for fast processing

plots = ['Hypno', 'HRV', 'Features','Spectrum_YASA','Spectrum','Topomap', 'Spindles', 'SlowWaves', 'SpindlesFreq', 'Radar'] # to plot all use: plots = ['Hypno', 'HRV', 'Features','Spectrum','Topomap']
#plots = ['Hypno','HRV','Features','Topomap', 'Spindles', 'SlowWaves', 'SpindlesFreq']
smooth_arousal = True # set True to smooth hypno by replace single awake epochs with previous epoch stage

# Channel types naming, everything not included threated as EEG. 
# Put unused channels to misc_ch
# Append ecg_ch if you have ECG channel with custom name
misc_ch = ['E1-Fpz', 'E2-Fpz']; acc_ch = ['ACC_X', 'ACC_Y', 'ACC_Z']
eog_ch = ['EOG-RL', 'ROC-A1','LOC-A2']; emg_ch = ['EMG-N','EMG1-EMG2']; ecg_ch = ['ECG', 'ECG-AS', 'ECG-AI', 'ECG-RA-V2','EKG-R-EKG-L']
n_acc = 3 # number accelerometer channels, 3 for OpenBCI
re_ref = False # set to True if you have recorded EEG with a single Ref channel and want to re-reference each channel to opposite hemisphere refs, e.g. F7-T3,F8-T3 will be changed to F7-T4,F8-T3. Always set to False if you have multiple refs

# methods for calculating band power and topoplots
freq_method = 'mne_psd_welch' # 'mne_psd_welch' / 'mne_trf_morlet' / 'mne_psd_multitaper' / 'mne_tfr_multitaper'
topo_method = 'yasa_band_amp' # 'yasa_band_power' / 'mne_trf_morlet' / 'mne_fft_welch'
w_fft = 4; m_bandwidth = 1; m_freq_bandwidth = 2; tfr_time_bandwidth = 4; 
topo_ref = 'AR' # 'REST' / 'AR' rereference type
bp_relative = True # bandpass is relative or abs for topomap

# sws & spindles extended metrics
sp_ch = ['C3-A2','C4-A1']
sp_slow_ch = ['C3-A2','C4-A1','CZ-A2']
sp_fast_ch = ['P3-A2','P4-A1','PZ-A2']
sw_ch = ['F3-A2','F4-A1']

# multitaper spectrograms settings, can leave as is if not sure what is it
spect_vlim = [6,24]; spect_lim = [1,16]; freq_lim = [1,30]
time_bandwidth = 24 # Set time-half bandwidth
num_tapers = time_bandwidth*2 - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = [60, 30]  # Window size is Xs with step size of Ys

# units for labels
units = {'psd_dB': 'dB(µV²/Hz)', 'amp': 'µV', 'p': 'µV²', 'p_dB': 'dB(µV²)', 'rel': '%'}
sig_specs = f'sf={sf_to}Hz, notch={nf}, bandpass={bpf}'
spect_specs = f'num_tapers={num_tapers}, window={window_params}'

stages = ['W','N1','N2','N3','R']
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
if load_data or not ('raws' in globals() or 'raws' in locals()):
    raws = []; refs = []; refs_ch = []; accs = []; eegs = []; ecgs = []; eogs = []; miscs = [];  
    raws_ori = []; dts = []
    for index, key in enumerate(sleeps):
        if sleeps[key]['file'].endswith('edf') or sleeps[key]['file'].endswith('EDF'):
            raw = mne.io.read_raw_edf(sleeps[key]['file'], preload=True, verbose=True)
        else:
            raw = mne.io.read_raw_bdf(sleeps[key]['file'], preload=True, verbose=True)
        dts.append(raw.info['meas_date'])
                        
        raw_processed, raw_ori, eeg_ch_names, ref, ref_ch, acc, ecg, eog, emg, misc = raw_preprocess(raw, eog_ch, emg_ch, ecg_ch, acc_ch, misc_ch, re_ref, nf, bpf, emg_bpf, eog_bpf, sf_to, nj=nj)
        
        if 'Spectrum_YASA' in plots:
            for spect_ch in raw.ch_names:
                sig = raw.get_data(spect_ch)
                png_file = f"{raw.info['meas_date'].strftime(cfg['file_dt_format'])} yasa spect {spect_ch}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
                fig = yasa.plot_spectrogram(sig[0], raw.info['sfreq'], None, trimperc=2.5)
                plt.title(f'#{raw.info["meas_date"]} {spect_ch} YASA Spectrogram (processed raw)')
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
        
        raws_ori.append(raw_ori) # original raw list
        raws.append(raw_processed) # raw list with edited channel names for each sleep session     
        accs.append(acc) # accelerometer channel list for each sleep session     
        ecgs.append(ecg) # ecg channel list for each sleep session     
        eogs.append(eog) # eog channel list for each sleep session     
        eegs.append(eeg_ch_names)
        miscs.append(misc) # other  channel list for each sleep session           
        refs.append(ref) # re-reference channel for each sleep session appears only when re_ref is True
        refs_ch.append(ref_ch) # reference list for each channel for each sleep session

if load_hypno or not ('hypnos' in globals() or 'hypnos' in locals()):
    hypnos = []; probs = []; hypnos_max = []; hypnos_adj = []
    hypno_dfs = []; sleep_stats_infos = []; acc_aggs = []
if load_sp_sw or not ('sps' in globals() or 'sps' in locals() or 'sws' in globals() or 'sws' in locals()):
    sps = []; sws =[]; sp_metrics=[]; sw_metrics=[]

for index, raw in enumerate(raws):
    # Process accelerometer
    acc_agg = None
    if len(acc) == n_acc:
        # Process accelerometer
        acc_agg = acc_process(raw, acc, dts[0])
        acc_aggs.append(acc_agg)
    
    if load_hypno or (len(hypnos) < 1):
        raw  = raws[index].copy().pick(eegs[index])
        hypnos_up_ch = []; hypnos_ch = []; probs_ch = []
        for ch_index, ch in enumerate(eegs[index]):
            sls = yasa.SleepStaging(raw, eeg_name=ch)
            prob = sls.predict_proba()
            probs_ch.append(prob)
            hypno_pred = sls.predict()  # Predict the sleep stages
            hypnos_ch.append(hypno_pred)
            hypno_pred_ch = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
        probs.append(probs_ch)
        hypnos.append(hypnos_ch)

        probs_consensus, probs_adj_consensus = process_hypno(raw, probs_ch, smooth_arousal=True)
        hypnos_max.append(probs_consensus)
        hypnos_adj.append(probs_adj_consensus)
        
        # Save CSV file for possible import into EDFBrowser
        # Sleep Staging > How do I edit the predicted hypnogram in https://raphaelvallat.com/yasa/faq.html#sleep-staging  for more details
        hyp_file = f"{dts[index].strftime(cfg['file_dt_format'])} {user} probs_adj_consensus.csv"; hyp_filename = os.path.join(cfg['cache_dir'], hyp_file)
        hypno_export = pd.DataFrame({"onset": np.arange(len(probs_adj_consensus)) * 30, "label": yasa.hypno_int_to_str(probs_adj_consensus), "duration": 30})
        hypno_export.to_csv(hyp_filename, index=False)
        
        # make hypno array for future merge during ECG processing
        hypno_df = pd.DataFrame({'h': probs_adj_consensus, 'dt': [dts[index] + timedelta(seconds=30*(i+1)) for i in range(len(probs_adj_consensus))]})
        hypno_df['dtr'] = hypno_df['dt'].dt.round('30s')
        hypno_df['cumtime'] = (hypno_df['dt']-dts[0]).dt.total_seconds()
        hypno_dfs.append(hypno_df)
        
        sleep_stats_info = sleep_stats(yasa.hypno_int_to_str(probs_adj_consensus))
        sleep_stats_infos.append(sleep_stats_info)

    if load_sp_sw or (len(sps) < 1):
        sp = yasa.spindles_detect(raw, include=(2), hypno=yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raw_processed))
        if not (1 == np.sum(np.isin(sp_ch, eegs[index]))/len(sp_ch)):
            sp_ch = [eegs[index][0]]
        
        sp_metric = spindle_metrics(sp, sleep_stats_infos[index]['SOL_ADJ'], hypnos_adj[index], sp_ch=sp_ch, stages=[2], period=4.5*3600)
        sp_metrics.append(sp_metric)

        sw = yasa.sw_detect(raw, include=(2,3), hypno=yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raws[index]))
                
        if not (1 == np.sum(np.isin(sw_ch, eegs[index]))/len(sw_ch)):
            sw_ch = [eegs[index][0]]
        sw_metric = sws_metrics(sw, sleep_stats_infos[index]['SOL_ADJ'], hypnos_adj[index], sw_ch=sw_ch, stages=[2,3], period=4.5*3600)
        
        sws.append(sw)
        sps.append(sp)
        sw_metrics.append(sw_metric)
        
if 'Hypno' in plots:
    for index, raw in enumerate(raws):
        raw  = raws[index].copy().pick(eegs[index])

        fig, axes = plt.subplots(round(len(eegs[index]))+2, 
                  figsize=(8, 4+len(eegs[index])*2))
        fig.suptitle(f'#{raw.info["meas_date"]} Multitaper spectrogram, {spect_specs}')
        hyp = yasa.hypno_int_to_str(hypnos_max[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[0])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Max Probs)')
        hyp = yasa.hypno_int_to_str(hypnos_adj[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[1])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Adj Probs)')
        for ch_index, ch in enumerate(eegs[index]):
            hyp = hypnos[index][ch_index]; hyp_stats = sleep_stats(hyp)
            ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[ch_index+2])
            ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} ({ch})')

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{dts[index].strftime(cfg['file_dt_format'])} hypno channels {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
            
        fig, ax = plt.subplots(figsize=(5, 2))
        hyp = yasa.hypno_int_to_str(hypnos_adj[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = ax, hl_lw=5)
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Adj Probs)')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{dts[index].strftime(cfg['file_dt_format'])} hypno {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

ecg_stats = []
if 'HRV' in plots: # Process ECG
    if load_ecg or not ('load_ecg' in globals() or 'load_ecg' in locals()):
        for index, raw in enumerate(raws):
            if len(ecgs[index]) > 0:
                raw  = raws[index].copy().pick(ecgs[index])
                ecg_invert = -1 if sleeps[str(index+1)]['ecg_invert'] else 1
                fig, ecg_stat = process_ecg(raw, ecgs[index], dts[index], hypno_dfs[index], acc_aggs[index], accs[index], sleep_stats_infos[index], 
                                             cfg, user, device, ecg_invert)
                ecg_stats.append(ecg_stat)
                png_file = f"{dts[index].strftime(cfg['file_dt_format'])} hrv {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if load_bp or not ('bps' in globals() or 'bps' in locals()):
    bps = []
    raws_bp = []
    for index, raw in enumerate(raws):  
        raw_bp, bps_s = process_bp(raws[index], eegs[index], refs[index], topo_ref, hypnos_adj[index], stages, re_ref, bp_bands, bp_relative)
        raws_bp.append(raw_bp)
        bps.append(bps_s)

if 'Topomap' in plots:
    for index, raw in enumerate(raws):
        if len(eegs[index]) > 2:
            fig = topomap_plot(raw_bp, bps[index], bp_relative, topo_ref, sig_specs, topo_method, hypnos_adj[index], stages, stages_plot, bp_bands, units)
            png_file = f"{dts[index].strftime(cfg['file_dt_format'])} topomap {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
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
    
    if load_spect or not ('spects_l' in globals() or 'spects_l' in locals()):
        spects_l = []; stimes_l = []; sfreqs_l = []
        for index, raw in enumerate(raws):                
            spects_c, stimes_c, sfreqs_c = create_spect(raws[index], eegs[index], multitaper_spectrogram, nanpow2db, spect_lim, frequency_range, time_bandwidth, num_tapers, 
                window_params, min_nfft, detrend_opt, multiprocess, cpus,
                weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
            spects_l.append(spects_c)
            stimes_l.append(stimes_c)
            sfreqs_l.append(sfreqs_c)

    for index, raw in enumerate(raws): 
            fig = plot_multitaper_spect_all(raws[index], dts[index], eegs[index], spects_l[index], stimes_l[index], sfreqs_l[index], hypno_dfs[index], spect_specs, cfg, nanpow2db, spect_vlim, clim_scale, sig_specs)
            png_file = f"{dts[index].strftime(cfg['file_dt_format'])} merged spectrum {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

            figs = plot_multitaper_spect_ch(raws[index], dts[index], eegs[index], refs_ch[index], spects_l[index], stimes_l[index], sfreqs_l[index], hypno_dfs[index], spect_specs, cfg, nanpow2db, spect_vlim, clim_scale, sig_specs)
            for cy, fig in enumerate(figs):                
                png_file = f"{dts[index].strftime(cfg['file_dt_format'])} spect {cy} {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'Spindles' in plots:
    for index, raw in enumerate(raws):
        fig = plot_rolling_spindle_density(
            sps[index].summary(), 
            dts[index], 
            channels=sp_ch, 
            window_minutes=10,
            stage_filter=[2],
            type_label='spindles',
            verbose=False
        )
        png_file = f"{dts[index].strftime(cfg['file_dt_format'])} SPD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
    
if 'SlowWaves' in plots:
    for index, raw in enumerate(raws):
        fig = plot_rolling_spindle_density(
            sws[index].summary(), 
            dts[index], 
            channels=sw_ch, 
            window_minutes=10,
            stage_filter=[2,3],
            type_label='slow wave',
            verbose=False
        )    
        png_file = f"{dts[index].strftime(cfg['file_dt_format'])} SWD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'SpindlesFreq':
    for index, raw in enumerate(raws):
        if (1 == np.sum(np.isin(sp_slow_ch, eegs[index]))/len(sp_slow_ch)) and (1 == np.sum(np.isin(sp_fast_ch, eegs[index]))/len(sp_fast_ch)):
            hypno = yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raw)
            fig = spindles_slow_fast(sps[index].summary(), hypno, raws[index], eegs[index], slow_ch=sp_slow_ch, fast_ch=sp_fast_ch)
            png_file = f"{dts[index].strftime(cfg['file_dt_format'])} Spindles Freqs {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)


# Power frequency plot with mne PSD computation
if 'Features' in plots:
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.flatten()
    for index, key in enumerate(raws_ori):
        raw  = raws[index].copy().pick(eegs[index])
        if freq_method == 'mne_psd_welch':
            n_fft = int(w_fft * raw.info['sfreq'])
            psds, freqs = mne.time_frequency.psd_array_welch(
                raw.get_data(units='uV'), raw.info['sfreq'],
                fmin=freq_lim[0], fmax=freq_lim[1],
                n_fft= n_fft, output = 'power', n_jobs = nj)
            psds = 10 * np.log10(psds) # convert to dB
            plot_unit = units['psd_dB']; plot_type = 'PSD'
            plot_params = f'fft_window={w_fft}s'
        for c in range(len(eeg_ch_names)):
            ax[3].plot(freqs, psds[c], label=f'{eegs[index][c]}-{refs_ch[index][eegs[index][c]]}', linewidth=1)
        ax[3].legend()
        ax[3].set(title=f'', xlabel='Frequency (Hz)', ylabel=plot_unit)
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 20})
    fig.suptitle(f'{plot_type} ({sig_specs}, {topo_method}=[{plot_params}]')
    plt.tight_layout()
    
    for index, raw in enumerate(raws):
        plot_average(sps[index], "spindles", ax=ax[2], legend=False)
        if len(sp_metrics) >= index:
            sp_metric = sp_metrics[index]
            ax[2].set_title(f'Density: {round(sp_metric[0],2)} CV: {round(sp_metric[3],2)} {sp_ch}\n Early {round(sp_metric[1],2)} Late {round(sp_metric[2],2)} E/L {round(sp_metric[1]/sp_metric[2],2)}')        

    for index, raw in enumerate(raws):
        axe = plot_average(sws[index], 'sw', center='PosPeak', ax=ax[0], legend=False);
        amps = round(sws[index].summary(grp_chan=True)[['Count','PTP']]).reset_index()
        max_amp = amps['Count'].argmax()
        axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{refs_ch[index][amps["Channel"][max_amp]]}')

        axe = plot_average(sws[index], 'sw', center='PosPeak', hue="Stage", ax=ax[1], legend=True)
        amps = round(sws[index].summary(grp_stage=True, grp_chan=True)[['Count','PTP']]).reset_index()
        max_amp = amps['Count'].argmax()
        axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{refs_ch[index][amps["Channel"][max_amp]]} in N{amps["Stage"][max_amp]}')
        if len(sw_metrics) >= index:
            sw_metric = sw_metrics[index]
            axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{refs_ch[index][amps["Channel"][max_amp]]} in N{amps["Stage"][max_amp]}\nCV: {round(sw_metric[0],2)} Early {round(sw_metric[1],2)} Late {round(sw_metric[2],2)}, E/L {round(sw_metric[1]/sw_metric[2],2)} {sw_ch}')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    png_file = f"{dts[index].strftime(cfg['file_dt_format'])} PSD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
    if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
    plt.rcParams.update({"font.size": old_fontsize})

if 'Radar' in plots:
    for index, key in enumerate(raws_ori):
        if len(ecg_stats) >= index + 1:
            sleep_stats_info = sleep_stats_infos[index]
            ecg_stats_info = ecg_stats[index]
            n3 = sleep_stats_info['N3']; n3_goal = 90; n3_rng = 20
            rem = sleep_stats_info['REM']; rem_goal = 105; rem_rng = 30
            awk = sleep_stats_info['SOL_ADJ']+sleep_stats_info['WASO_ADJ']; awk_goal = 30; awk_rng = 20
            hr = ecg_stats_info['hr']; hr_goal = 45; hr_rng = 10
            hrv = ecg_stats_info['rmssd_n3']; hrv_goal = 35; hrv_rng = 25
            if len(acc) > 0: 
                mh = ecg_stats_info['mh']
            else:
                mh = 4.3
            mh_goal = 4.3; mh_rng = 2
            
            values = [
                round(100*(n3/n3_goal)),
                round(100*(rem/rem_goal)),
                round(100*(awk_goal/awk)),
                round(100*(hr_goal/hr)),
                round(100*(hrv/hrv_goal)),
                round(100*(mh_goal/mh)),
            ]
            
            n3_d = round(100*(n3/n3_goal) - 100)
            rem_d = round(100*(rem/rem_goal) - 100)
            awk_d = round(100*(awk_goal/awk) - 100)
            hr_d = round(100*(hr_goal/hr) - 100)
            hrv_d = round(100*(hrv/hrv_goal) - 100)
            mh_d = round(100*(mh_goal/mh) - 100)
            
            labels = [f'N3 {n3_d if n3_d < 0 else "+" + str(n3_d)}%', 
                      f'REM {rem_d if rem_d < 0 else "+" + str(rem_d)}%',
                      f'AWAKE {awk_d if awk_d < 0 else "+" + str(awk_d)}%',
                      f'HR {hr_d if hr_d < 0 else "+" + str(hr_d)}%',
                      f'HRV N3 {hrv_d if hrv_d < 0 else "+" + str(hrv_d)}%',
                      f'Move/h {mh_d if mh_d < 0 else "+" + str(mh_d)}%',
                      ]
    
            
            num_vars = len(labels)
            angles_ul = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
            angles = angles_ul.tolist()
            values += values[:1]
            angles += angles[:1]
            
            old_fontsize = plt.rcParams["font.size"]
            plt.rcParams.update({"font.size": 14})
            fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': 'polar'})
            ax.plot(np.linspace(0, 2 * np.pi, 100), [100] * 100, color='green', linewidth=2)
            ax.plot(angles, values, color='#1aaf6c', linewidth=1)
            ax.fill(angles, values, color='#1aaf6c', alpha=0.25)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles_ul), labels)
            for label, angle in zip(ax.get_xticklabels(), angles):
              if angle in (0, np.pi):
                label.set_horizontalalignment('center')
              elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
              else:
                label.set_horizontalalignment('right')
            ax.set_ylim(0, 120)
            ax.set_rlabel_position(180 / num_vars)
            ax.tick_params(colors='#222222')
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(color='#AAAAAA')
            ax.spines['polar'].set_color('#222222')
            ax.set_facecolor('#FAFAFA')
            plt.tight_layout()
            plt.rcParams.update({"font.size": old_fontsize})
            
            png_file = f"{dts[index].strftime(cfg['file_dt_format'])} Radar {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
            plt.rcParams.update({"font.size": old_fontsize})
