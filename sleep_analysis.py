import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import mne
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
import yaml
import yasa

from datetime import datetime, timedelta, time as dtime, timezone
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
from sleep_functions import acc_process, psds, summarize_cycles, smooth_hypno_custom, detect_sleep_cycles, plot_radar, psd_plot, plot_multitaper_spect_ch, plot_multitaper_spect_all, create_spect, topomap_plot, process_bp, process_ecg, process_hypno, raw_preprocess, electrode_side, m2h, plot_average, plot_rolling_spindle_density, plot_hypnogram, sleep_stats, spindle_metrics, spindles_slow_fast, sws_metrics, compute_rolling_rem_propensity, compute_cumulative_rem_phase

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('future.no_silent_downcasting', True)

nj = cfg['n_jobs'] # n_jobs for multiprocessing, usually n_cpu - 1

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})
debug = True

# used for cache/image file names
device = cfg['device']
user = cfg['user']
sleeps = cfg_base['sleeps']
if debug:
    sleeps = {
        1: sleeps[1], 
        2: sleeps[2], 
        3: sleeps[3], 
        4: sleeps[4], 
        5: sleeps[5], 
        6: sleeps[6],
        7: sleeps[7],
              } 
    # uncomment and choose specific sleep file, by default all files will be processed.
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
sf_hypno = 1/30

pa_sp_center = "Peak"
pa_sw_center = 'PosPeak'
pa_time_before = 1
pa_time_after = 1
pa_filt = (None,None)
pa_mask = None
plot_avg = not debug

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

        if 'rename' in sleeps[key].keys():
            if len(sleeps[key]['rename']) > 0:
                raw.rename_channels(sleeps[key]['rename'])
                print(raw.info['ch_names'])

        sleeps[key]['pipeline'] = f"{sf_to} {int(sleeps[key]['re_ref'])} {nf} {bpf} {num_tapers} {window_params} {w_fft} {m_bandwidth}"
        sleeps[key]['dts'] = raw.info['meas_date']
        sleeps[key]['raw'], sleeps[key]['raw_ori'], sleeps[key]['eeg'], sleeps[key]['ref'], sleeps[key]['ref_ch'], sleeps[key]['acc'], sleeps[key]['ecg'], sleeps[key]['eog'], sleeps[key]['emg'], sleeps[key]['misc'] = raw_preprocess(raw, eog_ch, emg_ch, ecg_ch, acc_ch, misc_ch, sleeps[key]['re_ref'], nf, bpf, emg_bpf, eog_bpf, sf_to, nj=nj)

        if 'Spectrum_YASA' in plots:
            for spect_ch in raw.ch_names:
                if not (sleeps[key]['re_ref'] and sleeps[key]['ref'] == spect_ch) and (spect_ch not in sleeps[key]['misc']+sleeps[key]['acc']):
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
        hyp_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} probs_adj_consensus {session['pipeline']}.csv"; hyp_filename = os.path.join(cfg['cache_dir'], hyp_file)
        hypno_export = pd.DataFrame({"onset": np.arange(len(session['hypnos_adj'])) * 30, "label": yasa.hypno_int_to_str(session['hypnos_adj']), "duration": 30})
        hypno_export.to_csv(hyp_filename, index=False)
        
        # make hypno array for future merge during ECG processing
        session['hypno_df'] = pd.DataFrame({'h': session['hypnos_adj'], 'dt': [session['dts'] + timedelta(seconds=30*(i+1)) for i in range(len(session['hypnos_adj']))]})
        session['hypno_df']['dtr'] = session['hypno_df']['dt'].dt.round('30s')
        session['hypno_df']['cumtime'] = (session['hypno_df']['dt']-session['dts']).dt.total_seconds()
        session['sleep_stats'] = sleep_stats(yasa.hypno_int_to_str(session['hypnos_adj']))
        session['dts_sol'] = session['dts'] + timedelta(minutes=session['sleep_stats']['SOL_ADJ'])
        session['dte'] = (session['dts_sol'] + timedelta(minutes=session['sleep_stats']['SPT']))
        session['midtime'] = session['dts_sol'] + timedelta(seconds=(session['dte'] - session['dts_sol']).total_seconds()/2)

    if load_sp_sw:
        sp_summary_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sp_summary {session['pipeline']}.parquet"; sp_summary_filename = os.path.join(cfg['cache_dir'], sp_summary_file)
        sp_sync_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sp_sync {session['pipeline']}.parquet"; sp_sync_filename = os.path.join(cfg['cache_dir'], sp_sync_file)
        sp_metric_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sp_metric {session['pipeline']}.pkl"; sp_metric_filename = os.path.join(cfg['cache_dir'], sp_metric_file)
        if os.path.exists(sp_summary_filename) and os.path.exists(sp_sync_filename) and os.path.exists(sp_metric_filename):
            session['sp_summary'] = pd.read_parquet(sp_summary_filename)
            session['sp_sync'] = pd.read_parquet(sp_sync_filename)
            with open(sp_metric_filename, "rb") as f:
                session['sp_metric'] = pickle.load(f)
        else:
            sp = yasa.spindles_detect(raw, include=(2,3), hypno=yasa.hypno_upsample_to_data(session['hypnos_adj'], sf_hypno=sf_hypno, data=raw))
            session['sp_summary'] = sp.summary()
            session['sp_sync'] = sp.get_sync_events(center=pa_sp_center, time_before=pa_time_before, time_after=pa_time_after, filt=pa_filt, mask=pa_mask)            
            if not (1 == np.sum(np.isin(session['sp_ch'], session['eeg']))/len(session['sp_ch'])):
                sp_ch = [session['eeg'][0]]
            session['sp_metric'] = spindle_metrics(sp, session['sleep_stats']['SOL_ADJ'], session['hypnos_adj'], sp_ch=session['sp_ch'], stages=[2], period=4.5*3600)

            session['sp_summary'].to_parquet(sp_summary_filename, compression="brotli")
            session['sp_sync'].to_parquet(sp_sync_filename, compression="brotli")
            with open(sp_metric_filename, "wb") as f:
                pickle.dump(session['sp_metric'], f)

        sw_summary_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sw_summary {session['pipeline']}.parquet"; sw_summary_filename = os.path.join(cfg['cache_dir'], sw_summary_file)
        sw_summary_ch_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sw_summary_ch {session['pipeline']}.parquet"; sw_summary_ch_filename = os.path.join(cfg['cache_dir'], sw_summary_ch_file)
        sw_summary_chst_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sw_summary_chst {session['pipeline']}.parquet"; sw_summary_chst_filename = os.path.join(cfg['cache_dir'], sw_summary_chst_file)
        sw_sync_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sw_sync {session['pipeline']}.parquet"; sw_sync_filename = os.path.join(cfg['cache_dir'], sw_sync_file)
        sw_metric_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sw_metric {session['pipeline']}.pkl"; sw_metric_filename = os.path.join(cfg['cache_dir'], sw_metric_file)
        if os.path.exists(sw_summary_filename) and os.path.exists(sw_summary_ch_filename) and os.path.exists(sw_summary_chst_filename) and os.path.exists(sw_sync_filename) and os.path.exists(sw_metric_filename):
            session['sw_summary'] = pd.read_parquet(sw_summary_filename)
            session['sw_summary_ch'] = pd.read_parquet(sw_summary_ch_filename)
            session['sw_summary_chst'] = pd.read_parquet(sw_summary_chst_filename)
            session['sw_sync'] = pd.read_parquet(sw_sync_filename)
            with open(sw_metric_filename, "rb") as f:
                session['sw_metric'] = pickle.load(f)
        else:
            sw = yasa.sw_detect(raw, include=(2,3), hypno=yasa.hypno_upsample_to_data(session['hypnos_adj'], sf_hypno=sf_hypno, data=raw))                
            if not (1 == np.sum(np.isin(session['sw_ch'], session['eeg']))/len(session['sw_ch'])):
                sw_ch = [session['eeg'][0]]
            session['sw_metric'] = sws_metrics(sw, session['sleep_stats']['SOL_ADJ'], session['hypnos_adj'], sw_ch=session['sw_ch'], stages=[2,3], period=4.5*3600)
            session['sw_sync']= sw.get_sync_events(center=pa_sw_center, time_before=pa_time_before, time_after=pa_time_after, filt=pa_filt, mask=pa_mask)
            session['sw_summary'] = sw.summary()
            session['sw_summary_ch'] = sw.summary(grp_chan=True)
            session['sw_summary_chst'] = sw.summary(grp_stage=True, grp_chan=True)
            
            session['sw_summary'].to_parquet(sw_summary_filename, compression="brotli")
            session['sw_summary_ch'].to_parquet(sw_summary_ch_filename, compression="brotli")
            session['sw_summary_chst'].to_parquet(sw_summary_chst_filename, compression="brotli")
            session['sw_sync'].to_parquet(sw_sync_filename, compression="brotli")
            with open(sw_metric_filename, "wb") as f:
                pickle.dump(session['sw_metric'], f)

    sessions[index] = session

if 'Hypno' in plots:
    for index, session in enumerate(sessions):
        raw  = session['raw'].copy().pick(session['eeg'])

        fig, axes = plt.subplots(round(len(session['eeg']))+2, 
                  figsize=(8, 4+len(session['eeg'])*2))
        fig.suptitle(f"{session['dts'].strftime(cfg['plot_dt_format'])} Hypnograms, {spect_specs}")
        hyp = yasa.hypno_int_to_str(session['hypnos_max']); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = axes[0])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / (hyp_stats["SPT"] + hyp_stats["SOL_ADJ"])))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Max Probs)')
        hyp = yasa.hypno_int_to_str(session['hypnos_adj']); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = axes[1])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / (hyp_stats["SPT"] + hyp_stats["SOL_ADJ"])))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Adj Probs)')
        for ch_index, ch in enumerate(session['eeg']):
            hyp = session['hypnos'][ch_index]; hyp_stats = sleep_stats(hyp)
            ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = axes[ch_index+2])
            ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / (hyp_stats["SPT"] + hyp_stats["SOL_ADJ"])))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} ({ch})')

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} hypno channels {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
        plt.plot(session['hypnos_max'])
        fig, ax = plt.subplots(figsize=(7.5, 3))
        hyp = yasa.hypno_int_to_str(session['hypnos_adj']); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = ax, hl_lw=5)
        ax.set_title(f'{session["dts"].strftime(cfg["plot_dt_format"])} Hypno Adj Probs\n{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / (hyp_stats["SPT"] + hyp_stats["SOL_ADJ"])))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} hypno {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'Cycles' in plots:
    for index, session in enumerate(sessions):
        hypno = smooth_hypno_custom(session['hypnos_adj'].copy(), window=15)        
        cycles, cycle_hypno = detect_sleep_cycles(hypno, min_cycle_duration=60, last_cycle_merge_threshold=45, min_nrem_between_cycles=10, gap_assignment='closest_rem')
        cycles_sum = summarize_cycles(cycles, hypno)
        rem_prosp_time = session['dts'] + timedelta(minutes=cycles_sum.loc[cycles_sum['rem_nrem_ratio'].idxmax()]['midtime']/(sf_hypno*60))
        phase_time = datetime.combine((session['dts']+timedelta(minutes=session['sleep_stats']['TIB'])).date(), dtime(4, 0)).replace(tzinfo=timezone.utc)
        rem_prosp_diff = (rem_prosp_time - phase_time).total_seconds()/60
        
        fig, ax = plt.subplots(figsize=(7.5, 3))
        hyp = yasa.hypno_int_to_str(hypno); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(session['dts'])), ax = ax, hl_lw=5, cycles=cycles, vline_dt=rem_prosp_time)
        ax.set_title(f'{session["dts"].strftime(cfg["plot_dt_format"])} Cycles Smooth Adj Probs\n{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / (hyp_stats["SPT"] + hyp_stats["SOL_ADJ"])))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}, Phase {m2h(rem_prosp_diff)}')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} cycles {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'HRV' in plots: # Process ECG
    if load_ecg:
        for index, session in enumerate(sessions):
            if len(session['ecg']) > 0:
                raw  = session['raw'].copy().pick(session['ecg'])
                ecg_invert = -1 if session['ecg_invert'] else 1
                fig, session['ecg_stats'], session['hrv_stages'], major_acc_epoch, session['hrv_col'], session['hrv'] = process_ecg(raw, session['ecg'], session['dts'], session['hypno_df'], session['acc_agg'], session['acc'], session['sleep_stats'], cfg, user, device, ecg_invert)
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
            spects_arr_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} spects_arr {session['pipeline']}.npz"; spects_arr_filename = os.path.join(cfg['cache_dir'], spects_arr_file)
            stimes_arr_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} stimes_arr {session['pipeline']}.npz"; stimes_arr_filename = os.path.join(cfg['cache_dir'], stimes_arr_file)
            sfreqs_arr_file = f"{session['dts'].strftime(cfg['file_dt_format'])} {user} sfreqs_arr {session['pipeline']}.npz"; sfreqs_arr_filename = os.path.join(cfg['cache_dir'], sfreqs_arr_file)
            if os.path.exists(spects_arr_filename) and os.path.exists(stimes_arr_filename) and os.path.exists(sfreqs_arr_filename):
                loaded = np.load(spects_arr_filename)
                session['spects'] = [loaded[f"arr{i}"] for i in range(len(loaded.files))]
                loaded = np.load(stimes_arr_filename)
                session['stimes'] = [loaded[f"arr{i}"] for i in range(len(loaded.files))]
                loaded = np.load(sfreqs_arr_filename)
                session['sfreqs'] = [loaded[f"arr{i}"] for i in range(len(loaded.files))]
            else:
                session['spects'], session['stimes'], session['sfreqs'] = create_spect(session['raw'], session['eeg'], multitaper_spectrogram, nanpow2db, spect_lim, frequency_range, time_bandwidth, num_tapers, 
                    window_params, min_nfft, detrend_opt, multiprocess, cpus,
                    weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
                np.savez_compressed(spects_arr_filename, **{f"arr{i}": arr for i, arr in enumerate(session['spects'])})
                np.savez_compressed(stimes_arr_filename, **{f"arr{i}": arr for i, arr in enumerate(session['stimes'])})
                np.savez_compressed(sfreqs_arr_filename, **{f"arr{i}": arr for i, arr in enumerate(session['sfreqs'])})
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
            fig, sessions[index]['sp_density'], sp_density_max, sleep_midtime, sp_phase_offset = plot_rolling_spindle_density(session['sp_summary'], session['sleep_stats'], session['dts'], cfg, channels=session['sp_ch'], window_minutes=30,stage_filter=[2],type_label='Spindles', verbose=False)
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} SPD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'SlowWaves' in plots:
    for index, session in enumerate(sessions):
        if (1 == np.sum(np.isin(session['sw_ch'], session['eeg']))/len(session['sw_ch'])):
            fig, sessions[index]['sw_density'], sw_density_max, sleep_midtime, sw_phase_offset = plot_rolling_spindle_density(session['sw_summary'], session['sleep_stats'], session['dts'], cfg, channels=session['sw_ch'], window_minutes=45,stage_filter=[2,3],type_label='Slow Wave',verbose=False)    
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} SWD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'SpindlesFreq' in plots:
    for index, session in enumerate(sessions):
        if (1 == np.sum(np.isin(session['sp_slow_ch'], session['eeg']))/len(session['sp_slow_ch'])) and (1 == np.sum(np.isin(session['sp_fast_ch'], session['eeg']))/len(session['sp_fast_ch'])):
            fig = spindles_slow_fast(session['sp_summary'], yasa.hypno_upsample_to_data(session['hypnos_adj'], sf_hypno=sf_hypno, data=session['raw']), session['raw'], session['eeg'], slow_ch=session['sp_slow_ch'], fast_ch=session['sp_fast_ch'])
            if fig is not None:
                png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} Spindles Freqs {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'REMProspensity' in plots:
    for index, session in enumerate(sessions):
        session['sleep_stats']['Lat_REM_ADJ'] = session['sleep_stats']['Lat_REM'] + session['sleep_stats']['SOL'] - session['sleep_stats']['SOL_ADJ']
        phase_offset_hr, peak_time, fig = compute_rolling_rem_propensity(session['hypnos_adj'], session['dts'], 1/sf_hypno, session['sleep_stats']['SOL_ADJ'], window_min=45, stages=[4], lat_rem=session['sleep_stats']['Lat_REM_ADJ'])
        if fig is not None:
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} REM Prospensity {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'REMInflection' in plots:
    for index, session in enumerate(sessions):
        session['sleep_stats']['Lat_REM_ADJ'] = session['sleep_stats']['Lat_REM'] + session['sleep_stats']['SOL'] - session['sleep_stats']['SOL_ADJ']
        phase_offset_hr, peak_time, fig = compute_cumulative_rem_phase(session['hypnos_adj'], session['dts'], 1/sf_hypno, smooth_window=90, min_slope_threshold=0.7, sol=session['sleep_stats']['SOL_ADJ'])
        if fig is not None:
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} REM Inflection {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

# Power frequency plot with mne PSD computation
if 'Features' in plots:
    for index, session in enumerate(sessions):        
        psd, freq = psds(session['raw_ori'], freq_method, session['eeg'], w_fft, freq_lim, nj)
        fig, amp, max_amp = psd_plot(psd, freq, session['dts'], session['raw_ori'], session['eeg'], session['ref_ch'], session['sp_sync'], session['sp_summary'], session['sp_ch'], session['sp_metric'], session['sw_sync'], session['sw_summary_ch'], session['sw_summary_chst'], session['sw_ch'], session['sw_metric'], freq_method, w_fft, freq_lim, units, sig_specs, cfg, nj, plot_avg=plot_avg)
        png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} PSD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

if 'Radar' in plots:
    for index, session in enumerate(sessions):
        if 'ecg_stats' in session.keys():
            fig = plot_radar(session['dts'], session['sleep_stats'], session['ecg_stats'], session['acc'], cfg, n3_goal = 90, rem_goal = 105, awk_goal = 30, hr_goal = 45, hrv_goal = 35, mh_goal = 4.3)
            png_file = f"{session['dts'].strftime(cfg['file_dt_format'])} Radar {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
import numpy as np

def normalize_to_clock_time(df, density_col='density'):
    """
    Normalize timestamps to clock time (22:00-10:00) for overlay plotting.
    Assumes df has a 'time' column with timezone-aware timestamps.
    """
    df = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Extract just the time component
    df['clock_time'] = df['time'].dt.time
    
    # Create a reference date for plotting (using a dummy date)
    reference_date = datetime(2000, 1, 1)  # Arbitrary reference date
    
    # Convert times to datetime with reference date
    clock_datetimes = []
    for t in df['clock_time']:
        dt = datetime.combine(reference_date, t)
        
        # If time is before 22:00, it's the next day (add 24 hours)
        if t < time(22, 0):
            dt = dt + timedelta(days=1)
        
        clock_datetimes.append(dt)
    
    df['plot_time'] = clock_datetimes
    
    return df

def plot_multiple_nights(sessions_dict, density_type='sw_density', 
                         title='Sleep Density Across Multiple Nights',
                         ylabel='Density (per min)', 
                         figsize=(14, 8)):
    """
    Plot multiple nights of sleep density data on the same axes.
    
    Parameters:
    -----------
    sessions_dict : dict
        Dictionary where keys are session labels and values are DataFrames
        with 'time' and 'density' columns
    density_type : str
        Type of density being plotted (for title customization)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette - you can customize this
    colors = plt.cm.tab10(np.linspace(0, 1, len(sessions_dict)))
    
    # Plot each session
    for idx, (session_label, df) in enumerate(sessions_dict.items()):
        # Normalize to clock time
        df_normalized = normalize_to_clock_time(df)
        
        # Extract date from original timestamp for legend
        session_date = df['time'].iloc[0].strftime('%Y-%m-%d')
        
        # Plot
        ax.plot(df_normalized['plot_time'], 
                df_normalized['density'],
                label=f'{session_label} ({session_date})',
                color=colors[idx],
                alpha=0.8,
                linewidth=1.5)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    
    # Set x-axis limits (22:00 to 10:00)
    reference_date = datetime(2000, 1, 1)
    start_time = datetime.combine(reference_date, time(22, 0))
    end_time = datetime.combine(reference_date + timedelta(days=1), time(10, 0))
    ax.set_xlim(start_time, end_time)
    
    # Styling
    ax.set_xlabel('Clock Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add shaded regions for typical sleep stages (optional)
    ax.axvspan(start_time, datetime.combine(reference_date, time(23, 30)), 
               alpha=0.1, color='blue', label='_nolegend_')  # Early evening
    ax.axvspan(datetime.combine(reference_date + timedelta(days=1), time(6, 0)), 
               end_time, alpha=0.1, color='orange', label='_nolegend_')  # Morning
    
    plt.tight_layout()
    return fig, ax

def plot_with_statistics(sessions_dict, density_type='sw_density',
                         show_mean=True, show_std=False,
                         title='Sleep Density Across Multiple Nights',
                         ylabel='Density (per min)',
                         figsize=(14, 8)):
    """
    Enhanced version that can show mean and standard deviation across nights.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sessions_dict)))
    
    # Store normalized data for statistics
    all_normalized = {}
    
    # Plot each session
    for idx, (session_label, df) in enumerate(sessions_dict.items()):
        df_normalized = normalize_to_clock_time(df)
        all_normalized[session_label] = df_normalized
        
        session_date = df['time'].iloc[0].strftime('%Y-%m-%d')
        
        ax.plot(df_normalized['plot_time'], 
                df_normalized['density'],
                label=f'{session_label} ({session_date})',
                color=colors[idx],
                alpha=0.6,
                linewidth=1.2)
    
    # Calculate and plot statistics if requested
    if show_mean and len(sessions_dict) > 1:
        # Resample all sessions to common time grid
        reference_date = datetime(2000, 1, 1)
        time_grid = pd.date_range(
            start=datetime.combine(reference_date, time(22, 0)),
            end=datetime.combine(reference_date + timedelta(days=1), time(10, 0)),
            freq='1min'
        )
        
        # Interpolate all sessions to common grid
        interpolated_data = []
        for session_label, df_norm in all_normalized.items():
            df_temp = df_norm.set_index('plot_time')['density']
            df_resampled = df_temp.reindex(time_grid, method='nearest', limit=1)
            interpolated_data.append(df_resampled.values)
        
        # Calculate statistics
        density_matrix = np.array(interpolated_data)
        mean_density = np.nanmean(density_matrix, axis=0)
        
        # Plot mean
        ax.plot(time_grid, mean_density, 
                color='black', 
                linewidth=2.5, 
                label='Mean', 
                linestyle='--',
                alpha=0.8)
        
        # Plot standard deviation band if requested
        if show_std:
            std_density = np.nanstd(density_matrix, axis=0)
            ax.fill_between(time_grid, 
                          mean_density - std_density,
                          mean_density + std_density,
                          color='gray', 
                          alpha=0.2, 
                          label='±1 STD')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    
    # Set x-axis limits
    reference_date = datetime(2000, 1, 1)
    start_time = datetime.combine(reference_date, time(22, 0))
    end_time = datetime.combine(reference_date + timedelta(days=1), time(10, 0))
    ax.set_xlim(start_time, end_time)
    
    # Styling
    ax.set_xlabel('Clock Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Assuming you have your sessions dictionary structured like:
    # sessions = {
    #     0: {'sw_density': DataFrame},
    #     1: {'sw_density': DataFrame},
    #     2: {'sw_density': DataFrame},
    # }
    
    # Method 1: Convert your sessions to the format needed
    sessions_for_plotting = {}
    for idx, session_data in enumerate(sessions):
        if 'sw_density' in session_data:
            sessions_for_plotting[f'Night {idx+1}'] = session_data['sw_density']
    
    # Basic plot
    fig1, ax1 = plot_multiple_nights(
        sessions_for_plotting,
        density_type='Slow Wave',
        title='Slow Wave Density Across Multiple Nights',
        ylabel='SW Density (per min)'
    )
    plt.show()
    
    # Plot with statistics
    fig2, ax2 = plot_with_statistics(
        sessions_for_plotting,
        show_mean=True,
        show_std=True,
        density_type='Slow Wave',
        title='Slow Wave Density - Multiple Nights with Mean',
        ylabel='SW Density (per min)'
    )
    plt.show()
    
    # You can also plot spindle density the same way
    spindles_for_plotting = {}
    for idx, session_data in enumerate(sessions):
        if 'sp_density' in session_data:
            spindles_for_plotting[f'Night {idx+1}'] = session_data['sp_density']
    
    if spindles_for_plotting:
        fig3, ax3 = plot_multiple_nights(
            spindles_for_plotting,
            density_type='Spindle',
            title='Sleep Spindle Density Across Multiple Nights',
            ylabel='Spindle Density (per min)'
        )
        plt.show()
        
#try6
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def align_to_sleep_onset(df, sol_minutes, density_col='density'):
    """
    Align timestamps to minutes from sleep onset latency (SOL).
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with 'time' and 'density' columns
    sol_minutes : float
        Sleep onset time in minutes from recording start
    density_col : str
        Name of the density column
    
    Returns:
    --------
    DataFrame with added 'minutes_from_sol' column
    """
    df = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Get the first timestamp as reference (recording start)
    recording_start = df['time'].iloc[0]
    
    # Calculate minutes from recording start for each point
    df['minutes_from_start'] = (df['time'] - recording_start).dt.total_seconds() / 60
    
    # Calculate minutes from SOL by subtracting SOL offset
    df['minutes_from_sol'] = df['minutes_from_start'] - sol_minutes
    
    return df

def plot_density_from_sol(sessions_list, 
                         density_type='sw_density',
                         title='Sleep Density from Sleep Onset',
                         ylabel='Density (per min)', 
                         xlim_minutes=(0, 600),  # Default to 10 hours
                         figsize=(14, 8),
                         show_legend=True):
    """
    Plot multiple nights of sleep density data aligned to sleep onset.
    
    Parameters:
    -----------
    sessions_list : list or dict
        List or dictionary of session data, where each session contains
        'sw_density' or 'spindle_density' DataFrame and 'sleep_stats' dict
    density_type : str
        Type of density to plot ('sw_density' or 'spindle_density')
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlim_minutes : tuple
        X-axis limits in minutes from SOL (min, max)
    figsize : tuple
        Figure size
    show_legend : bool
        Whether to show legend
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle both list and dict inputs
    if isinstance(sessions_list, dict):
        sessions_items = sessions_list.items()
        num_sessions = len(sessions_list)
    else:
        sessions_items = enumerate(sessions_list)
        num_sessions = len(sessions_list)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, num_sessions))
    
    # Plot each session
    valid_sessions = 0
    for idx, (session_idx, session_data) in enumerate(sessions_items):
        try:
            # Get density data
            if density_type not in session_data:
                print(f"Warning: {density_type} not found in session {session_idx}")
                continue
            
            df = session_data[density_type].copy()
            
            # Get SOL time
            if 'sleep_stats' not in session_data or 'SOL_ADJ' not in session_data['sleep_stats']:
                print(f"Warning: SOL_ADJ not found for session {session_idx}")
                continue
            
            sol_minutes = session_data['sleep_stats']['SOL_ADJ']
            
            # Align to SOL
            df_aligned = align_to_sleep_onset(df, sol_minutes)
            
            # Get session date for label
            session_date = df['time'].iloc[0].strftime('%Y-%m-%d')
            
            # Plot
            ax.plot(df_aligned['minutes_from_sol'], 
                   df_aligned['density'],
                   label=f'Night {session_idx} ({session_date})',
                   color=colors[idx],
                   alpha=0.7,
                   linewidth=1.5)
            
            valid_sessions += 1
            
        except Exception as e:
            print(f"Error processing session {session_idx}: {e}")
            continue
    
    if valid_sessions == 0:
        print("No valid sessions to plot")
        return fig, ax
    
    # Set x-axis limits
    if xlim_minutes:
        ax.set_xlim(xlim_minutes[0], xlim_minutes[1])
    
    # Add vertical lines for typical sleep cycle markers
    cycle_duration = 90  # minutes
    for cycle_num in range(1, 7):  # Mark first 6 cycles
        x_pos = cycle_num * cycle_duration
        if xlim_minutes and x_pos <= xlim_minutes[1]:
            ax.axvline(x=x_pos, color='gray', linestyle=':', 
                      alpha=0.3, linewidth=1)
            ax.text(x_pos, ax.get_ylim()[1] * 0.95, f'C{cycle_num}',
                   ha='center', va='top', fontsize=8, color='gray')
    
    # Styling
    ax.set_xlabel('Minutes from Sleep Onset', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Add hour markers on top x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(xlim_minutes[0]/60, xlim_minutes[1]/60)
    ax2.set_xlabel('Hours from Sleep Onset', fontsize=11)
    
    plt.tight_layout()
    return fig, ax

def plot_density_with_statistics(sessions_list,
                                density_type='sw_density',
                                show_mean=True,
                                show_std=False,
                                show_median=False,
                                title='Sleep Density from Sleep Onset',
                                ylabel='Density (per min)',
                                xlim_minutes=(0, 600),
                                figsize=(14, 8)):
    """
    Enhanced version with statistical overlays (mean, median, std).
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle both list and dict inputs
    if isinstance(sessions_list, dict):
        sessions_items = sessions_list.items()
        num_sessions = len(sessions_list)
    else:
        sessions_items = enumerate(sessions_list)
        num_sessions = len(sessions_list)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_sessions))
    
    # Store aligned data for statistics
    all_aligned = []
    max_minutes = 0
    
    # Plot each session and collect data
    valid_sessions = 0
    for idx, (session_idx, session_data) in enumerate(sessions_items):
        try:
            if density_type not in session_data:
                continue
            
            df = session_data[density_type].copy()
            sol_time = session_data['sleep_stats']['SOL_ADJ']
            
            df_aligned = align_to_sleep_onset(df, sol_time)
            
            # Track maximum minutes for statistics
            max_minutes = max(max_minutes, df_aligned['minutes_from_sol'].max())
            
            session_date = df['time'].iloc[0].strftime('%Y-%m-%d')
            
            # Plot individual night
            ax.plot(df_aligned['minutes_from_sol'], 
                   df_aligned['density'],
                   label=f'Night {session_idx} ({session_date})',
                   color=colors[idx],
                   alpha=0.5,
                   linewidth=1.2)
            
            all_aligned.append(df_aligned)
            valid_sessions += 1
            
        except Exception as e:
            print(f"Error processing session {session_idx}: {e}")
            continue
    
    # Calculate and plot statistics if we have multiple nights
    if valid_sessions > 1 and (show_mean or show_median or show_std):
        # Create common time grid (1-minute resolution)
        time_grid = np.arange(0, min(max_minutes, xlim_minutes[1]), 1)
        
        # Interpolate all sessions to common grid
        interpolated_data = []
        for df_aligned in all_aligned:
            # Interpolate to common grid
            interp_density = np.interp(
                time_grid,
                df_aligned['minutes_from_sol'].values,
                df_aligned['density'].values,
                left=np.nan,
                right=np.nan
            )
            interpolated_data.append(interp_density)
        
        # Convert to numpy array for statistics
        density_matrix = np.array(interpolated_data)
        
        # Calculate statistics
        if show_mean:
            mean_density = np.nanmean(density_matrix, axis=0)
            ax.plot(time_grid, mean_density,
                   color='black',
                   linewidth=2.5,
                   label='Mean',
                   linestyle='-',
                   alpha=0.9)
        
        if show_median:
            median_density = np.nanmedian(density_matrix, axis=0)
            ax.plot(time_grid, median_density,
                   color='darkred',
                   linewidth=2,
                   label='Median',
                   linestyle='--',
                   alpha=0.8)
        
        if show_std and show_mean:
            std_density = np.nanstd(density_matrix, axis=0)
            ax.fill_between(time_grid,
                          mean_density - std_density,
                          mean_density + std_density,
                          color='gray',
                          alpha=0.2,
                          label='±1 STD')
    
    # Set x-axis limits
    if xlim_minutes:
        ax.set_xlim(xlim_minutes[0], xlim_minutes[1])
    
    # Add cycle markers
    cycle_duration = 90
    for cycle_num in range(1, 7):
        x_pos = cycle_num * cycle_duration
        if xlim_minutes and x_pos <= xlim_minutes[1]:
            ax.axvline(x=x_pos, color='gray', linestyle=':', 
                      alpha=0.3, linewidth=1)
            ax.text(x_pos, ax.get_ylim()[1] * 0.95, f'C{cycle_num}',
                   ha='center', va='top', fontsize=8, color='gray')
    
    # Styling
    ax.set_xlabel('Minutes from Sleep Onset', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add hour markers on top
    ax2 = ax.twiny()
    ax2.set_xlim(xlim_minutes[0]/60, xlim_minutes[1]/60)
    ax2.set_xlabel('Hours from Sleep Onset', fontsize=11)
    
    plt.tight_layout()
    return fig, ax

def plot_comparison_subplots(sessions_list,
                            figsize=(14, 10)):
    """
    Create subplot comparison of both slow wave and spindle density.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Handle both list and dict inputs
    if isinstance(sessions_list, dict):
        sessions_items = sessions_list.items()
        num_sessions = len(sessions_list)
    else:
        sessions_items = list(enumerate(sessions_list))
        num_sessions = len(sessions_list)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_sessions))
    
    density_configs = [
        ('sw_density', 'Slow Wave Density', 'SW Density (per min)', axes[0]),
        ('sp_density', 'Spindle Density', 'Spindle Density (per min)', axes[1])
    ]
    
    for density_type, title, ylabel, ax in density_configs:
        valid_sessions = 0
        
        # Need to iterate through sessions_items again for each density type
        for idx, (session_idx, session_data) in enumerate(sessions_items):
            try:
                if density_type not in session_data:
                    continue
                
                df = session_data[density_type].copy()
                sol_minutes = session_data['sleep_stats']['SOL_ADJ']
                
                df_aligned = align_to_sleep_onset(df, sol_minutes)
                
                session_date = df['time'].iloc[0].strftime('%m/%d')
                
                ax.plot(df_aligned['minutes_from_sol'],
                       df_aligned['density'],
                       label=f'{session_date}',
                       color=colors[idx],
                       alpha=0.7,
                       linewidth=1.5)
                
                valid_sessions += 1
                
            except Exception as e:
                continue
        
        # Add cycle markers
        for cycle_num in range(1, 7):
            x_pos = cycle_num * 90
            if x_pos <= 600:
                ax.axvline(x=x_pos, color='gray', linestyle=':', 
                          alpha=0.3, linewidth=1)
        
        ax.set_xlim(0, 600)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', framealpha=0.9, ncol=3, fontsize=9)
    
    axes[1].set_xlabel('Minutes from Sleep Onset', fontsize=12)
    
    # Add hour scale on top
    ax2 = axes[0].twiny()
    ax2.set_xlim(0, 10)
    ax2.set_xlabel('Hours from Sleep Onset', fontsize=11)
    
    plt.suptitle('Sleep Microarchitecture Aligned to Sleep Onset', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, axes

# Example usage:
if __name__ == "__main__":
    # Basic plot for slow wave density
    fig1, ax1 = plot_density_from_sol(
        sessions,
        density_type='sw_density',
        title='Slow Wave Density from Sleep Onset',
        ylabel='SW Density (per min)',
        xlim_minutes=(0, 600)  # First 10 hours
    )
    plt.show()
    
    # Plot with statistics
    fig2, ax2 = plot_density_with_statistics(
        sessions,
        density_type='sw_density',
        show_mean=True,
        show_std=True,
        show_median=False,
        title='Slow Wave Density - Multiple Nights with Mean',
        ylabel='SW Density (per min)',
        xlim_minutes=(0, 600)
    )
    plt.show()
    
    # Spindle density
    fig3, ax3 = plot_density_from_sol(
        sessions,
        density_type='sp_density',
        title='Spindle Density from Sleep Onset',
        ylabel='Spindle Density (per min)',
        xlim_minutes=(0, 600)
    )
    plt.show()
    
    # Combined subplot view
    fig4, axes4 = plot_comparison_subplots(
        sessions,
        figsize=(14, 10)
    )
    plt.show()