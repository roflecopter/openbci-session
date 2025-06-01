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
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# comment qskit line if you dont process ECG
from qskit import butter_lowpass_filter, hrv_process, hrv_quality, sc_interp # https://github.com/roflecopter/qskit
from time import sleep

# config, if relative path not working then use explicit path to working dir (repo dir with scripts and yml) or modify working directory in IDE/GUI settings
# working_dir = '/path/to/openbci-session'
working_dir = os.getcwd()
cfg_file = os.path.join(working_dir, "sleep_analysis.yml")

# rename sleep_analysis.yml.sample to sleep_analysis.yml and set directories
with open(cfg_file, "r") as yamlfile:
    cfg_base = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg = cfg_base['default']

# git pull https://github.com/preraulab/multitaper_toolbox/
os.chdir(cfg['multitaper_dir'])
from multitaper_spectrogram_python import multitaper_spectrogram, nanpow2db
os.chdir(working_dir)

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('future.no_silent_downcasting', True)

nj = 15 # n_jobs for multiprocessing, usually n_cpu - 1

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})

# used for cache/image file names
device = 'openbci'
user = 'user'

# make array with bdf files recorded with session_start.py
f_name = os.path.join(cfg['data_dir'], '2024-01-25_22-02-25-Alex-OBCI_F1.TXT.bdf')
sleeps = {'1': {'file': f_name, 'ecg_invert': False}} # ecg_invert flips ecg signal, in case electrodes were placed inverse by mistake

# overwrite image files if exists (HRV, Hypno in PNG)
image_overwrite = True

# signal filtering
bpf = [.35, 45] # band pass filter, [0.1, None] or [.35, 45]
nf = [50,1] # notch filter, set to 50 or 60 Hz powerline noise freq depending on your country
eog_bpf = [.5,8]; emg_bpf = [10,70] # filter for EOG data
sf_to = 256 # sampling rate to resample for fast processing

#plots = ['Hypno', 'HRV', 'Features','Spectrum_YASA','Spectrum','Topomap', 'Radar'] # to plot all use: plots = ['Hypno', 'HRV', 'Features','Spectrum','Topomap']
plots = ['Hypno', 'Spectrum_YASA','Features','Topomap']
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
sp_ch = ['F7','F8']
sw_ch = ['F7','F8']

# multitaper spectrograms settings, can leave as is if not sure what is it
spect_vlim = [6,24]; spect_lim = [1,16]; freq_lim = [1,30]
time_bandwidth = 24 # Set time-half bandwidth
num_tapers = time_bandwidth*2 - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = [60, 30]  # Window size is Xs with step size of Ys

# units for labels
units = {'psd_dB': 'dB(µV²/Hz)', 'amp': 'µV', 'p': 'µV²', 'p_dB': 'dB(µV²)', 'rel': '%'}
sig_specs = f'sf={sf_to}Hz, notch={nf}, bandpass={bpf}'
spect_specs = f'num_tapers={num_tapers}, window={window_params}'

# list of bands for band power
bp_bands = [
    (1, 4, "Delta"),
    (4, 8, "Theta"),
    (8, 12, "Alpha"),
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

def m2h(mins):
    h = int(mins/60)
    m = round(mins-h*60)
    if h > 0:
        if m < 10:
            return f"{h}h0{m}m"
        else:
            return f"{h}h{m}m"
    else:
        return f"{round(mins)}m"

def sleep_stats(hypno, hypno_sf = 1/30):
    # calc sleep onset latency
    sol = 0; stop = False; epoch_time = 1 / hypno_sf
    major_awakenings_n = 0; awake_window_mins = 20; min_awake_rate = .5
    # find first N epochs where awake occured less than 50% of time
    # find last awake epoch in these N epochs and threat it as end of sleep onset
    for h in range(0, len(hypno)-1):
        if hypno[h] == 'W' and hypno[h+1] != 'W':
            major_awakenings_n = major_awakenings_n + 1
        if not stop:
            # check how much awakenings in next awake_mins minutes
            awakes_n = 0
            last_awake = h
            for hj in range(h, min(h+int(awake_window_mins*60/epoch_time), len(hypno))):
                if hypno[hj] == 'W':
                    awakes_n = awakes_n + 1
                    last_awake = hj
            if hypno[h] == 'W':
                sol = sol + 30
            # stop when there are less than 50% awakenings in window
            if awakes_n < awake_window_mins*(60/epoch_time)*min_awake_rate: 
                stop = True
                sol = sol + 30 * (last_awake - h)
    tst = 0; waso = 0
    for h in range(0, len(hypno)):
        if (h > (sol / epoch_time)):
            if (hypno[h] != 'W'):
                tst = tst + epoch_time
            else:
                waso = waso + epoch_time
            
    hyp_stats = yasa.sleep_statistics(yasa.hypno_str_to_int(hypno), 1/30)
    hyp_stats['SOL_ADJ'] = sol / 60
    hyp_stats['TST_ADJ'] = tst / 60
    hyp_stats['N_AWAKE'] = major_awakenings_n
    hyp_stats['WASO_ADJ'] = waso / 60
    return hyp_stats

def plot_hypnogram(hyp, lw=1, hl_lw=3, font_size=10, highlight={'*': 'dimgrey', 'WAKE': 'orange', 'N1': None, 'N2': 'lightskyblue', 'N3': 'indigo', 'REM': 'red'}, fill_color=None, ax=None):
    """
    Plot a hypnogram.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    hyp : :py:class:`yasa.Hypnogram`
        A YASA hypnogram instance.
    lw : float
        Linewidth.
    highlight : str or None
        Optional stage to highlight with alternate color.
    fill_color : str or None
        Optional color to fill space above hypnogram line.
    ax : :py:class:`matplotlib.axes.Axes`
        Axis on which to draw the plot, optional.

    Returns
    -------
    ax : :py:class:`matplotlib.axes.Axes`
        Matplotlib Axes

    Examples
    --------
    .. plot::

        >>> from yasa import simulate_hypnogram
        >>> import matplotlib.pyplot as plt
        >>> hyp = simulate_hypnogram(tib=300, seed=11)
        >>> ax = hyp.plot_hypnogram()
        >>> plt.tight_layout()

    .. plot::

        >>> from yasa import Hypnogram
        >>> values = 4 * ["W", "N1", "N2", "N3", "REM"] + ["ART", "N2", "REM", "W", "UNS"]
        >>> hyp = Hypnogram(values, freq="24min").upsample("30s")
        >>> ax = hyp.plot_hypnogram(lw=2, fill_color="thistle")
        >>> plt.tight_layout()

    .. plot::

        >>> from yasa import simulate_hypnogram
        >>> import matplotlib.pyplot as plt
        >>> fig, axes = plt.subplots(nrows=2, figsize=(6, 4), constrained_layout=True)
        >>> hyp_a = simulate_hypnogram(n_stages=3, seed=99)
        >>> hyp_b = simulate_hypnogram(n_stages=3, seed=99, start="2022-01-31 23:30:00")
        >>> hyp_a.plot_hypnogram(lw=1, fill_color="whitesmoke", highlight=None, ax=axes[0])
        >>> hyp_b.plot_hypnogram(lw=1, fill_color="whitesmoke", highlight=None, ax=axes[1])
    """
    
    # hyp = yasa.Hypnogram(yasa.hypno_int_to_str(hypnos_max[index]), start=pd.to_datetime(dts[index]))
    # lw=1; hl_lw=3; font_size=10; highlight={'*': 'lightgrey', 'WAKE': 'orange', 'N1': None, 'N2': 'lightskyblue', 'N3': 'indigo', 'REM': 'red'}; fill_color=None; ax=None

    assert isinstance(hyp, yasa.Hypnogram), "`hypno` must be YASA Hypnogram."

    # Work with a copy of the Hypnogram to not alter the original
    hyp = hyp.copy()

    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": font_size})

    ## Remap stages to be in desired y-axis order ##
    # Start with default of all allowed labels
    stage_order = hyp.labels.copy()
    stages_present = hyp.hypno.unique()
    # Remove Art/Uns from stage order, and place back individually at front to be higher on plot
    art_str = stage_order.pop(stage_order.index("ART"))
    uns_str = stage_order.pop(stage_order.index("UNS"))
    if "ART" in stages_present:
        stage_order.insert(0, art_str)
    if "UNS" in stages_present:
        stage_order.insert(0, uns_str)
    # Put REM after WAKE if all 5 standard stages are allowed
    if hyp.n_stages == 5:
        stage_order.insert(stage_order.index("WAKE") + 1, stage_order.pop(stage_order.index("REM")))
    # Reset the Hypnogram mapping so any future returns have this order
    hyp.mapping = {stage: i for i, stage in enumerate(stage_order)}

    ## Extract values to plot ##
    hypno = hyp.as_int()
    # Reduce to breakpoints (where stages change) to avoid drawing individual lines for every epoch
    hypno = hypno[hypno.shift().ne(hypno)]
    # Extract x-values (bins) and y-values to plot
    yvalues = hypno.to_numpy()
    if hyp.start is not None:
        final_bin_edge = pd.Timestamp(hyp.start) + pd.Timedelta(hyp.duration, unit="min")
        bins = np.append(hypno.index.to_list(), final_bin_edge)
        bins = [mdates.date2num(b) for b in bins]
        xlabel = "Time"
    else:
        final_bin_edge = hyp.duration * 60
        bins = np.append(hyp.timedelta[hypno.index].total_seconds(), final_bin_edge)
        bins /= 60 if hyp.duration <= 90 else 3600
        xlabel = "Time [mins]" if hyp.duration <= 90 else "Time [hrs]"

    # Make mask to draw the highlighted stage
    
    # Open the figure
    if ax is None:
        ax = plt.gca()

    # Draw background filling
    if fill_color is not None:
        bline = hyp.mapping["WAKE"]  # len(stage_order) - 1 to fill from bottom
        ax.stairs(yvalues.clip(bline), bins, baseline=bline, color=fill_color, fill=True, lw=0)
    # Draw main hypnogram line, highlighted stage line, and Artefact/Unscored line
    ax.stairs(yvalues, bins, baseline=None, color=highlight['*'], lw=lw)
    for stage in highlight:
        stage_highlight = np.ma.masked_not_equal(yvalues, hyp.mapping.get(stage))
        if not stage_highlight.mask.all():
            ax.hlines(stage_highlight, xmin=bins[:-1], xmax=bins[1:], color=highlight[stage], lw=hl_lw)

    # Aesthetics
    ax.use_sticky_edges = False
    ax.margins(x=0, y=1 / len(stage_order) / 2)  # 1/n_epochs/2 gives half-unit margins
    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(stage_order)
    ax.set_ylabel("Stage")
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.spines[["right", "top"]].set_visible(False)
    if hyp.start is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Revert font-size
    plt.rcParams.update({"font.size": old_fontsize})
    return ax

def plot_average(
    self,
    event_type,
    center="Peak",
    hue="Channel",
    time_before=1,
    time_after=1,
    filt=(None, None),
    mask=None,
    figsize=(6, 4.5),
    ax=None,
    **kwargs,
):
    """Plot the average event (not for REM, spindles & SW only)"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_sync = self.get_sync_events(
        center=center, time_before=time_before, time_after=time_after, filt=filt, mask=mask
    )
    assert not df_sync.empty, "Could not calculate event-locked data."
    assert hue in ["Stage", "Channel"], "hue must be 'Channel' or 'Stage'"
    assert hue in df_sync.columns, "%s is not present in data." % hue

    # if event_type == "spindles":
    #     title = "Average spindle"
    # else:  # "sw":
    #     title = "Average SW"

    # Start figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(data=df_sync, x="Time", y="Amplitude", hue=hue, ax=ax, **kwargs)
    ax.set_xlim(df_sync["Time"].min(), df_sync["Time"].max())
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude (uV)")
    return ax

def spindle_metrics(sp, sol, hypno, sp_ch=['F7','F8'], stages=[2], period=4.5*3600):
    # Claude 4 Sonnet recommendations
    # 1. N2 Spindle Density Ratio
    # (F7_spindle_density + F8_spindle_density) / 2 in N2 stage
    # Decision threshold: If < 0.8 spindles/minute → "Sleep architecture was fragmented - consider what disrupted deep sleep last night"
    # Normal range: 0.8-1.5 spindles/minute based on your data
    sp_total = sp.summary(grp_chan=True, grp_stage=True, aggfunc='mean')
    spindle_density = np.mean(sp_total.loc[(2, sp_ch), 'Density'].values)
    
    # 2. Early/Late N2 Spindle Density Ratio:
    # - Ratio < 0.5: Excellent deep sleep dominance early (very healthy)
    # - Ratio 0.5-1.0: Normal healthy pattern
    # - Ratio > 1.2: Concerning - possibly poor deep sleep or fragmented early sleep    f3hd_sum = spf3h.summary(grp_chan=True, grp_stage=True, aggfunc='mean')
    sp_summary = sp.summary()
    sp_data_first = sp_summary[
        (sp_summary['Stage'].isin(stages)) &
        (sp_summary['Channel'].isin(sp_ch)) & 
        (sp_summary['Start'] > sol * 60) &  
        (sp_summary['Start'] < sol * 60 + period)
    ]    
    stage_epochs_first = np.sum(np.isin(hypno[int(sol*2):int(sol*2 + period/30)], stages))
    stage_duration_first = stage_epochs_first * 30 / 60  # Assuming 30s epochs
    
    first_density = (len(sp_data_first)/len(sp_ch))/stage_duration_first
    
    sp_data_last= sp_summary[
        (sp_summary['Stage'].isin(stages)) &
        (sp_summary['Channel'].isin(sp_ch)) & 
        (sp_summary['Start'] > 60*len(hypno)/2 - period)
    ]    
    stage_epochs_last = np.sum(np.isin(hypno[int(len(hypno) - period/30):], stages))
    stage_duration_last = stage_epochs_last * 30 / 60  # Assuming 30s epochs
    
    last_density = (len(sp_data_last)/len(sp_ch))/stage_duration_last
    sp_early_late_density =  first_density/last_density
    
    # 3. Spindle Amplitude Consistency Score
    # 1 - (Standard deviation of F7+F8 spindle amplitudes / Mean amplitude)
    # Decision threshold: If < 0.7 → "Variable sleep depth - check for environmental disruptions or stress"
    # Interpretation:
    # Score close to 1.0: Very consistent spindle amplitudes → stable sleep depth
    # Score < 0.7: High variability → potentially fragmented sleep architecture
    # Negative scores: Extremely high variability (rare, but possible with artifacts)
    
    spindle_data = sp.summary()[(sp.summary()['Stage'] == 2) & (sp.summary()['Channel'].isin(sp_ch))]
    spindle_cv = np.std(spindle_data['Amplitude'].values)/np.mean(spindle_data['Amplitude'].values)
    return spindle_density, first_density, last_density, spindle_cv

def sws_metrics(sw, sol, hypno, sw_ch = ['F7','F8'], stages=[2,3], period = 4.5*3600):
    sw_summary = sw.summary()
    # 1. SWS Power Stability Index
    # Calculate coefficient of variation for SWS amplitude across the night
    # Decision: If < 0.6 → "Inconsistent deep sleep - check sleep environment or stress levels"
    sws_amplitudes = sw_summary['PTP']  # However you extract SWS events
    sws_amp_cv = np.std(sws_amplitudes) / np.mean(sws_amplitudes)
        
    # 2. Early Night SWS Dominance Ratio
    # First 3 hours SWS density / Last 3 hours SWS density
    # (Similar to spindle approach but for slow waves)
    # Decision: If < 2.0 → "Sleep pressure was low - consider earlier bedtime or more physical activity"
    # Normal: Should be >2.0 since SWS heavily front-loads
    sw_data_first = sw_summary[
        (sw_summary['Stage'].isin(stages)) &
        (sw_summary['Channel'].isin(sw_ch)) & 
        (sw_summary['Start'] > sol * 60) &  
        (sw_summary['Start'] < sol * 60 + period)
    ]    
    stage_epochs_first = np.sum(np.isin(hypno[int(sol*2):int(sol*2 + period/30)], stages))
    stage_duration_first = stage_epochs_first * 30 / 60  # Assuming 30s epochs
    
    first_density = (len(sw_data_first)/len(sw_ch))/stage_duration_first
    
    sw_data_last= sw_summary[
        (sw_summary['Stage'].isin(stages)) &
        (sw_summary['Channel'].isin(sw_ch)) & 
        (sw_summary['Start'] > 60*len(hypno)/2 - period)
    ]    
    stage_epochs_last = np.sum(np.isin(hypno[int(len(hypno) - period/30):], stages))
    stage_duration_last = stage_epochs_last * 30 / 60  # Assuming 30s epochs
    
    last_density = (len(sw_data_last)/len(sw_ch))/stage_duration_last
    sw_early_late_density =  first_density/last_density
    return sws_amp_cv, first_density, last_density


def electrode_side(c):
    if c[-1] == 'z':
        return 'mid'
    elif int(c[-1]) % 2 == 1:
        return 'left'
    elif int(c[-1]) % 2 == 0:
        return 'right'
    return None

# Pre-loading BDFs: read and format channel names, resample, filter, reoder, make raws array
if load_data or not ('raws' in globals() or 'raws' in locals()):
    raws = []; refs = []; refs_ch = []; accs = []; ecgs = []; eogs = []; miscs = [];  
    raws_ori = []; dts = []
    for index, key in enumerate(sleeps):
        if sleeps[key]['file'].endswith('edf') or sleeps[key]['file'].endswith('EDF'):
            raw = mne.io.read_raw_edf(sleeps[key]['file'], preload=True, verbose=True)
        else:
            raw = mne.io.read_raw_bdf(sleeps[key]['file'], preload=True, verbose=True)
        dts.append(raw.info['meas_date'])
        mapping = {ch: ch.replace(' ', '') for ch in raw.ch_names}
        raw.rename_channels(mapping)

        ch = raw.ch_names.copy()
                
        # classify channels by types: eog, emg, ecg, accelerometer
        eog = [elem for elem in ch if elem in eog_ch]
        if len(eog) > 0:
            ch_types = {}; 
            for eog_c in eog:
                ch_types[eog_c] = 'eog'
            raw.set_channel_types(ch_types)
        
        emg = [elem for elem in ch if elem in emg_ch]
        if len(emg) > 0:
            ch_types = {}; 
            for emg_c in emg:
                ch_types[emg_c] = 'emg'
            raw.set_channel_types(ch_types)

        ecg = [elem for elem in ch if elem in ecg_ch]
        if len(ecg) > 0:
            ch_types = {}; 
            for ecg_c in ecg:
                ch_types[ecg_c] = 'ecg'
            raw.set_channel_types(ch_types)

        acc = [elem for elem in ch if elem in acc_ch]
        if len(acc) > 0:
            ch_types = {}; 
            for acc_c in acc:
                ch_types[acc_c] = 'misc'
            raw.set_channel_types(ch_types)
        
        ch_types = {}; 
        
        # custom channels, which were recorded but excluded from analysis explicitly
        misc = [elem for elem in ch if elem in misc_ch]
        for misc_c in misc:
            if misc_c in raw.ch_names:
                ch_types[misc_c] = 'misc'
        if len(ch_types) > 0: 
            raw.set_channel_types(ch_types)
        
        # eeg channels will be all channels except non_eeg_ch
        non_eeg_ch = ecg + acc + eog + emg + misc

        # extract references for each channel
        electrodes = []; ch_refs = []
        for s in ch:
            ch_split = s.split('-')
            if len(ch_split) > 1:
                electrodes.append(ch_split[0])
                ch_refs.append(ch_split[1])
        if re_ref:
            ref = ch_refs[0]
        
            # rename channels so the name is not include reference, only channel
            ch = [x.replace('-'+ref, '') for x in ch]
            raw.rename_channels(dict(zip(raw.ch_names, ch)))
            raw.add_reference_channels(ref)
            ch.append(ref)

        # finally, build eeg channels list
        eeg_ch = [c for c in ch if c not in non_eeg_ch]
                
        # apply notch and bandpass filters
        if nf is not None:
            raw.notch_filter(freqs=nf[0], notch_widths=nf[1], picks = eeg_ch, n_jobs=nj)
        raw.filter(bpf[0], bpf[1], picks = eeg_ch, n_jobs=nj)
        
        # apply eog filters
        if len(eog) > 0:
            if nf is not None:
                raw.notch_filter(freqs=nf[0], notch_widths=nf[1], picks = eog, n_jobs=nj)
            raw.filter(eog_bpf[0], eog_bpf[1], picks = eog, n_jobs=nj)
        
        # apply emg filters
        if len(emg) > 0:
            if nf is not None:
                raw.notch_filter(freqs=nf[0], notch_widths=nf[1], picks = emg, n_jobs=nj)
            raw.filter(emg_bpf[0], emg_bpf[1], picks = emg, n_jobs=nj)
        
        # resample if needed
        if sf_to != raw.info['sfreq']:
            raw.resample(sfreq=sf_to)
            
        if 'Spectrum_YASA' in plots:
            for spect_ch in eeg_ch + eog + emg + ecg:
                sig = raw.get_data(spect_ch, units='uV')
                png_file = f"{raw.info['meas_date'].strftime(cfg['file_dt_format'])} yasa spect {spect_ch}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)
                fig = yasa.plot_spectrogram(sig[0], raw.info['sfreq'], None, trimperc=2.5)
                plt.title(f'#{raw.info["meas_date"]} {spect_ch} YASA Spectrogram')
                if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

        
        raws_ori.append(raw.copy())
        # copy raw and make final changes in a copy
        raw_c = raw.copy()
        
        if re_ref:
            # split channels each side into separate array
            left = []; right = []; mid = []
            for c in eeg_ch:
                print(c)
                if electrode_side(c) == 'mid':
                    mid.append(c)
                elif electrode_side(c) == 'left':
                    left.append(c)
                elif electrode_side(c) == 'right':
                    right.append(c)
    
            # if reference located at one of sides (e.g. T3 or T4), 
            # we will re-refence channels to opposite reference (e.g. if ref is at T3, 
            # then F3-T3 will be re-referenced to F3-T4)
            # as result left hemisphere will be refenced to right, right hemisphere to left
            # if reference is at center (e.g. Fz) then no re-reference will be done
            ch_eeg_refs = {}
            if electrode_side(ref) == 'left':
                right_ref = ref[:-1] + str(int(ref[-1])+1)
                left_ref = ref
                raw_c = mne.set_bipolar_reference(
                    raw_c, left,  [right_ref] * len(left))
                ch_eeg = [x.replace('-'+right_ref, '') for x in raw_c.ch_names if x not in non_eeg_ch]
                ch_eeg_to_rename = [x for x in raw_c.ch_names if x not in non_eeg_ch]
                raw_c.rename_channels(dict(zip(ch_eeg_to_rename, ch_eeg)))
            elif electrode_side(ref) == 'right':
                left_ref = ref[:-1] + str(int(ref[-1])-1)
                right_ref = ref
                raw_c = mne.set_bipolar_reference(
                    raw_c, right,  [left_ref]  * len(right))             
                ch_eeg = [x.replace('-'+left_ref, '') for x in raw_c.ch_names if x not in non_eeg_ch]
                ch_eeg_to_rename = [x for x in raw_c.ch_names if x not in non_eeg_ch]
                raw_c.rename_channels(dict(zip(ch_eeg_to_rename, ch_eeg)))
            elif electrode_side(ref) == 'mid':
                left_ref = ref
                right_ref = ref
                ch_eeg = [x.replace('-'+ref, '') for x in raw_c.ch_names if x not in non_eeg_ch]
                ch_eeg_to_rename = [x for x in raw_c.ch_names if x not in non_eeg_ch]
                raw_c.rename_channels(dict(zip(ch_eeg_to_rename, ch_eeg)))
            for l in left: ch_eeg_refs[l] = right_ref
            for r in right: ch_eeg_refs[r] = left_ref
            for m in mid: ch_eeg_refs[m] = ref
            ch_eeg_with_ref = eeg_ch
            del eeg_ch

        # apply montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw_c.set_montage(ten_twenty_montage , match_case=False, on_missing="ignore")
        
        # make arrays with raws and all channel types vocabulary arrays
        raws.append(raw_c)
        
        accs.append(acc)
        ecgs.append(ecg)
        eogs.append(eog)
        miscs.append(misc)        
        if not re_ref:
            ch_eeg_refs = {}
            for e, ech in enumerate(eeg_ch):
                ch_split = ech.split('-')
                ch_eeg_refs[ech] =  ch_split[1]
            refs.append(None)
            ch_eeg_refs_exist = {k: v for k, v in ch_eeg_refs.items() if k in eeg_ch}
        else:
            refs.append(ref)
            ch_eeg_refs_exist = {k: v for k, v in ch_eeg_refs.items() if k in ch_eeg}
        refs_ch.append(ch_eeg_refs_exist)

if load_hypno or not ('hypnos' in globals() or 'hypnos' in locals()):
    hypnos = []; probs = []; hypnos_max = []; hypnos_adj = []
    hypno_dfs = []; sleep_stats_infos = []; acc_aggs = []
if load_sp_sw or not ('sps' in globals() or 'sps' in locals() or 'sws' in globals() or 'sws' in locals()):
    sps = []; sws =[]; sp_metrics=[]; sw_metrics=[]

for index, raw in enumerate(raws):
    # list of eeg channels names without ref
    eeg_ch_names = list(refs_ch[index].keys())
    if re_ref: eeg_ch_names.remove(refs[index])

    # Process accelerometer
    acc_agg = None
    if len(acc) == n_acc:
        raw  = raws[index].copy().pick(accs[index])
        acc_signal = raw.get_data(acc) 
        # g = x^2 + y^2 + z^2
        acc_g = np.sqrt(np.sum(np.array(acc_signal)**2, axis=0))
        # https://www.researchgate.net/publication/264503253_Estimation_of_Force_during_Vertical_Jumps_using_Body_Fixed_Accelerometers
        # A 4th order Butterworth filter with a cut-off of 10 Hz was applied to smooth the accelerometer signals and the force platform traces[16]. A cut off frequency of 10 Hz was shown to be the best cut off frequency when analysing accelerometer data[17].
        acc_g_bp = butter_lowpass_filter(acc_g, 10, raw.info['sfreq'], order=4)
        # downsample to 100Hz
        acc_sf = 100; acc_sf_k = raw.info['sfreq'] / acc_sf
        acc_g_bp_ls = sc_interp(acc_g_bp, round(len(acc_g_bp) / acc_sf_k))
        
        acc_df = pd.DataFrame({'g': acc_g_bp_ls, 'dt': [dts[0] + timedelta(seconds=i/acc_sf) for i in range(len(acc_g_bp_ls))]})
        acc_df.set_index('dt', inplace=True)

        # aggregate by 10s
        acc_agg = acc_df.resample('10S').mean()
        acc_agg = acc_agg.reset_index()

        # calc diff
        acc_agg['g_diff'] = acc_agg['g'].diff()
        acc_agg = acc_agg.dropna()
        acc_agg['g_diff_norm'] = (acc_agg['g_diff'] - acc_agg['g_diff'].mean()) / np.std(acc_agg['g_diff'])
        acc_agg['g_diff_norm_abs'] = abs(acc_agg['g_diff_norm'])        
        acc_aggs.append(acc_agg)
    
    if load_hypno or (len(hypnos) < 1):
        raw  = raws[index].copy().pick(eeg_ch_names)
        hypnos_up_ch = []; hypnos_ch = []; probs_ch = []
        for ch_index, ch in enumerate(eeg_ch_names):
            sls = yasa.SleepStaging(raw, eeg_name=ch)
            prob = sls.predict_proba()
            probs_ch.append(prob)
            hypno_pred = sls.predict()  # Predict the sleep stages
            hypnos_ch.append(hypno_pred)
            hypno_pred_ch = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
        probs.append(probs_ch)
        hypnos.append(hypnos_ch)

        # N1 = 0, N2 = 1, N3 = 2, R =3, W = 4
        # max stage by avg probability from all channels
        probs_consensus = np.array(probs_ch).sum(axis=0)/len(probs_ch) 
        # W = 0, N1 = 1, N2 = 2, N3 = 3, R = 4
        probs_consensus = probs_consensus.argmax(axis=1) + 1
        probs_consensus[probs_consensus == 5] = 0
        hypnos_max.append(probs_consensus)
        
        # max probability stages for each channel
        hyp_consensus = np.array(probs_ch).argmax(axis=2) + 1
        hyp_consensus[hyp_consensus == 5] = 0

        # consensus with rem/n3 detected by at least single channel
        probs_adj_consensus = probs_consensus.copy()
        rem_occurences = np.sum(hyp_consensus==4, axis=0)
        n3_occurences = np.sum(hyp_consensus==3, axis=0)

        # at least 1/4 of channels sure that there is REM, then REM
        probs_adj_consensus[rem_occurences > round(len(raws[index].ch_names)/4)] = 4
        # at least single channel is enough to say that there is N3
        probs_adj_consensus[n3_occurences > 0 ] = 3

        # replace single awake (next and previous not awake) epochs with previous stage
        if smooth_arousal:
            for h in range(1, len(probs_adj_consensus)-1):
                if probs_adj_consensus[h] == 0 and probs_adj_consensus[h-1] != 0 and probs_adj_consensus[h+1] != 0:
                    probs_adj_consensus[h] = probs_adj_consensus[h-1]  # Replace with the previous stage

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
        sp = yasa.spindles_detect(raw, include=(2), hypno=yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raw_c))
        sps.append(sp)

        if not (1 == np.sum(np.isin(sp_ch, eeg_ch_names))/len(sp_ch)):
            sp_ch = [eeg_ch_names[0]]
        
        sp_metric = spindle_metrics(sp, sleep_stats_infos[index]['SOL_ADJ'], hypnos_adj[index], sp_ch=sp_ch, stages=[2], period=4.5*3600)
        sp_metrics.append(sp_metric)

        sw = yasa.sw_detect(raw, include=(2,3), hypno=yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raws[index]))
        sws.append(sw)
        
        if not (1 == np.sum(np.isin(sw_ch, eeg_ch_names))/len(sw_ch)):
            sw_ch = [eeg_ch_names[0]]
        sw_metric = sws_metrics(sw, sleep_stats_infos[index]['SOL_ADJ'], hypnos_adj[index], sw_ch=sw_ch, stages=[2,3], period=4.5*3600)
        sw_metrics.append(sw_metric)
    
if 'Hypno' in plots:
    for index, raw in enumerate(raws):
        eeg_ch_names = list(refs_ch[index].keys())
        if re_ref: eeg_ch_names.remove(refs[index])
        raw  = raws[index].copy().pick(eeg_ch_names)

        fig, axes = plt.subplots(round(len(eeg_ch_names))+2, 
                  figsize=(8, len(eeg_ch_names)*2))
        fig.suptitle(f'#{raw.info["meas_date"]} Multitaper spectrogram, {spect_specs}')
        hyp = yasa.hypno_int_to_str(hypnos_max[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[0])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Max Probs)')
        hyp = yasa.hypno_int_to_str(hypnos_adj[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[1])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} (Adj Probs)')
        for ch_index, ch in enumerate(eeg_ch_names):
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
                hypno_df = hypno_dfs[index]
                sleep_stats_info = sleep_stats_infos[index]
                sol_adj = sleep_stats_info['SOL_ADJ']
                tst_adj = sleep_stats_info['TST_ADJ']
                waso_adj = sleep_stats_info['WASO_ADJ']
                hypno_stages = {}
                hrv_exclude_quality = True
                hrv_exclude_acc = True
                ecg_invert = -1 if sleeps[str(index+1)]['ecg_invert'] else 1
    
                # get major movements from accelerometer data            
                major_acc_epoch = None
                if len(acc) > 0:
                    acc_agg = acc_aggs[index]
                    if acc_agg is not None:
                        acc_agg['dtr'] = acc_agg['dt'].dt.round('30s')
                        hypno_acc = pd.merge(acc_agg,  hypno_df, on = "dtr", how = 'left')
                        hypno_acc['dt'] = hypno_acc['dt_x']
                        acc_th = 2 # in standard deviations
                        # major_acc_nonwake_epoch = np.unique(hypno_acc[(hypno_acc['g_diff_norm_abs'] > acc_th) & (hypno_acc['h'] != 0)]['dtr'])
                        major_acc_epoch = np.unique(hypno_acc[hypno_acc['g_diff_norm_abs'] > acc_th]['dtr'])
                        major_acc_epoch = major_acc_epoch[major_acc_epoch > (dts[0] + timedelta(seconds=sol_adj*60))]
                        len(major_acc_epoch)
    
                # process ECG to HR
                window = 15; slide = 5; metrics = None
                hr_col = f'_{window}s'
                hrv_cache_tag = f'hrv_p{window}_s{slide}'; 
                hrv_file = f"{hrv_cache_tag}-{user}-{dts[index].strftime(cfg['file_dt_format'])}.csv"
                hrv_filepath = os.path.join(cfg['cache_dir'], hrv_file)
                if not os.path.isfile(hrv_filepath):
                    hr = hrv_process(raw.get_data(ecgs[index], units='uV')[0]*ecg_invert, sf = round(raw.info['sfreq']), 
                        window = window, slide = slide, user = user, device = device, 
                        dts = dts[index], metrics = metrics, cache_dir = cfg['cache_dir'])
                else:
                    hr = pd.read_csv(hrv_filepath)
    
                # process ECG to HRV
                windows = [60]; slides = [20]; metrics = ['time','freq','ans','r_rr', 'nl']
                for iw, win in enumerate(windows):
                    print(f'{iw} {win}')
                    window = windows[iw]; slide = slides[iw]
                    hrv_col = f'_{window}s'
                    hrv_cache_tag = f'hrv_p{window}_s{slide}'; 
                    hrv_file = f"{hrv_cache_tag}-{user}-{dts[index].strftime(cfg['file_dt_format'])}.csv"
                    hrv_filepath = os.path.join(cfg['cache_dir'], hrv_file)
                    if not os.path.isfile(hrv_filepath):
                        hrv = hrv_process(raw.get_data(ecgs[index], units='uV')[0]*ecg_invert, sf = round(raw.info['sfreq']), 
                            window = window, slide = slide, user = user, device = device, 
                            dts = dts[index], metrics = metrics, cache_dir = cfg['cache_dir'])
                    else:
                        hrv = pd.read_csv(hrv_filepath)
                        
                    sleep_end = np.max(hypno_df['dt'])
                    hr_len = len(hr); hrv_len = len(hrv)
                    hr_q_art = 0; hrv_q_art = 0
                    hr_acc_art = 0; hrv_acc_art = 0
                    if hrv_exclude_quality:
                        r4_cor_th = .75; r4_cor_th_hr = .66; r3_th = 2.2
                        hrv = hrv_quality(hrv, r3_th = r3_th, r4_cor_th = r4_cor_th)
                        hr = hrv_quality(hr, r3_th = r3_th, r4_cor_th = r4_cor_th_hr)
                        hr_q_art = np.sum(~hr['q'])
                        hr = hr[hr['q']]
                        hrv_q_art = np.sum(~hrv['q'])
                        hrv = hrv[hrv['q']]
                    
                    hr['dt'] = pd.to_datetime(hr['dt'])
                    hr['dtr'] = hr['dt'].dt.round('30s')
            
                    hypno_hr = pd.merge(hr, hypno_df, on = "dtr", how = 'left')
                    hypno_hr['dt'] = hypno_hr['dt_x']
            
                    hrv['dt'] = pd.to_datetime(hrv['dt'])
                    hrv['dtr'] = hrv['dt'].dt.round('30s')
            
                    hrv[f'lfhf{hrv_col}'] = hrv[f'lf{hrv_col}']/hrv[f'hf{hrv_col}']
                    hrv[f'anomaly{hrv_col}'] = hrv['ectopic'] + hrv['missed'] + hrv['extra'] + hrv['longshort']
            
                    hypno_hrv = pd.merge(hrv,  hypno_df, on = "dtr", how = 'left')
                    hypno_hrv['dt'] = hypno_hrv['dt_x']
                    
                    acc_title = ''; movements_per_hour = 0
                    if major_acc_epoch is not None:
                        movements_per_hour = round(len(major_acc_epoch)/((tst_adj + waso_adj)/60),2)
                        acc_title = f"""M{len(major_acc_epoch)} / MH{movements_per_hour}"""
                        if hrv_exclude_acc:
                            hr_acc_art = np.sum(hr['dtr'].isin(major_acc_epoch))
                            hrv_acc_art = np.sum(hrv['dtr'].isin(major_acc_epoch))
                            hr = hr[~hr['dtr'].isin(major_acc_epoch)]
                            hrv = hrv[~hrv['dtr'].isin(major_acc_epoch)]
                    
                    hrv_stages = {}
                    hr_stages = {}
                    sleep_cycles_limit = 2 * (3600 * 1.5)
                    # N3=3, N2=2, N1=1, R=4, W=0
                    hypno_stages['n3'] = hypno_df[(hypno_df['h'] == 3)]
            
                    def extract_n3(hypno, sol, sleep_cycles_limit):
                        prev = 0
                        n3_skip = 20 # how much 30s epochs to skip at end of period
                        n3_min_len = 10 # min len for internal n3 period
                        n3_df = []
                        for ind, hyp_epoch in hypno.iterrows():
                            if hyp_epoch['cumtime'] < (float(sol)*60 + sleep_cycles_limit):
                                n3 = hyp_epoch['h'] == 3 # N3=3
                                if prev != hyp_epoch['h']: # stage change
                                    if n3:
                                        for ind_in, epoch_in in hypno.iloc[ind:].iterrows():
                                            n3_in = epoch_in['h'] == 3 # N3=3
                                            if not n3_in:
                                                break
                                        n3_len = ind_in-ind
                                        # print(f'n3: {n3_len}')
                                        if n3_len >= (n3_skip + n3_min_len):
                                            n3_df.append(hypno.iloc[ind:(ind_in - n3_skip)])
                            prev = hyp_epoch['h']
                        if len(n3_df) > 0:
                            return pd.concat(n3_df, axis=0)
                        return None
                
                    hypno_stages['n3_hrv'] = extract_n3(hypno_df, sol_adj, sleep_cycles_limit)
                    if hypno_stages['n3_hrv'] is not None:
                        # N3=3, N2=2, N1=1, R=4, W=0
                        hypno_stages['r'] = hypno_df[(hypno_df['h'] == 4)]
                        hypno_stages['n2'] = hypno_df[(hypno_df['h'] == 2)]
                        hypno_stages['w'] = hypno_df[(hypno_df['h'] == 0)]
            
                        hrv_stages['n3'] = pd.merge(hrv, hypno_stages['n3_hrv'], on = "dtr", how = 'inner')
                        hrv_stages['r'] = hypno_hrv[(hypno_hrv['h'] == 4)]
                        hrv_stages['n2'] = hypno_hrv[(hypno_hrv['h'] == 2)]
                        hrv_stages['w'] = hypno_hrv[(hypno_hrv['h'] == 0)]
                        hrv_stages['tib'] = hypno_hrv[(hypno_hrv['h'].isin([0,1,2,3,4]))]
                        hrv_stages['tst'] = hypno_hrv[(hypno_hrv['h'].isin([1,2,3,4]))]
                        
                        hr_stages['n3'] = pd.merge(hr, hypno_stages['n3_hrv'], on = "dtr", how = 'inner')
                        hr_stages['r'] = hypno_hr[(hypno_hr['h'] == 4)]
                        hr_stages['n2'] = hypno_hr[(hypno_hr['h'] == 2)]
                        hr_stages['w'] = hypno_hr[(hypno_hr['h'] == 0)]
                        hr_stages['tib'] = hypno_hr[(hypno_hr['h'].isin([0,1,2,3,4]))]
                        hr_stages['tst'] = hypno_hr[(hypno_hr['h'].isin([1,2,3,4]))]
                        
                        # h_min = np.percentile(hypno_hr['rmssd'],10); h_max = np.percentile(hypno_hrv['rmssd'],90)
                        h_min = 20; h_max = 80
                        hypno_hrv.set_index('dt', inplace=True)
                        hypno_hrv['rolling_rmssd'] = hypno_hrv[f'rmssd{hrv_col}'].rolling(window='5min').mean()
                        hypno_hrv = hypno_hrv.reset_index()
                        
                        hypno_hr.set_index('dt', inplace=True)
                        hypno_hr['rolling_hr'] = hypno_hr[f'hr{hr_col}'].rolling(window='5min').mean()
                        hypno_hr = hypno_hr.reset_index()
                        
                        if len(hrv_stages['n3']) > 0:
                                                    
                            hr_t = {
                                'n3': round(np.mean(hr_stages['n3'][f'hr{hr_col}']),1),
                                'n3_sd': round(np.std(hr_stages['n3'][f'hr{hr_col}']),1),
                                'n2': round(np.mean(hr_stages['n2'][f'hr{hr_col}']),1),
                                'n2_sd': round(np.std(hr_stages['n2'][f'hr{hr_col}']),1),
                                'r': round(np.mean(hr_stages['r'][f'hr{hr_col}']),1),
                                'r_sd': round(np.std(hr_stages['r'][f'hr{hr_col}']),1),
                                'tst': round(np.mean(hr_stages['tst'][f'hr{hr_col}']),1),
                                'tst_sd': round(np.std(hr_stages['tst'][f'hr{hr_col}']),1)
                            }
                            rmssd_t = {
                                'n3': round(np.mean(hrv_stages['n3'][f'rmssd{hrv_col}'])),
                                'n3_sd': round(np.std(hrv_stages['n3'][f'rmssd{hrv_col}']),1),
                                'n2': round(np.mean(hrv_stages['n2'][f'rmssd{hrv_col}'])),
                                'n2_sd': round(np.std(hrv_stages['n2'][f'rmssd{hrv_col}']),1),
                                'r': round(np.mean(hrv_stages['r'][f'rmssd{hrv_col}'])),
                                'r_sd': round(np.std(hrv_stages['r'][f'rmssd{hrv_col}']),1),
                                'tst': round(np.mean(hrv_stages['tst'][f'rmssd{hrv_col}'])),
                                'tst_sd': round(np.std(hrv_stages['tst'][f'rmssd{hrv_col}']),1)
                            }
                            lfhf_t = {
                                'n3': round(np.mean(hrv_stages['n3'][f'lfhf{hrv_col}']),1),
                                'n3_sd': round(np.std(hrv_stages['n3'][f'lfhf{hrv_col}']),1),
                                'n2': round(np.mean(hrv_stages['n2'][f'lfhf{hrv_col}']),1),
                                'n2_sd': round(np.std(hrv_stages['n2'][f'lfhf{hrv_col}']),1),
                                'r': round(np.mean(hrv_stages['r'][f'lfhf{hrv_col}']),1),
                                'r_sd': round(np.std(hrv_stages['r'][f'lfhf{hrv_col}']),1),
                                'tst': round(np.mean(hrv_stages['tst'][f'lfhf{hrv_col}']),1),
                                'tst_sd': round(np.std(hrv_stages['tst'][f'lfhf{hrv_col}']),1)
                            }
                            
                            abnormal_title = f"A{round(np.sum(hrv_stages['tst'][f'anomaly{hrv_col}'])/(tst_adj/60))} M{round(np.sum(hrv_stages['tst'][f'missed'])/(tst_adj/60),1)} E{round(np.sum(hrv_stages['tst'][f'extra'])/(tst_adj/60),1)} Ec{round(np.sum(hrv_stages['tst'][f'ectopic'])/(tst_adj/60),1)} L{round(np.sum(hrv_stages['tst'][f'longshort'])/(tst_adj/60),1)}"
                            
                            art_title = ''
                            if hrv_exclude_acc or hrv_exclude_quality:
                                art_title = f'({round(100*(hrv_acc_art + hrv_q_art) / hrv_len)}% / {round(100*(hr_acc_art + hr_q_art) / hr_len)}%)'
                            
                            title = f"""{dts[0].strftime(cfg['plot_dt_format'])} {acc_title} | OBCI {ecgs[index]} {art_title} p{hrv_col}
            HR {hr_t['tst']}±{hr_t['tst_sd']} N3 {hr_t['n3']}±{hr_t['n3_sd']} R {hr_t['r']}±{hr_t['r_sd']}
            RMSSD {rmssd_t['tst']}±{rmssd_t['tst_sd']} N3 {rmssd_t['n3']}±{rmssd_t['n3_sd']} R {rmssd_t['r']}±{rmssd_t['r_sd']}
            L/H {lfhf_t['tst']}±{lfhf_t['tst_sd']} N3 {lfhf_t['n3']}±{lfhf_t['n3_sd']} R {lfhf_t['r']}±{lfhf_t['r_sd']}
            {abnormal_title}"""
            
                            ecg_stats.append({
                                'hr': hr_t['tst'],
                                'rmssd': rmssd_t['tst'],
                                'rmssd_n3': rmssd_t['n3'],
                                'mh': movements_per_hour
                                })
                            old_fontsize = plt.rcParams["font.size"]
                            plt.rcParams.update({"font.size": 15})
                            fig, ax = plt.subplots(1,1, figsize=(9,7))
                            plt.title(title)
                            p_size = 3
                            plt.scatter(hypno_hrv['dt'], hypno_hrv[f'rmssd{hrv_col}'], c='springgreen', s=p_size)
                            plt.plot(hypno_hrv['dt'], hypno_hrv['rolling_rmssd'], c='green', linewidth=2)
                            p_size = 4
                            plt.scatter(hrv_stages['n3']['dt_x'], hrv_stages['n3'][f'rmssd{hrv_col}'], c='violet', s=p_size)
                            plt.scatter(hrv_stages['r']['dt_x'], hrv_stages['r'][f'rmssd{hrv_col}'], c='red', s=p_size)
                            p_size = 8
                            plt.scatter(hypno_stages['n3']['dt'], np.repeat(h_min, len(hypno_stages['n3'])), c='blueviolet', s=p_size, marker='s')
                            plt.scatter(hypno_stages['n2']['dt'], np.repeat(h_min + (h_max-h_min)/3, len(hypno_stages['n2'])), c='dodgerblue', s=p_size, marker='s')
                            plt.scatter(hypno_stages['r']['dt'], np.repeat(h_min + 2*(h_max-h_min)/3, len(hypno_stages['r'])), c='red', s=p_size, marker='s')
                            plt.scatter(hypno_stages['w']['dt'], np.repeat(h_max, len(hypno_stages['w'])), c='orange', s=p_size, marker='s')
                            if acc_agg is not None:
                                p_size = 2
                                plt.scatter(major_acc_epoch, np.repeat(78, len(major_acc_epoch)), c = 'blue', s=p_size, marker='s')
                                plt.plot(acc_agg['dt'], acc_agg['g_diff_norm_abs']*5, linewidth=1)
                            plt.axvline(dts[0] + timedelta(seconds=float(sol_adj)*60), c='grey', linestyle='--', linewidth=.5)
                            plt.axvline(dts[0] + timedelta(seconds=float(sol_adj)*60 + sleep_cycles_limit), c='grey', linestyle='--', linewidth=.5)
                            hrv_lim = [10, 100]; ax.set_ylim(hrv_lim[0],hrv_lim[1])
                            ax2 = ax.twinx()
                            ax2.plot(hypno_hr['dt'], hypno_hr['rolling_hr'], c='red', linewidth=1.5)
                            ax2.tick_params(axis='y', labelsize=13)
                            ax2.axhline(50, c='red', linestyle='--', linewidth=.5)
                            hr_lim = [30, 90]; ax2.set_ylim(hr_lim[0],hr_lim[1])
                            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            ax.set_yticks(np.arange(0, hrv_lim[1]+10, 10))
                            ax2.set_yticks(np.arange(hr_lim[0], max([hr_lim[1], 5*round(max(hypno_hr['rolling_hr'])/5+0.5)])+5, 5))
                            ax.tick_params(axis='x', labelsize=13)  # Set the x-axis ticks font size to 12
                            ax.tick_params(axis='y', labelsize=13)  # Set the y-axis ticks font size to 12
                            ax.spines['top'].set_visible(False)
                            ax2.spines['top'].set_visible(False)
                            plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01) 
                            plt.tight_layout()
                            plt.rcParams.update({"font.size": old_fontsize})
                            png_file = f"{dts[index].strftime(cfg['file_dt_format'])} hrv {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
                            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

stages = ['W','N1','N2','N3','R']
stages_plot = [0,4,2,3]
bp_bands = [
    (1, 4, "Delta"),
    (4, 6, "Theta 1"),
    (6, 8, "Theta 2"),
    (8, 10, "Alpha 1"),
    (10, 12, "Alpha 2"),
    (12, 30, "Beta"),
    (30, 48, "Gamma"),
]

if load_bp or not ('bps' in globals() or 'bps' in locals()):
    bps = []
    raws_bp = []
    for index, raw in enumerate(raws):
        eeg_ch_names = list(refs_ch[index].keys())
        raw_bp  = raws[index].copy().pick(eeg_ch_names)
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw_bp.set_montage(ten_twenty_montage , match_case=False, on_missing="ignore")

        if topo_ref == 'REST':
            sphere = mne.make_sphere_model("auto", "auto", raw.info)
            src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=5.0)
            forward = mne.make_forward_solution(raw_bp.info, trans=None, src=src, bem=sphere)
            raw_bp.set_eeg_reference("REST", forward=forward)
        elif topo_ref == 'AR':
            raw.set_eeg_reference(ref_channels = 'average')

        raws_bp.append(raw_bp)
        bps_s = []
        for s in range(len(stages)):
            hypno_up = yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raws[index])
            bandpower = yasa.bandpower(raw_bp, hypno=hypno_up, include=(s), bands=bp_bands, relative=bp_relative)
            bp_b = []
            for b in range(len(bp_bands)):
                bp = np.sqrt(bandpower.xs(s)[bp_bands[b][2]])
                bp_b.append(bp)
            bps_s.append(bp_b)
        bps.append(bps_s)

if 'Topomap' in plots:
    for index, raw in enumerate(raws):
        eeg_ch_names = list(refs_ch[index].keys())
        raw  = raws[index].copy().pick(eeg_ch_names)
        if len(eeg_ch_names) > 2:
            if not re_ref:
                mapping = {ch: ch.split("-")[0] for ch in raw.ch_names if "-" in ch}
                print(f'rename {mapping}')
                raw.rename_channels(mapping)
                ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(ten_twenty_montage , match_case=False)
                raws_bp[index].rename_channels(mapping)
                raws_bp[index].set_montage(ten_twenty_montage , match_case=False)
    
            fig, axes = plt.subplots(len(stages_plot),len(bp_bands), 
                     figsize=(len(bp_bands)*2, len(stages_plot)*2))
            plot_type = f'{raws_bp[index].info["meas_date"]} Amplitude (ref={topo_ref})'; plot_params = ''
            hypno_up = yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raws[index])
            for s_index, s in enumerate(stages_plot):
                for b in range(len(bp_bands)):
                    bp = bps[index][s][b]
                    if not bp_relative:
                        p_max = np.max(bp)
                        p_min = np.min(bp)
                    else:
                        p_max = max(np.array(bps[index]).max(axis=2)[...,b])
                        p_min = min(np.array(bps[index]).min(axis=2)[...,b])*1.2
                    vlim = (p_min,p_max)
                    ax = axes[s_index,b]
                    im, _ = mne.viz.plot_topomap(
                        bp, 
                        raws_bp[index].info,
                        cmap=cm.jet,
                        axes=ax,
                        vlim=vlim,
                        show=False)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, format='%0.1f', ticks = [vlim[0], vlim[1]], aspect=10)
                    cbar.ax.set_position([0.85, 0.1, 0.05, 0.8])
                    cbar.set_label(units['rel'])
                    bl = f'{bp_bands[b][2]} ({bp_bands[b][0]} - {bp_bands[b][1]} Hz)'
                    if b == 0:
                        ax.set_title(f'{stages[s]} {bl}')
                    elif b == 1:
                        ax.set_title(f'{bl}')
                    else:
                        ax.set_title(f'{bl}')
            fig.suptitle(f'{plot_type} ({sig_specs}, {topo_method}=[{plot_params}]')
            plt.tight_layout()
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
    
    ch_order = ['F', 'C', 'O', 'T', 'A']
    order_map = {letter: l_index for l_index, letter in enumerate(ch_order)}
    def order_value(string):
        first_letter = string[0]
        return order_map.get(first_letter, float('inf'))  # Use inf for letters not in custom_order

    if load_spect or not ('spects_l' in globals() or 'spects_l' in locals()):
        spects_l = []; stimes_l = []; sfreqs_l = []
        for index, raw in enumerate(raws):
            spects_c = []; stimes_c = []; sfreqs_c = []
            if re_ref: eeg_ch_names.remove(refs[index])
            eeg_ch_names = list(refs_ch[index].keys())
            ch_eeg_sorted = sorted(eeg_ch_names, key=order_value)

            raw  = raws[index].copy().pick(ch_eeg_sorted)

            for ch_index, ch in enumerate(ch_eeg_sorted):
                spect, stimes, sfreqs = multitaper_spectrogram(
                    raw.get_data(picks = [ch], units='uV'), raw.info['sfreq'], 
                    frequency_range, time_bandwidth, num_tapers, 
                    window_params, min_nfft, detrend_opt, multiprocess, cpus,
                    weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
                spects_c.append(spect)
                stimes_c.append(stimes)
                sfreqs_c.append(sfreqs)
            spects_l.append(spects_c)
            stimes_l.append(stimes_c)
            sfreqs_l.append(sfreqs_c)

    for index, raw in enumerate(raws):        
        eeg_ch_names = list(refs_ch[index].keys())
        if re_ref: eeg_ch_names.remove(refs[index])
        raw  = raws[index].copy().pick(eeg_ch_names)
        
        # summary spectrum
        fig, ax = plt.subplots(figsize=(14, 5))
        old_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({"font.size": 18})
        fig.suptitle(f'#{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} Spectrogram, {spect_specs}')

        spect, stimes, sfreqs = np.array(spects_c).max(axis=0), stimes_c[0], sfreqs_c[0]
        spect_data = nanpow2db(spect)
        
        start_time = raw.info['meas_date']
        times = [start_time + timedelta(seconds=int(s)) for s in stimes]
        
        dtx = times[1] - times[0]
        dy = sfreqs[1] - sfreqs[0]
        x_s = mdates.date2num(times[0]-dtx)
        x_e = mdates.date2num(times[-1]+dtx)
        extent = [x_s, x_e, sfreqs[-1]+dy, sfreqs[0]-dy]

        im = ax.imshow(
            spect_data, extent=extent, aspect='auto', 
            cmap=plt.get_cmap('jet'), 
            vmin = spect_vlim[0], vmax = spect_vlim[1],
            )
        ax.xaxis_date()  # Interpret x-axis values as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format time as HH:MM:SS
        ax.invert_yaxis()
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f'All Channels ({sig_specs})')
        tick_intervals = np.linspace(x_s, x_e, 11)  # 11 points include 0% to 100%
        ax.set_xticks(tick_intervals)
        
        h_min = 1.75; h_max = 13
        p_size = 2
        hypno_df = hypno_dfs[index]
        hypno_stages = {}
        # N3=3, N2=2, N1=1, R=4, W=0
        hypno_stages['n3'] = hypno_df[(hypno_df['h'] == 3)]
        hypno_stages['r'] = hypno_df[(hypno_df['h'] == 4)]
        hypno_stages['n2'] = hypno_df[(hypno_df['h'] == 2)]
        hypno_stages['w'] = hypno_df[(hypno_df['h'] == 0)]

        if len(hypno_stages)> 0:
            plt.scatter(hypno_stages['n3']['dt'], np.repeat(h_min, len(hypno_stages['n3'])), c='blueviolet', s=p_size, marker='s')
            plt.scatter(hypno_stages['n2']['dt'], np.repeat(h_min + (h_max-h_min)/2, len(hypno_stages['n2'])), c='blue', s=p_size, marker='s')
            plt.scatter(hypno_stages['r']['dt'], np.repeat(h_min + 2*(h_max-h_min)/3, len(hypno_stages['r'])), c='red', s=p_size, marker='s')
            plt.scatter(hypno_stages['w']['dt'], np.repeat(h_max, len(hypno_stages['w'])), c='orange', s=p_size, marker='s')

        # Scale colormap
        if clim_scale:
            clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
            im.set_clim(clim)  # actually change colorbar scale

        plt.tight_layout()
        plt.rcParams.update({"font.size": old_fontsize})
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05) 
        
        png_file = f"{dts[index].strftime(cfg['file_dt_format'])} merged spectrum {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
        if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
        
        
        n_ax = 2
        n_up = 0 if (len(eeg_ch_names) % 2) == 0 else .5
        n_cycles = round(len(eeg_ch_names)/n_ax + n_up)

        for cy in range(n_cycles):
            fig, axes = plt.subplots(n_ax, 
                      figsize=(14, n_ax*5))
            old_fontsize = plt.rcParams["font.size"]
            plt.rcParams.update({"font.size": 18})
            fig.suptitle(f'#{raw.info["meas_date"].strftime(cfg["plot_dt_format"])} Spectrogram, {spect_specs}')
            axes = axes.flatten()
            idx_range = cy*n_ax + np.arange(0, n_ax, 1)
            for cch_index, cch in enumerate(ch_eeg_sorted[min(idx_range):(max(idx_range)+1)]):
                spect, stimes, sfreqs = spects_c[cy*n_ax + cch_index], stimes_c[cch_index], sfreqs_c[cy*n_ax + cch_index]
                spect_data = nanpow2db(spect)
                
                start_time = raw.info['meas_date']
                times = [start_time + timedelta(seconds=int(s)) for s in stimes]
                
                dtx = times[1] - times[0]
                dy = sfreqs[1] - sfreqs[0]
                x_s = mdates.date2num(times[0]-dtx)
                x_e = mdates.date2num(times[-1]+dtx)
                extent = [x_s, x_e, sfreqs[-1]+dy, sfreqs[0]-dy]
        
                ax = axes[cch_index]
                im = ax.imshow(
                    spect_data, extent=extent, aspect='auto', 
                    cmap=plt.get_cmap('jet'), 
                    vmin = spect_vlim[0], vmax = spect_vlim[1],
                    )
                ax.xaxis_date()  # Interpret x-axis values as dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format time as HH:MM:SS
                ax.invert_yaxis()
                ax.set_xlabel("Time")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title(f'{ch_eeg_sorted[cy*n_ax + cch_index]} - {refs_ch[index][ch_eeg_sorted[cy*n_ax + cch_index]]} ({sig_specs})')
                tick_intervals = np.linspace(x_s, x_e, 11)  # 11 points include 0% to 100%
                ax.set_xticks(tick_intervals)
                
                # Scale colormap
                if clim_scale:
                    clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
                    im.set_clim(clim)  # actually change colorbar scale
    
            plt.tight_layout()
            plt.rcParams.update({"font.size": old_fontsize})
            plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01) 
            
            png_file = f"{dts[index].strftime(cfg['file_dt_format'])} spect {cy} {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
            if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)

# Power frequency plot with mne PSD computation
if 'Features' in plots:
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.flatten()
    for index, key in enumerate(raws_ori):
        eeg_ch_names = list(refs_ch[index].keys())
        if re_ref: eeg_ch_names.remove(refs[index])
        raw  = raws[index].copy().pick(eeg_ch_names)
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
            ax[3].plot(freqs, psds[c], label=f'{eeg_ch_names[c]}-{refs_ch[index][eeg_ch_names[c]]}', linewidth=1)
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
            axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{ch_eeg_refs[amps["Channel"][max_amp]]} in N{amps["Stage"][max_amp]}\nCV: {round(sw_metric[0],2)} Early {round(sw_metric[1],2)} Late {round(sw_metric[2],2)}, E/L {round(sw_metric[1]/sw_metric[2],2)} {sw_ch}')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    png_file = f"{dts[index].strftime(cfg['file_dt_format'])} PSD {user}.png"; png_filename = os.path.join(cfg['image_dir'], png_file)    
    if not os.path.isfile(png_filename) or image_overwrite: fig.savefig(png_filename)
    plt.rcParams.update({"font.size": old_fontsize})

if 'Radar' in plots:
    for index, key in enumerate(raws_ori):
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
