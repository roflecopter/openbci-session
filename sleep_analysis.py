import mne
import yasa
import os
import sys
import seaborn as sns
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from time import sleep
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta

# git pull https://github.com/preraulab/multitaper_toolbox/
multitaper_dir = '/path/to/multitaper_toolbox' 
multitaper_dir = '/Volumes/Data/Storage/Dev/multitaper_toolbox/python'
os.chdir(multitaper_dir)
from multitaper_spectrogram_python import multitaper_spectrogram, nanpow2db

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('future.no_silent_downcasting', True)

file_dt_format = '%Y-%m-%d %H_%M_%S'
plot_dt_format = "%d %b'%y %H:%M"
nj = 15 # n_jobs for multiprocessing, usually n_cpu - 1

old_fontsize = plt.rcParams["font.size"]
plt.rcParams.update({"font.size": 8})

# enter dir and file for bdf file recorded with session_start.py
data_dir = '/path/to/bdf_files_dir'
data_dir = '/Volumes/Data/Storage/Self/!raw-1OBCI/archive'
f_name = os.path.join(data_dir, '2024-05-01_22-47-27-max-OBCI_F9.TXT.bdf')
sleeps = {'1': {'file': f_name}}

# signla filtering
bpf = [.35, 45] # band pass filter, [0.1, None] or [.35, 45]
nf = [50,1] # notch filter, set to 50 or 60 Hz powerline noise freq depending on your country
plots = ['Hypno'] # to plot all use: plots = ['Hypno', 'Features','Spectrum','Topomap']

sf_to = 256 # sampling rate to resample for fast processing
freq_method = 'mne_psd_welch' # 'mne_psd_welch' / 'mne_trf_morlet' / 'mne_psd_multitaper' / 'mne_tfr_multitaper'
topo_method = 'yasa_band_amp' # 'yasa_band_power' / 'mne_trf_morlet' / 'mne_fft_welch'
w_fft = 4; m_bandwidth = 1; m_freq_bandwidth = 2; tfr_time_bandwidth = 4; 
topo_ref = 'REST' # 'REST' / 'AR' rereference type
bp_relative = True # bandpass is relative or abs for topomap

# multitaper spectrograms settings, can leave as is if not sure what is it
spect_vlim = [6,24]
spect_lim = [1,16]
freq_lim = [1,30]
time_bandwidth = 5 # Set time-half bandwidth
num_tapers = time_bandwidth*2 - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
window_params = [90, 45]  # Window size is Xs with step size of Ys

# units for labels
units = {'psd_dB': 'dB(µV²/Hz)', 'amp': 'µV', 'p': 'µV²', 'p_dB': 'dB(µV²)', 'rel': '%'}
sig_specs = f'sf={sf_to}Hz, notch={nf}, bandpass={bpf}'
spect_specs = f'num_tapers={num_tapers}, window={window_params}'

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

units = {'psd_dB': 'dB(µV²/Hz)', 'amp': 'µV', 'p': 'µV²', 'p_dB': 'dB(µV²)', 'rel': '%'}
load_spect = True
load_data = True
load_sp_sw = True
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


def electrode_side(c):
    if c[-1] == 'z':
        return 'mid'
    elif int(c[-1]) % 2 == 1:
        return 'left'
    elif int(c[-1]) % 2 == 0:
        return 'right'
    return None

if load_data or not ('raws' in globals() or 'raws' in locals()):
    raws = []; refs = []; refs_ch = []; raws_ori = []; dts = []
    for index, key in enumerate(sleeps):
        raw = mne.io.read_raw_bdf(sleeps[key]['file'], preload=True, verbose=True)
        dts.append(raw.info['meas_date'])
        # remove '-ref' postfix from OpenBCI channel names and accelerometer channels
        ch = raw.ch_names.copy()
        ch.remove('ACC_X'); ch.remove('ACC_Y'); ch.remove('ACC_Z')
        electrodes = []; refs = []
        for s in ch:
            ch_split = s.split('-')
            if len(ch_split) > 1:
                electrodes.append(ch_split[0])
                refs.append(ch_split[1])
        ref = refs[0]
        ch = [x.replace('-'+ref, '') for x in ch]
        raw.rename_channels(dict(zip(raw.ch_names, ch)))
        raw.add_reference_channels(ref)
        ch.append(ref)
        
        # pick EEG channels
        raw.pick(ch)
        
        # apply notch and bandpass filters
        if nf is not None:
            raw.notch_filter(freqs=nf[0], notch_widths=nf[1], n_jobs=nj)
        raw.filter(bpf[0], bpf[1], n_jobs=nj)
        
        if sf_to != raw.info['sfreq']:
            raw.resample(sfreq=sf_to)
        
        raws_ori.append(raw.copy())
        
        # split channels and re-ref, left hemisphere to right, right to left
        left = []; right = []; mid = []
        for c in ch:
            if electrode_side(c) == 'mid':
                mid.append(c)
            elif electrode_side(c) == 'left':
                left.append(c)
            elif electrode_side(c) == 'right':
                right.append(c)
    
        raw_c = raw.copy()
        ch_final = []
        ch_refs = {}
        if electrode_side(ref) == 'left':
            right_ref = ref[:-1] + str(int(ref[-1])+1)
            left_ref = ref
            raw_c = mne.set_bipolar_reference(
                raw_c, left,  [right_ref] * len(left))
            ch = [x.replace('-'+right_ref, '') for x in raw_c.ch_names]
            raw_c.rename_channels(dict(zip(raw_c.ch_names, ch)))
        elif electrode_side(ref) == 'right':
            left_ref = ref[:-1] + str(int(ref[-1])-1)
            right_ref = ref
            raw_c = mne.set_bipolar_reference(
                raw_c, right,  [left_ref]  * len(right))             
            ch = [x.replace('-'+left_ref, '') for x in raw_c.ch_names]
            raw_c.rename_channels(dict(zip(raw_c.ch_names, ch)))
        for l in left: ch_refs[l] =right_ref
        for r in right: ch_refs[r] =left_ref
        for m in mid: ch_refs[m] = ref

        # apply montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw_c.set_montage(ten_twenty_montage , match_case=False, on_missing="ignore")
        
        raws.append(raw_c)
        refs.append(ref)
        refs_ch.append(ch_refs)

if load_hypno or not ('hypnos' in globals() or 'hypnos' in locals()):
    hypnos = []; probs = []; hypnos_max = []; hypnos_adj = []
if load_sp_sw or not ('sps' in globals() or 'sps' in locals()):
    sps = []; sws =[]

for index, raw in enumerate(raws):
    if load_hypno or (len(hypnos) < 1):
        hypnos_up_ch = []; hypnos_ch = []; probs_ch = []
        for ch_index, ch in enumerate(raws[index].ch_names):
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
        for h in range(1, len(probs_adj_consensus)-1):
            if probs_adj_consensus[h] == 0 and probs_adj_consensus[h-1] != 0 and probs_adj_consensus[h+1] != 0:
                probs_adj_consensus[h] = probs_adj_consensus[h-1]  # Replace with the previous stage
        hypnos_adj.append(probs_adj_consensus)
        
    if load_sp_sw or (len(sps) < 1):
        sp = yasa.spindles_detect(raws[index], include=(2), hypno=yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raws[index]))
        sps.append(sp)
        sw = yasa.sw_detect(raws[index], include=(2,3), hypno=yasa.hypno_upsample_to_data(hypnos_adj[index], sf_hypno=1/30, data=raws[index]))
        sws.append(sw)

if 'Hypno' in plots:
    for index, raw in enumerate(raws):
        fig, axes = plt.subplots(round(len(raw.ch_names))+2, 
                  figsize=(8, len(raw.ch_names)*2))
        fig.suptitle(f'#{raw.info["meas_date"]} Multitaper spectrogram, {spect_specs}')
        hyp = yasa.hypno_int_to_str(hypnos_max[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[0])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(plot_dt_format)} (Max Probs)')
        hyp = yasa.hypno_int_to_str(hypnos_adj[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[1])
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(plot_dt_format)} (Adj Probs)')
        for ch_index, ch in enumerate(raws[index].ch_names):
            hyp = hypnos[index][ch_index]; hyp_stats = sleep_stats(hyp)
            ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = axes[ch_index+2])
            ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(plot_dt_format)} ({ch})')

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        fig, ax = plt.subplots(figsize=(5, 2))
        hyp = yasa.hypno_int_to_str(hypnos_adj[index]); hyp_stats = sleep_stats(hyp)
        ax = plot_hypnogram(yasa.Hypnogram(hyp, start=pd.to_datetime(dts[index])), ax = ax, hl_lw=5)
        ax.set_title(f'{m2h(hyp_stats["TST"])} ({round(100 * (hyp_stats["TST_ADJ"] / hyp_stats["TIB"]))}%), SOL {m2h(hyp_stats["SOL_ADJ"])}, WASO {m2h(hyp_stats["WASO_ADJ"])}\nN3 {m2h(hyp_stats["N3"])}, R {m2h(hyp_stats["REM"])}, Awk {hyp_stats["N_AWAKE"]}\n{raw.info["meas_date"].strftime(plot_dt_format)} (Adj Probs)')

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
        raw_bp = raws_ori[index].copy()
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
            spects_c = []; stimes_c = []; sfreqs_c = []
            for ch_index, ch in enumerate(raw.ch_names):
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
        n_ax = 4
        n_cycles = round(len(raw.ch_names)/n_ax + .5)
        for cy in range(n_cycles):
            fig, axes = plt.subplots(n_ax, 
                      figsize=(12, n_ax*6))
            fig.suptitle(f'#{raw.info["meas_date"]} Multitaper spectrogram, {spect_specs}')
            axes = axes.flatten()
            idx_range = cy*n_ax + np.arange(0, n_ax, 1)
            for ch_index, ch in enumerate(raw.ch_names[min(idx_range):(max(idx_range)+1)]):
                spect, stimes, sfreqs = spects_l[index][cy*n_ax + ch_index], stimes_l[index][ch_index], sfreqs_l[index][cy*n_ax + ch_index]
                spect_data = nanpow2db(spect)
                
                start_time = raw.info['meas_date']
                times = [start_time + timedelta(seconds=int(s)) for s in stimes]
                
                dtx = times[1] - times[0]
                dy = sfreqs[1] - sfreqs[0]
                x_s = mdates.date2num(times[0]-dtx)
                x_e = mdates.date2num(times[-1]+dtx)
                extent = [x_s, x_e, sfreqs[-1]+dy, sfreqs[0]-dy]
        
                ax = axes[ch_index]
                im = ax.imshow(
                    spect_data, extent=extent, aspect='auto', 
                    cmap=plt.get_cmap('jet'), 
                    vmin = spect_vlim[0], vmax = spect_vlim[1],
                    )
                ax.xaxis_date()  # Interpret x-axis values as dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format time as HH:MM:SS
                fig.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)
                
                ax.invert_yaxis()
                
                ax.set_xlabel("Time")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title(f'{raw.ch_names[cy*n_ax + ch_index]} - {refs_ch[index][raw.ch_names[cy*n_ax + ch_index]]} ({sig_specs})')
                tick_intervals = np.linspace(x_s, x_e, 11)  # 11 points include 0% to 100%
                ax.set_xticks(tick_intervals)
                
                # Scale colormap
                if clim_scale:
                    clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
                    im.set_clim(clim)  # actually change colorbar scale
            plt.tight_layout()

# Power frequency plot with mne PSD computation
if 'Features' in plots:
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.flatten()
    for index, key in enumerate(raws_ori):
        raw = raws_ori[index].copy()
        raw.drop_channels(refs[index])
        ch = raw.ch_names
        if freq_method == 'mne_psd_welch':
            n_fft = int(w_fft * raw.info['sfreq'])
            psds, freqs = mne.time_frequency.psd_array_welch(
                raw.get_data(units='uV'), raw.info['sfreq'],
                fmin=freq_lim[0], fmax=freq_lim[1],
                n_fft= n_fft, output = 'power', n_jobs = nj)
            psds = 10 * np.log10(psds) # convert to dB
            plot_unit = units['psd_dB']; plot_type = 'PSD'
            plot_params = f'fft_window={w_fft}s'
        for c in range(len(ch)):
            ax[3].plot(freqs, psds[c], label=f'{ch[c]}-{refs_ch[index][ch[c]]}', linewidth=1)
        ax[3].legend()
        ax[3].set(title=f'', xlabel='Frequency (Hz)', ylabel=plot_unit)
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 20})
    fig.suptitle(f'{plot_type} ({sig_specs}, {topo_method}=[{plot_params}]')
    plt.tight_layout()
    
    for index, raw in enumerate(raws):
        plot_average(sps[index], "spindles", ax=ax[2], legend=False)

    for index, raw in enumerate(raws):
        axe = plot_average(sws[index], 'sw', center='PosPeak', ax=ax[0], legend=False);
        amps = round(sws[index].summary(grp_chan=True)[['Count','PTP']]).reset_index()
        max_amp = amps['Count'].argmax()
        axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{refs_ch[index][amps["Channel"][max_amp]]}')

        axe = plot_average(sws[index], 'sw', center='PosPeak', hue="Stage", ax=ax[1], legend=True)
        amps = round(sws[index].summary(grp_stage=True, grp_chan=True)[['Count','PTP']]).reset_index()
        max_amp = amps['Count'].argmax()
        axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{refs_ch[index][amps["Channel"][max_amp]]} in N{amps["Stage"][max_amp]}')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.rcParams.update({"font.size": old_fontsize})
