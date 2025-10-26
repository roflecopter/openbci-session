import numpy as np
import pandas as pd
import mysql.connector
import mne
import yasa
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import threading
import logging

from scipy.stats import mode
from scipy.ndimage import label, binary_dilation, binary_erosion
from datetime import datetime, timedelta, time as dtime, timezone
from qskit import butter_lowpass_filter, hrv_process, hrv_quality, sc_interp 
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging

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
    last_sleep_idx = None
    for h in range(len(hypno) - 1, -1, -1):  # Search backwards
        if hypno[h] != 'W':
            last_sleep_idx = h
            break
    
    if last_sleep_idx is None:
        # No sleep detected
        tst = 0
        waso = 0
    else:
        tst = 0
        waso = 0
        sol_idx = int(sol / epoch_time)
        
        for h in range(sol_idx + 1, last_sleep_idx + 1):  # Only count up to last sleep
            if hypno[h] != 'W':
                tst += epoch_time
            else:
                waso += epoch_time            
    hyp_stats = yasa.sleep_statistics(yasa.hypno_str_to_int(hypno), 1/30)
    hyp_stats['SOL_ADJ'] = sol / 60
    hyp_stats['TST_ADJ'] = tst / 60
    hyp_stats['N_AWAKE'] = major_awakenings_n
    hyp_stats['WASO_ADJ'] = waso / 60
    return hyp_stats

def plot_hypnogram(hyp, lw=1, hl_lw=3, font_size=10, highlight={'*': 'lightgrey', 'WAKE': 'orange', 'N1': None, 'N2': 'lightskyblue', 'N3': 'indigo', 'REM': 'red'}, fill_color=None, ax=None, cycles=None, sf_hypno=1/30, vline_dt=None):
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
    
    # hyp = yasa.Hypnogram(yasa.hypno_int_to_str(session['hypnos_adj']), start=pd.to_datetime(session['dts'])); cycles = None
    # lw=1; hl_lw=3; font_size=10; vline=None; highlight={'*': 'lightgrey', 'WAKE': 'orange', 'N1': None, 'N2': 'lightskyblue', 'N3': 'indigo', 'REM': 'red'}; fill_color=None; ax=None

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
            
    sf_hypno = 1/30
    if (cycles is not None) and len(cycles) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(cycles)))
        for i, cycle in enumerate(cycles):
            start_time = mdates.date2num(pd.Timestamp(hyp.start) + timedelta(seconds=cycle['start'] / sf_hypno))
            end_time = mdates.date2num(pd.Timestamp(hyp.start) + timedelta(seconds=cycle['end'] / sf_hypno))
            ax.axvspan(start_time, end_time, alpha=0.2, color=colors[i], 
                        label=f'Cycle {cycle["cycle"]}')
            
            # Mark REM period if present
            if cycle['rem_start'] is not None:
                rem_start_time = mdates.date2num(pd.Timestamp(hyp.start) + timedelta(seconds=cycle['rem_start'] / sf_hypno))
                rem_end_time = mdates.date2num(pd.Timestamp(hyp.start) + timedelta(seconds=cycle['rem_end'] / sf_hypno))
                ax.axvspan(rem_start_time, rem_end_time, alpha=0.1, 
                            color='red', ymin=0.65, ymax=0.75)
        
    if vline_dt is not None:
        ax.axvline(mdates.date2num(vline_dt), c='black', linestyle='--', linewidth=1)
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
    df_sync,
    event_type,
    hue="Channel",
    figsize=(6, 4.5),
    ax=None,
    **kwargs,
):
    """Plot the average event (not for REM, spindles & SW only)"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    assert not df_sync.empty, "Could not calculate event-locked data."
    assert hue in ["Stage", "Channel"], "hue must be 'Channel' or 'Stage'"
    assert hue in df_sync.columns, "%s is not present in data." % hue

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


def plot_rolling_spindle_density(sp_summary, sleep_stats_infos, dts, cfg, channels=['F7', 'F8'], 
                                window_minutes=10, stage_filter=[2], type_label ='spindle', verbose=False):
    """
    Create a rolling spindle/sws density plot over time
    
    Parameters:
    -----------
    sp_summary : pd.DataFrame
        YASA spindle summary dataframe
    session_start : datetime
        Recording start time
    channels : list
        Channels to include (default: ['F7', 'F8'])
    window_minutes : int
        Rolling window size in minutes (default: 10)
    stage_filter : list
        Sleep stages to include (default: [2] for N2 only)
    """
    
    # Filter spindles for specified channels and stages
    filtered_spindles = sp_summary[
        (sp_summary['Channel'].isin(channels)) &
        (sp_summary['Stage'].isin(stage_filter))
    ].copy()
    
    if len(filtered_spindles) == 0:
        print("No {type_label}s found with specified criteria")
        return
    
    # Convert Start times to datetime
    filtered_spindles['DateTime'] = pd.to_datetime(dts) + pd.to_timedelta(filtered_spindles['Start'], unit='s')
    
    # Create time bins (every minute)
    start_time = filtered_spindles['DateTime'].min().floor('min')
    end_time = filtered_spindles['DateTime'].max().ceil('min')
    time_range = pd.date_range(start_time, end_time, freq='1min')
    
    # Count spindles per minute for each channel
    density_data = []
    
    for current_time in time_range:
        window_start = current_time - timedelta(minutes=window_minutes/2)
        window_end = current_time + timedelta(minutes=window_minutes/2)
        
        # Get spindles in this window
        window_spindles = filtered_spindles[
            (filtered_spindles['DateTime'] >= window_start) &
            (filtered_spindles['DateTime'] < window_end)
        ]
        
        # Calculate density (spindles per minute per channel)
        spindle_count = len(window_spindles)
        density = spindle_count / (window_minutes * len(channels))  # per minute per channel
        
        density_data.append({
            'time': current_time,
            'density': density,
            'count': spindle_count
        })
    
    density_df = pd.DataFrame(density_data)
    max_density = density_df.loc[density_df['density'].idxmax()]
    
    dts_sol = dts + timedelta(minutes=sleep_stats_infos['SOL_ADJ'])
    dte = (dts_sol + timedelta(minutes=sleep_stats_infos['SPT']))
    mid = dts_sol + timedelta(seconds=(dte - dts_sol).total_seconds()/2)
    phase_diff = (mid - max_density["time"]).total_seconds()/60
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Main density plot
    ax1.plot(density_df['time'], density_df['density'], 
             color='blue', linewidth=2, alpha=0.8)
    ax1.fill_between(density_df['time'], density_df['density'], 
                     alpha=0.3, color='lightblue')
    ax1.set_ylabel(f'{type_label.capitalize()} Density\n({type_label}s/min/channel)', fontsize=12)
    ax1.set_title(f'{dts.strftime(cfg["plot_dt_format"])} Rolling {window_minutes}-Minute {type_label.capitalize()} Density Over Time\n'
                  f'{type_label} Phase Offset: {m2h(phase_diff)} | Peak {round(max_density["density"],2)} at {max_density["time"].strftime("%H:%M")} | Channels: {", ".join(channels)} | Stages: {stage_filter}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(mid,color="blue", linestyle="--", linewidth=1)
    ax1.axvline(max_density["time"],color="green", linestyle="--", linewidth=1)

    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Raw count plot
    ax2.plot(density_df['time'], density_df['count'], 
             color='red', linewidth=1, alpha=0.7)
    ax2.set_ylabel(f'Raw {type_label} Count\n(per 10min window)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    ax2.axvline(mid,color="blue", linestyle="--", linewidth=1)
    ax2.axvline(max_density["time"],color="green", linestyle="--", linewidth=1)
    
    # Format x-axis
    ax2.tick_params(axis='x', rotation=45)
    
    # Add statistics
    mean_density = density_df['density'].mean()
    ax1.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_density:.2f}')
    ax1.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    if verbose:
        print(f"{type_label.capitalize()} Density Summary:")
        print(f"Mean density: {mean_density:.2f} spindles/min/channel")
        print(f"Max density: {density_df['density'].max():.2f} {type_label}s/min/channel")
        print(f"Min density: {density_df['density'].min():.2f} {type_label}s/min/channel")
        print(f"Total {type_label}: {filtered_spindles.shape[0]}")
        print(f"Recording duration: {(end_time - start_time).total_seconds()/3600:.1f} hours")

    
    return fig, density_df, max_density, mid, phase_diff

def spindles_slow_fast(sp_sum, hypno, raw, eeg_ch_names, slow_ch=['F7','F8'], fast_ch=['O1','O2'], methods = [1,2]):
    # https://github.com/raphaelvallat/yasa/blob/master/notebooks/04_spindles_slow_fast.ipynb
    # method 1
    import seaborn as sns
    from scipy.stats import skewnorm
    from scipy.linalg import eigh
    from scipy.interpolate import RectBivariateSpline
    from scipy.signal import find_peaks, welch, detrend

    if 1 in methods:
        def draw_skewnorm(data):
            params = skewnorm.fit(data)
            x = np.linspace(min(data), max(data), 100)
            curve = skewnorm.pdf(x, *params)
            plt.plot(x, curve, color="black")
    
        plt.figure(figsize=(10, 6))
        histplot_kwargs = dict(stat='density', alpha=0.4, edgecolor=(1, 1, 1, 0.4))
        sns.histplot(sp_sum.loc[sp_sum['Channel'].isin(slow_ch), 'Frequency'], label=f'{slow_ch} - Slow', **histplot_kwargs)
        sns.histplot(sp_sum.loc[sp_sum['Channel'].isin(fast_ch), 'Frequency'], label=f'{fast_ch} - Fast', **histplot_kwargs)
        draw_skewnorm(sp_sum.loc[sp_sum['Channel'].isin(slow_ch), 'Frequency'])
        draw_skewnorm(sp_sum.loc[sp_sum['Channel'].isin(fast_ch), 'Frequency'])
        plt.legend()
        plt.xlabel('Frequency (Hz)')
        _ = plt.ylabel('Density')
        
        _, slow_mu, slow_std = skewnorm.fit(sp_sum.loc[sp_sum['Channel'].isin(slow_ch), 'Frequency'])
        _, fast_mu, fast_std = skewnorm.fit(sp_sum.loc[sp_sum['Channel'].isin(fast_ch), 'Frequency'])
        
        title = f"Method 1: slow and fast spindles peak frequencies\nSlow spindles ({slow_ch}) have a mean frequency of {round(slow_mu,2)}  Hz and a standard deviation of {round(slow_std,2)} Hz\nFast spindles ({fast_ch}) have a mean frequency of {round(fast_mu,2)} Hz and a standard deviation of {round(fast_std,2)} Hz"
        plt.title(title)

    # method 2
    # Compute Welch spectrum of N2 sleep for each channel
    if 2 in methods:
        sf = raw.info['sfreq']
        data = raw.get_data(eeg_ch_names,units='uV')
        f, pxx = welch(data[:, hypno == 2], sf, nperseg=(5 * sf))
        
        # Convert to dB to reduce 1/f (optional)
        # pxx = 10 * np.log10(pxx)
        
        # Keep only frequencies of interest
        pxx = pxx[:, np.logical_and(f >= 8, f <= 15)]
        f = f[np.logical_and(f >= 8, f <= 15)]
        
        # Plot average spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(f, pxx.mean(0), 'ko-', lw=3)
        plt.plot(f, np.rollaxis(pxx, axis=1), lw=1.5, ls=':', color='grey')
        plt.xlim(10, 15)
        plt.xlabel('Frequency (Hz)')
        _ = plt.ylabel('Power (dB)')
        
        idx_peaks, _ = find_peaks(pxx.mean(0))
        title = 'Method 2: channel-based power spectrum'
        
        if len(idx_peaks) > 1:
            title = f"{title}\nSlow spindles peak frequency = {round(f[idx_peaks[0]],2)} Hz\nFast spindles peak frequency = {round(f[idx_peaks[1]],2)} Hz"
        else:
            title = f"{title}\n No peaks found"
        plt.title(title)

    # method 3
    # Get filtered slow (10 - 12 Hz) and fast (12.5 - 15 Hz) data
    if 3 in methods:
        slow_n2_filt = mne.filter.filter_data(data[:, hypno == 2].astype(np.float64), sf, 10, 12, 
                                              h_trans_bandwidth=1, l_trans_bandwidth=1, verbose=0)
        fast_n2_filt = mne.filter.filter_data(data[:, hypno == 2].astype(np.float64), sf, 12.5, 15, 
                                              h_trans_bandwidth=1, l_trans_bandwidth=1, verbose=0)
        # Remove the mean (= detrend)
        slow_n2_filt = detrend(slow_n2_filt, type='constant')
        fast_n2_filt = detrend(fast_n2_filt, type='constant')
        
        # Compute the covariance matrices between channels
        slow_n2_cov = np.cov(slow_n2_filt)
        fast_n2_cov = np.cov(fast_n2_filt)
        
        # Plot the slow covariance matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(slow_n2_cov, cmap='Blues', square=True, 
                    xticklabels=eeg_ch_names, yticklabels=eeg_ch_names)
        plt.title('Slow sigma variance-covariance matrix')
        plt.xlabel('Channels')
        _ = plt.ylabel('Channels')
        
        # Get the eigenvalues / eigenvectors
        eigval, eigvec = eigh(slow_n2_cov, fast_n2_cov)
        
        # Flip to descending order
        eigval = np.flip(eigval)
        eigvec = np.fliplr(eigvec)
        
        print('Eigenvalues =', list(np.round(eigval, 2)))
        
        # Apply spatial filters by multiplying data with eigenvectors
        sf_comp_n2 = np.dot(data[:, hypno == 2].T, eigvec).T
        print(sf_comp_n2.shape)
        
        # Compute Welch spectrum of N2 sleep
        f, pxx = welch(sf_comp_n2, sf, nperseg=(5 * sf))
        
        pxx = pxx[:, np.logical_and(f >= 10, f <= 15)]
        f = f[np.logical_and(f >= 10, f <= 15)]
        
        # Convert to dB to reduce 1/f (optional)
        # pxx = 10 * np.log10(pxx)
        
        # Select the number of components to keep
        n_comp = 2
        
        # Plot first slow component
        plt.figure(figsize=(10, 6))
        plt.plot(f, np.rollaxis(pxx[:n_comp], axis=1), lw=1, ls=':', color='blue')
        plt.plot(f, np.rollaxis(pxx[-n_comp:], axis=1), lw=1, ls=':', color='red')
        
        # Plot average spectrum of four first / last components
        plt.plot(f, pxx[:n_comp].mean(0), 'bo-', lw=3)
        plt.plot(f, pxx[-n_comp:].mean(0), 'ro-', lw=3)
        _ = plt.xlim(10, 15)
    
        # Identify the peaks
        plt.title(f'Slow spindles peak frequency: {round(f[pxx[:n_comp].mean(0).argmax()],2)} Hz\nFast spindles peak frequency: {round(f[pxx[-n_comp:].mean(0).argmax()],2)} Hz')


def acc_process(raw, acc, dts):
    acc_signal = raw.get_data(acc) 
    # g = x^2 + y^2 + z^2
    acc_g = np.sqrt(np.sum(np.array(acc_signal)**2, axis=0))
    # https://www.researchgate.net/publication/264503253_Estimation_of_Force_during_Vertical_Jumps_using_Body_Fixed_Accelerometers
    # A 4th order Butterworth filter with a cut-off of 10 Hz was applied to smooth the accelerometer signals and the force platform traces[16]. A cut off frequency of 10 Hz was shown to be the best cut off frequency when analysing accelerometer data[17].
    acc_g_bp = butter_lowpass_filter(acc_g, 10, raw.info['sfreq'], order=4)
    # downsample to 100Hz
    acc_sf = 100; acc_sf_k = raw.info['sfreq'] / acc_sf
    acc_g_bp_ls = sc_interp(acc_g_bp, round(len(acc_g_bp) / acc_sf_k))
    
    acc_df = pd.DataFrame({'g': acc_g_bp_ls, 'dt': [dts + timedelta(seconds=i/acc_sf) for i in range(len(acc_g_bp_ls))]})
    acc_df.set_index('dt', inplace=True)
    
    # aggregate by 10s
    acc_agg = acc_df.resample('10S').mean()
    acc_agg = acc_agg.reset_index()

    # calc diff
    acc_agg['g_diff'] = acc_agg['g'].diff()
    acc_agg = acc_agg.dropna()
    acc_agg['g_diff_norm'] = (acc_agg['g_diff'] - acc_agg['g_diff'].mean()) / np.std(acc_agg['g_diff'])
    acc_agg['g_diff_norm_abs'] = abs(acc_agg['g_diff_norm'])        
    return acc_agg

def raw_preprocess(raw, eog_ch, emg_ch, ecg_ch, acc_ch, misc_ch, 
                   re_ref, nf, bpf, emg_bpf, eog_bpf, sf_to, nj=1):
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
    
    if len(emg_ch) > 0:
        emg = [elem for elem in ch if elem in emg_ch]
        if len(emg) > 0:
            ch_types = {}; 
            for emg_c in emg:
                ch_types[emg_c] = 'emg'
            raw.set_channel_types(ch_types)

    if len(ecg_ch) > 0:
        ecg = [elem for elem in ch if elem in ecg_ch]
        if len(ecg) > 0:
            ch_types = {}; 
            for ecg_c in ecg:
                ch_types[ecg_c] = 'ecg'
            raw.set_channel_types(ch_types)

    if len(acc_ch) > 0:
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
  
    raw_c = raw.copy()
    if re_ref:
        # split channels each side into separate array
        left = []; right = []; mid = []
        for c in eeg_ch:
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

    if not re_ref:
        ch_eeg_refs = {}
        for e, ech in enumerate(eeg_ch):
            ch_split = ech.split('-')
            ch_eeg_refs[ech] =  ch_split[1]
        ref_result = None
        ch_eeg_refs_exist = {k: v for k, v in ch_eeg_refs.items() if k in eeg_ch}
    else:
        ref_result = ref
        ch_eeg_refs_exist = {k: v for k, v in ch_eeg_refs.items() if k in ch_eeg}

    eeg_ch_names = list(ch_eeg_refs_exist.keys())
    if re_ref: eeg_ch_names.remove(ref_result)

    return raw_c, raw, eeg_ch_names, ref_result, ch_eeg_refs_exist, acc, ecg, eog, emg, misc

def process_hypno(raw, probs_ch, smooth_arousal=True):
    # N1 = 0, N2 = 1, N3 = 2, R =3, W = 4
    # max stage by avg probability from all channels
    probs_consensus = np.array(probs_ch).sum(axis=0)/len(probs_ch) 
    # W = 0, N1 = 1, N2 = 2, N3 = 3, R = 4
    probs_consensus = probs_consensus.argmax(axis=1) + 1
    probs_consensus[probs_consensus == 5] = 0
    
    # max probability stages for each channel
    hyp_consensus = np.array(probs_ch).argmax(axis=2) + 1
    hyp_consensus[hyp_consensus == 5] = 0

    # consensus with rem/n3 detected by at least single channel
    probs_adj_consensus = probs_consensus.copy()
    rem_occurences = np.sum(hyp_consensus==4, axis=0)
    n3_occurences = np.sum(hyp_consensus==3, axis=0)

    # at least 1/4 of channels sure that there is REM, then REM
    probs_adj_consensus[rem_occurences > round(len(raw.ch_names)/4)] = 4
    # at least single channel is enough to say that there is N3
    probs_adj_consensus[n3_occurences > 0 ] = 3

    # replace single awake (next and previous not awake) epochs with previous stage
    if smooth_arousal:
        for h in range(1, len(probs_adj_consensus)-1):
            if probs_adj_consensus[h] == 0 and probs_adj_consensus[h-1] != 0 and probs_adj_consensus[h+1] != 0:
                probs_adj_consensus[h] = probs_adj_consensus[h-1]  # Replace with the previous stage
    return probs_consensus, probs_adj_consensus

def process_ecg(raw, channel, dts, hypno_df, acc_agg, acc_ch, sleep_stats_info, 
                cfg, user, device, ecg_invert=False):
    sol_adj = sleep_stats_info['SOL_ADJ']
    tst_adj = sleep_stats_info['TST_ADJ']
    waso_adj = sleep_stats_info['WASO_ADJ']
    hypno_stages = {}
    hrv_exclude_quality = True
    hrv_exclude_acc = True
    
    # get major movements from accelerometer data            
    major_acc_epoch = None
    if len(acc_ch) > 0:
        if acc_agg is not None:
            acc_agg['dtr'] = acc_agg['dt'].dt.round('30s')
            hypno_acc = pd.merge(acc_agg,  hypno_df, on = "dtr", how = 'left')
            hypno_acc['dt'] = hypno_acc['dt_x']
            acc_th = 2 # in standard deviations
            # major_acc_nonwake_epoch = np.unique(hypno_acc[(hypno_acc['g_diff_norm_abs'] > acc_th) & (hypno_acc['h'] != 0)]['dtr'])
            major_acc_epoch = np.unique(hypno_acc[hypno_acc['g_diff_norm_abs'] > acc_th]['dtr'])
            major_acc_epoch = major_acc_epoch[major_acc_epoch > (dts + timedelta(seconds=sol_adj*60))]
            len(major_acc_epoch)

    # process ECG to HR
    window = 15; slide = 5; metrics = None
    hr_col = f'_{window}s'
    hrv_cache_tag = f'hrv_p{window}_s{slide}'; 
    hrv_file = f"{hrv_cache_tag}-{user}-{dts.strftime('%Y_%m_%d-%H_%M_%S')}.csv"
    hrv_filepath = os.path.join(cfg['cache_dir'], hrv_file)
    print(f'load: {hrv_filepath}')
    if not os.path.isfile(hrv_filepath):
        hr = hrv_process(raw.get_data(channel, units='uV')[0]*ecg_invert, sf = round(raw.info['sfreq']), 
            window = window, slide = slide, user = user, device = device, 
            dts = dts, metrics = metrics, cache_dir = cfg['cache_dir'])
    else:
        hr = pd.read_csv(hrv_filepath)

    # process ECG to HRV
    windows = [60]; slides = [20]; metrics = ['time','freq','ans','r_rr', 'nl']
    for iw, win in enumerate(windows):
        print(f'{iw} {win}')
        window = windows[iw]; slide = slides[iw]
        hrv_col = f'_{window}s'
        hrv_cache_tag = f'hrv_p{window}_s{slide}'; 
        hrv_file = f"{hrv_cache_tag}-{user}-{dts.strftime('%Y_%m_%d-%H_%M_%S')}.csv"
        hrv_filepath = os.path.join(cfg['cache_dir'], hrv_file)
        print(f'load: {hrv_filepath}')
        if not os.path.isfile(hrv_filepath):
            hrv = hrv_process(raw.get_data(channel, units='uV')[0]*ecg_invert, sf = round(raw.info['sfreq']), 
                window = window, slide = slide, user = user, device = device, 
                dts = dts, metrics = metrics, cache_dir = cfg['cache_dir'])
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
                
                title = f"""{dts.strftime(cfg['plot_dt_format'])} {acc_title} | OBCI {channel} {art_title} p{hrv_col}
HR {hr_t['tst']}±{hr_t['tst_sd']} N3 {hr_t['n3']}±{hr_t['n3_sd']} R {hr_t['r']}±{hr_t['r_sd']}
RMSSD {rmssd_t['tst']}±{rmssd_t['tst_sd']} N3 {rmssd_t['n3']}±{rmssd_t['n3_sd']} R {rmssd_t['r']}±{rmssd_t['r_sd']}
L/H {lfhf_t['tst']}±{lfhf_t['tst_sd']} N3 {lfhf_t['n3']}±{lfhf_t['n3_sd']} R {lfhf_t['r']}±{lfhf_t['r_sd']}
{abnormal_title}"""

                ecg_stat = {
                    'hr': hr_t['tst'],
                    'rmssd': rmssd_t['tst'],
                    'rmssd_n3': rmssd_t['n3'],
                    'rmssd_r': rmssd_t['r'],
                    'lh_n3': lfhf_t['n3'],
                    'lh_r': lfhf_t['r'],
                    'lh': lfhf_t['tst'],
                    'ecg_miss': round(np.sum(hrv_stages['tst'][f'missed'])/(tst_adj/60),3),
                    'ecg_ectopic': round(np.sum(hrv_stages['tst'][f'ectopic'])/(tst_adj/60),3),
                    'ecg_extra': round(np.sum(hrv_stages['tst'][f'extra'])/(tst_adj/60),3),
                    'ecg_longshort': round(np.sum(hrv_stages['tst'][f'longshort'])/(tst_adj/60),3),
                    'mh': movements_per_hour
                    }
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
                plt.axvline(dts + timedelta(seconds=float(sol_adj)*60), c='grey', linestyle='--', linewidth=.5)
                plt.axvline(dts + timedelta(seconds=float(sol_adj)*60 + sleep_cycles_limit), c='grey', linestyle='--', linewidth=.5)
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
    return fig, ecg_stat, hrv_stages, major_acc_epoch, hrv_col, hrv

def process_hrvst(raw, channel, hrvst_pre, hrvst_post, hrvst_period, hrvst_peroid_last, dts, cfg, user, device, ecg_invert=False):
    raw_hrv = raw.copy().pick(channel)
    hrvst_duration = hrvst_pre*2 + hrvst_post + hrvst_period
    crop_secs = int(raw.n_times/raw.info['sfreq']-hrvst_duration)
    dts = dts + timedelta(seconds=crop_secs)
    raw_hrv.crop(tmin = crop_secs)
    windows = [60]; slides = [1]; metrics = ['time','freq','ans','r_rr', 'nl']
    for iw, win in enumerate(windows):
        print(f'{iw} {win}')
        window = windows[iw]; slide = slides[iw]
        hrv_col = f'_{window}s'
        hrv_cache_tag = f'hrv_p{window}_s{slide}'; 
        hrv_file = f"{hrv_cache_tag}-{user}-{dts.strftime('%Y_%m_%d-%H_%M_%S')}.csv"
        hrv_filepath = os.path.join(cfg['cache_dir'], hrv_file)
        print(f'load: {hrv_filepath}')
        if not os.path.isfile(hrv_filepath):
            from qskit import hrv_process
            hrv = hrv_process(raw_hrv.get_data(channel, units='uV')[0]*ecg_invert, sf = round(raw.info['sfreq']), 
                window = window, slide = slide, user = user, device = device, 
                dts = dts, metrics = metrics, cache_dir = cfg['cache_dir'])
        else:
            hrv = pd.read_csv(hrv_filepath)
        hrv['dt'] = pd.to_datetime(hrv['dt'])
        hrvst_start = hrvst_pre*2 + hrvst_period - hrvst_peroid_last                    
        hrvst_end = hrvst_start + hrvst_peroid_last
        hrvst_values = hrv[
            (hrv['dt'] >= (dts + timedelta(seconds=hrvst_start))) &
            (hrv['dt'] <= (dts + timedelta(seconds=hrvst_end)))
            ]
        st_hr = np.median(hrvst_values[f'hr{hrv_col}'])
        st_hrv = np.median(hrvst_values[f'rmssd{hrv_col}'])
        title = f"""{dts.strftime(cfg['plot_dt_format'])}\nHR {round(st_hr,1)} RMSSD {round(st_hrv,1)}"""

        old_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({"font.size": 15})
        fig, ax = plt.subplots(1,1, figsize=(9,7))
        plt.title(title)
        p_size = 20
        plt.plot(hrv['dt'], hrv[f'rmssd{hrv_col}'], c='green', linewidth=2)
        plt.vlines(hrv['dt'],0, 100*hrv[f'artifacts_rate'], color='red')
        plt.scatter(hrvst_values['dt'], hrvst_values[f'rmssd{hrv_col}'], c='b', s=p_size)
        
        plt.axvline(x=dts+timedelta(seconds=hrvst_pre*2), color='blue', linestyle='--', linewidth=2)
        plt.axvline(x=dts+timedelta(seconds=hrvst_pre*2+hrvst_period), color='blue', linestyle='--', linewidth=2)
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.dates as mdates
        
        hrv_lim = [3, 45]; ax.set_ylim(hrv_lim[0],hrv_lim[1])
        ax2 = ax.twinx()
        ax2.plot(hrv['dt'], hrv[f'hr{hrv_col}'], c='red', linewidth=1.5)
        ax2.tick_params(axis='y', labelsize=13)
        ax2.axhline(50, c='red', linestyle='--', linewidth=.5)
        hr_lim = [60, 120]; ax2.set_ylim(hr_lim[0],hr_lim[1])
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_yticks(np.arange(0, hrv_lim[1]+10, 10))
        ax2.set_yticks(np.arange(hr_lim[0], max([hr_lim[1], 5*round(max(hrv[f'hr{hrv_col}'])/5+0.5)])+5, 5))
        ax.tick_params(axis='x', labelsize=13, rotation=45)  # Set the x-axis ticks font size to 12
        ax.tick_params(axis='y', labelsize=13)  # Set the y-axis ticks font size to 12
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax.tick_params(axis='x')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01) 
        plt.tight_layout()
        plt.rcParams.update({"font.size": old_fontsize})
        return fig, dts, st_hr, st_hrv

def process_bp(raw, channels, ref_channel, topo_ref, hypno_adj, stages, re_ref, bp_bands, bp_relative):
    raw_bp  = raw.copy().pick(channels)
    
    if re_ref:
        raw_bp.add_reference_channels(ref_channel)

    if not re_ref:
        mapping = {ch: ch.split("-")[0] for ch in channels if "-" in ch}                
        raw_bp.rename_channels(mapping)

    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw_bp.set_montage(ten_twenty_montage , match_case=False)

    if topo_ref == 'REST':
        sphere = mne.make_sphere_model("auto", "auto", raw.info)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=5.0)
        forward = mne.make_forward_solution(raw_bp.info, trans=None, src=src, bem=sphere)
        raw_bp.set_eeg_reference("REST", forward=forward)
    elif topo_ref == 'AR':
        raw_bp.set_eeg_reference(ref_channels = 'average')

    
    bps_s = []; stages_return = {}
    for ss, s in stages.items():
        hypno_up = yasa.hypno_upsample_to_data(hypno_adj, sf_hypno=1/30, data=raw)
        if s in hypno_up:
            bandpower = yasa.bandpower(raw_bp, hypno=hypno_up, include=(s), bands=bp_bands, relative=bp_relative)
            bp_b = []
            for b in range(len(bp_bands)):
                bp = np.sqrt(bandpower.xs(s)[bp_bands[b][2]])
                bp_b.append(bp)
            bps_s.append(bp_b)
            stages_return[ss] = s
    return raw_bp, bps_s, stages_return

def topomap_plot(dts, raw_bp, bps_s, bp_relative, topo_ref, sig_specs, topo_method, hypno_adj, stages, stages_plot, stages_return, bp_bands, units, cfg):
    fig, axes = plt.subplots(len(stages_plot),len(bp_bands), 
             figsize=(len(bp_bands)*2, len(stages_plot)*2))
    plot_type = f'{dts.strftime(cfg["plot_dt_format"])} Amplitude (ref={topo_ref})'; plot_params = ''
    hypno_up = yasa.hypno_upsample_to_data(hypno_adj, sf_hypno=1/30, data=raw_bp)
    for sp in range(len(stages_plot)):
        s = stages_plot[sp]
        s_key = next((k for k, v in stages_return.items() if v == s), None)
        if s_key is not None:
            si = list(stages_return.keys()).index(s_key)
            for b in range(len(bp_bands)):
                bp = bps_s[si][b]
                if not bp_relative:
                    p_max = np.max(bp)
                    p_min = np.min(bp)
                else:
                    p_max = max(np.array(bps_s).max(axis=2)[...,b])
                    p_min = min(np.array(bps_s).min(axis=2)[...,b])*1.2
                vlim = (p_min,p_max)
                ax = axes[sp,b]
                im, _ = mne.viz.plot_topomap(
                    bp, 
                    raw_bp.info,
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
                    ax.set_title(f'{s_key} {bl}')
                elif b == 1:
                    ax.set_title(f'{bl}')
                else:
                    ax.set_title(f'{bl}')
    fig.suptitle(f'{plot_type} ({sig_specs}, {topo_method}=[{plot_params}]')
    plt.tight_layout()
    return fig

def create_spect(raw, channels, multitaper_spectrogram, nanpow2db, spect_lim, frequency_range, time_bandwidth, num_tapers, 
    window_params, min_nfft, detrend_opt, multiprocess, cpus,
    weighting, plot_on, return_fig, clim_scale, verbose, xyflip):
    
    ch_order = ['F', 'C', 'O', 'T', 'A']
    order_map = {letter: l_index for l_index, letter in enumerate(ch_order)}
    def order_value(string):
        first_letter = string[0]
        return order_map.get(first_letter, float('inf'))  # Use inf for letters not in custom_order
    
    spects_c = []; stimes_c = []; sfreqs_c = []
    ch_eeg_sorted = sorted(channels, key=order_value)
    
    raw_spect  = raw.copy().pick(ch_eeg_sorted)
    
    for ch_index, ch in enumerate(ch_eeg_sorted):
        spect, stimes, sfreqs = multitaper_spectrogram(
            raw_spect.get_data(picks = [ch], units='uV'), raw_spect.info['sfreq'], 
            frequency_range, time_bandwidth, num_tapers, 
            window_params, min_nfft, detrend_opt, multiprocess, cpus,
            weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
        spects_c.append(spect)
        stimes_c.append(stimes)
        sfreqs_c.append(sfreqs)
    return spects_c, stimes_c, sfreqs_c

def plot_multitaper_spect_all(raw, dts, channels, spects_c, stimes_c, sfreqs_c, hypno_df, spect_specs, cfg, nanpow2db, spect_vlim, clim_scale, sig_specs):
    raw_spect  = raw.copy().pick(channels)
    
    # summary spectrum
    fig, ax = plt.subplots(figsize=(14, 5))
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})
    fig.suptitle(f'{dts.strftime(cfg["plot_dt_format"])} Spectrogram, {spect_specs}')

    spect, stimes, sfreqs = np.array(spects_c).max(axis=0), stimes_c[0], sfreqs_c[0]
    spect_data = nanpow2db(spect)
    
    times = [dts + timedelta(seconds=int(s)) for s in stimes]
    
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
    hypno_df = hypno_df
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
    return fig
        
def plot_multitaper_spect_ch(raw, dts, channels, ref_ch, spects_c, stimes_c, sfreqs_c, hypno_df, spect_specs, cfg, nanpow2db, spect_vlim, clim_scale, sig_specs):    
    ch_order = ['F', 'C', 'O', 'T', 'A']
    order_map = {letter: l_index for l_index, letter in enumerate(ch_order)}
    def order_value(string):
        first_letter = string[0]
        return order_map.get(first_letter, float('inf'))  # Use inf for letters not in custom_order

    ch_eeg_sorted = sorted(channels, key=order_value)
    
    n_ax = 2
    n_up = 0 if (len(channels) % 2) == 0 else .5
    n_cycles = round(len(channels)/n_ax + n_up)
    figs = []
    for cy in range(n_cycles):
        fig, axes = plt.subplots(n_ax, 
                  figsize=(14, n_ax*5))
        old_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({"font.size": 18})
        fig.suptitle(f'{dts.strftime(cfg["plot_dt_format"])} Spectrogram, {spect_specs}')
        axes = axes.flatten()
        idx_range = cy*n_ax + np.arange(0, n_ax, 1)
        for cch_index, cch in enumerate(ch_eeg_sorted[min(idx_range):(max(idx_range)+1)]):
            spect, stimes, sfreqs = spects_c[cy*n_ax + cch_index], stimes_c[cch_index], sfreqs_c[cy*n_ax + cch_index]
            spect_data = nanpow2db(spect)
            
            times = [dts + timedelta(seconds=int(s)) for s in stimes]
            
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
            ax.set_title(f'{ch_eeg_sorted[cy*n_ax + cch_index]} - {ref_ch[ch_eeg_sorted[cy*n_ax + cch_index]]} ({sig_specs})')
            tick_intervals = np.linspace(x_s, x_e, 11)  # 11 points include 0% to 100%
            ax.set_xticks(tick_intervals)
            
            # Scale colormap
            if clim_scale:
                clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
                im.set_clim(clim)  # actually change colorbar scale

        plt.tight_layout()
        plt.rcParams.update({"font.size": old_fontsize})
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01) 
        figs.append(fig)
    return figs

def psds(raw_ori, freq_method, channels, w_fft, freq_lim, nj):
    raw  = raw_ori.copy().pick(channels)
    if freq_method == 'mne_psd_welch':
        n_fft = int(w_fft * raw.info['sfreq'])
        psds, freqs = mne.time_frequency.psd_array_welch(
            raw.get_data(units='uV'), raw.info['sfreq'],
            fmin=freq_lim[0], fmax=freq_lim[1],
            n_fft= n_fft, output = 'power', n_jobs = nj)
        psds = 10 * np.log10(psds) # convert to dB
    return psds, freqs

def psd_plot(psds, freqs, dts, raw_ori, channels, ref_ch, sp_sync, sp_summary, sp_ch, sp_metric, sw_sync, sw_summary_ch, sw_summary_chst, sw_ch, sw_metric, freq_method, w_fft, freq_lim, units, sig_specs, cfg, nj, plot_avg = True):
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.flatten()
    plot_unit = units['psd_dB']; plot_type = 'PSD'
    plot_params = f'fft_window={w_fft}s'
    for c in range(len(channels)):
        ax[3].plot(freqs, psds[c], label=f'{channels[c]}-{ref_ch[channels[c]]}', linewidth=1)
    ax[3].legend()
    ax[3].set(title=f'', xlabel='Frequency (Hz)', ylabel=plot_unit)
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 20})
    fig.suptitle(f'{dts.strftime(cfg["plot_dt_format"])} {plot_type} ({sig_specs}, [{plot_params}]')
    
    if plot_avg:
        plot_average(sp_sync, "spindles", ax=ax[2], legend=False)
        if sp_metric is not None:
            ax[2].set_title(f'Density: {round(sp_metric[0],2)} CV: {round(sp_metric[3],2)} {sp_ch}\n Early {round(sp_metric[1],2)} Late {round(sp_metric[2],2)} E/L {round(sp_metric[1]/sp_metric[2],2)}')        

    amps = round(sw_summary_ch[['Count','PTP']]).reset_index()
    max_amp = amps['Count'].argmax()
    
    if plot_avg:
        axe = plot_average(sw_sync, 'sw', ax=ax[0], legend=False);
        axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{ref_ch[amps["Channel"][max_amp]]}')

    amps = round(sw_summary_chst[['Count','PTP']]).reset_index()
    max_amp = amps['Count'].argmax()
    if plot_avg:
        axe = plot_average(sw_sync, 'sw', hue="Stage", ax=ax[1], legend=True)
        axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{ref_ch[amps["Channel"][max_amp]]} in N{amps["Stage"][max_amp]}')
        if sw_metric is not None:
            axe.set_title(f'SW Amp: {amps["Count"][max_amp]}*{round(amps["PTP"][max_amp])}{units["amp"]} for {amps["Channel"][max_amp]}-{ref_ch[amps["Channel"][max_amp]]} in N{amps["Stage"][max_amp]}\nCV: {round(sw_metric[0],2)} Early {round(sw_metric[1],2)} Late {round(sw_metric[2],2)}, E/L {round(sw_metric[1]/sw_metric[2],2)} {sw_ch}')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.rcParams.update({"font.size": old_fontsize})
    return fig, amps, max_amp

def plot_radar(dts, sleep_stats_info, ecg_stats_info, acc, cfg, n3_goal = 90, rem_goal = 105, awk_goal = 30, hr_goal = 45, hrv_goal = 35, mh_goal = 4.3):
    n3 = sleep_stats_info['N3']
    rem = sleep_stats_info['REM']
    awk = sleep_stats_info['SOL_ADJ']+sleep_stats_info['WASO_ADJ']
    hr = ecg_stats_info['hr']
    hrv = ecg_stats_info['rmssd_n3']
    if len(acc) > 0: 
        mh = ecg_stats_info['mh']
    else:
        mh = 4.3
    
    
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
    ax.set_title(f'{dts.strftime(cfg["plot_dt_format"])} Radar')
    plt.tight_layout()
    plt.rcParams.update({"font.size": old_fontsize})
    return fig

# def fill_cycle_gaps(cycles, cycle_hypno, hypno_smooth, gap_assignment='previous'):
#     """
#     Fill gaps between cycles with appropriate cycle assignments.
    
#     Parameters
#     ----------
#     cycles : list
#         List of cycle dictionaries
#     cycle_hypno : array
#         Current cycle assignments
#     hypno_smooth : array
#         Smoothed hypnogram
#     gap_assignment : str
#         How to assign gaps: 'previous', 'next', 'split', 'closest_rem'
    
#     Returns
#     -------
#     cycles : list
#         Updated cycle list
#     cycle_hypno : array
#         Updated cycle assignments
#     """
#     if len(cycles) <= 1:
#         return cycles, cycle_hypno
    
#     for i in range(len(cycles) - 1):
#         current_cycle = cycles[i]
#         next_cycle = cycles[i + 1]
        
#         gap_start = current_cycle['end'] + 1
#         gap_end = next_cycle['start'] - 1
        
#         if gap_start <= gap_end:
#             gap_stages = hypno_smooth[gap_start:gap_end + 1]
            
#             # Only process if there's sleep in the gap
#             if np.any(gap_stages > 0):
#                 if gap_assignment == 'previous':
#                     # Assign all to previous cycle
#                     cycle_hypno[gap_start:gap_end + 1] = current_cycle['cycle']
#                     current_cycle['end'] = gap_end
                    
#                 elif gap_assignment == 'next':
#                     # Assign all to next cycle
#                     cycle_hypno[gap_start:gap_end + 1] = next_cycle['cycle']
#                     next_cycle['start'] = gap_start
                    
#                 elif gap_assignment == 'split':
#                     # Split the gap in the middle
#                     mid_point = (gap_start + gap_end) // 2
#                     cycle_hypno[gap_start:mid_point + 1] = current_cycle['cycle']
#                     cycle_hypno[mid_point + 1:gap_end + 1] = next_cycle['cycle']
#                     current_cycle['end'] = mid_point
#                     next_cycle['start'] = mid_point + 1
                    
#                 elif gap_assignment == 'closest_rem':
#                     # Assign based on proximity to REM periods
#                     if current_cycle['rem_end'] is not None and next_cycle['rem_start'] is not None:
#                         # Find the point closest to equidistant from both REM periods
#                         for idx in range(gap_start, gap_end + 1):
#                             dist_to_prev_rem = idx - current_cycle['rem_end']
#                             dist_to_next_rem = next_cycle['rem_start'] - idx
                            
#                             if dist_to_prev_rem <= dist_to_next_rem:
#                                 cycle_hypno[idx] = current_cycle['cycle']
#                             else:
#                                 cycle_hypno[idx] = next_cycle['cycle']
                        
#                         # Update cycle boundaries
#                         current_end = np.where(cycle_hypno[gap_start:gap_end + 1] == current_cycle['cycle'])[0]
#                         if len(current_end) > 0:
#                             current_cycle['end'] = gap_start + current_end[-1]
                        
#                         next_start = np.where(cycle_hypno[gap_start:gap_end + 1] == next_cycle['cycle'])[0]
#                         if len(next_start) > 0:
#                             next_cycle['start'] = gap_start + next_start[0]
#                     else:
#                         # Fallback to previous assignment
#                         cycle_hypno[gap_start:gap_end + 1] = current_cycle['cycle']
#                         current_cycle['end'] = gap_end
    
#     # Handle gap after last cycle
#     last_cycle = cycles[-1]
#     final_sleep = np.where(hypno_smooth[last_cycle['end'] + 1:] > 0)[0]
#     if len(final_sleep) > 0:
#         final_sleep_start = last_cycle['end'] + 1 + final_sleep[0]
#         final_sleep_end = last_cycle['end'] + 1 + final_sleep[-1]
#         cycle_hypno[final_sleep_start:final_sleep_end + 1] = last_cycle['cycle']
#         last_cycle['end'] = final_sleep_end
    
#     return cycles, cycle_hypno


# # Update the main function to use gap filling
# def detect_sleep_cycles(hypno, sf_hypno=1/30, min_nrem_duration=15, min_rem_duration=1, 
#                        awakening_threshold=5, merge_close_rem=10, first_rem_window=120,
#                        min_cycle_duration=60, last_cycle_merge_threshold=45,
#                        fill_gaps=True, gap_assignment='previous'):
#     """
#     Detect sleep cycles from a hypnogram using published criteria.
    
#     Based on Feinberg & Floyd (1979) and updated by Jenni & Carskadon (2004),
#     with modifications for micro-awakening handling and end-of-night short cycles.
    
#     Parameters
#     ----------
#     hypno : array
#         Sleep stage array (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
#     sf_hypno : float
#         Sampling frequency of hypnogram (default 1/30 Hz for 30s epochs)
#     min_nrem_duration : int
#         Minimum NREM duration in minutes to start a cycle (default 15)
#     min_rem_duration : float
#         Minimum REM duration in minutes to be counted (default 1)
#     awakening_threshold : int
#         Maximum awakening duration in minutes to not break cycle (default 5)
#     merge_close_rem : int
#         Merge REM periods separated by less than this many minutes (default 10)
#     first_rem_window : int
#         Window in minutes to look for first REM period (default 120)
#     min_cycle_duration : int
#         Minimum cycle duration in minutes (default 60)
#     last_cycle_merge_threshold : int
#         Merge last cycle if shorter than this many minutes (default 45)
#     fill_gaps : bool
#         Whether to fill gaps between cycles (default True)
#     gap_assignment : str
#         How to assign gaps: 'previous', 'next', 'split', 'closest_rem' (default 'previous')
    
#     Returns
#     -------
#     cycles : list of dicts
#         Each dict contains 'start', 'end', 'nrem_start', 'rem_start', 'rem_end'
#     cycle_hypno : array
#         Same length as hypno, with cycle number (0 = no cycle)
#     """
    
#     # Convert parameters from minutes to epochs
#     epochs_per_min = int(60 * sf_hypno)
#     min_nrem_epochs = min_nrem_duration * epochs_per_min
#     min_rem_epochs = int(min_rem_duration * epochs_per_min)
#     wake_threshold_epochs = awakening_threshold * epochs_per_min
#     merge_rem_epochs = merge_close_rem * epochs_per_min
#     first_rem_epochs = first_rem_window * epochs_per_min
#     min_cycle_epochs = min_cycle_duration * epochs_per_min
#     merge_threshold_epochs = last_cycle_merge_threshold * epochs_per_min
    
#     # Step 1: Handle micro-awakenings by filling short wake periods
#     hypno_smooth = hypno.copy()
#     wake_mask = hypno == 0
    
#     # Label continuous wake periods
#     wake_labeled, n_wake_periods = label(wake_mask)
    
#     # Fill short wake periods (less than threshold)
#     for i in range(1, n_wake_periods + 1):
#         wake_period = wake_labeled == i
#         period_length = np.sum(wake_period)
        
#         if period_length < wake_threshold_epochs:
#             # Find what sleep stage surrounds this wake period
#             indices = np.where(wake_period)[0]
#             start_idx = indices[0]
#             end_idx = indices[-1]
            
#             # Look at stages before and after
#             before_stage = hypno_smooth[start_idx - 1] if start_idx > 0 else 0
#             after_stage = hypno_smooth[end_idx + 1] if end_idx < len(hypno_smooth) - 1 else 0
            
#             # Fill with most common surrounding stage (excluding wake)
#             if before_stage > 0 and after_stage > 0:
#                 fill_stage = before_stage if before_stage == after_stage else min(before_stage, after_stage)
#                 hypno_smooth[wake_period] = fill_stage
    
#     # Step 2: Merge close REM periods
#     rem_mask = hypno_smooth == 4
#     rem_labeled, n_rem_periods = label(rem_mask)
    
#     # Check each pair of consecutive REM periods
#     for i in range(1, n_rem_periods):
#         rem1_end = np.where(rem_labeled == i)[0][-1]
#         rem2_start = np.where(rem_labeled == i + 1)[0][0]
        
#         # If gap is small and contains NREM (not wake), merge
#         gap = rem2_start - rem1_end
#         if gap < merge_rem_epochs:
#             gap_stages = hypno_smooth[rem1_end + 1:rem2_start]
#             if np.all(gap_stages > 0):  # All NREM, no wake
#                 hypno_smooth[rem1_end + 1:rem2_start] = 4
    
#     # Step 3: Identify sleep cycles
#     cycles = []
#     cycle_hypno = np.zeros_like(hypno)
    
#     # Find sleep onset (first N2 or N3)
#     sleep_onset = np.where((hypno_smooth == 2) | (hypno_smooth == 3))[0]
#     if len(sleep_onset) == 0:
#         return cycles, cycle_hypno
    
#     sleep_onset = sleep_onset[0]
#     current_pos = sleep_onset
#     cycle_num = 1
    
#     while current_pos < len(hypno_smooth):
#         # Find start of NREM period (N2 or N3)
#         nrem_start = None
#         for i in range(current_pos, len(hypno_smooth)):
#             if hypno_smooth[i] in [2, 3]:
#                 nrem_start = i
#                 break
        
#         if nrem_start is None:
#             break
        
#         # Find continuous NREM period of sufficient length
#         nrem_period_start = nrem_start
#         nrem_length = 0
        
#         for i in range(nrem_start, len(hypno_smooth)):
#             if hypno_smooth[i] in [1, 2, 3]:  # Any NREM
#                 nrem_length += 1
#             else:
#                 break
        
#         if nrem_length < min_nrem_epochs:
#             current_pos = nrem_period_start + nrem_length + 1
#             continue
        
#         # Look for REM period
#         rem_start = None
#         search_end = min(nrem_period_start + first_rem_epochs, len(hypno_smooth))
        
#         # For first cycle, look within first REM window
#         if cycle_num == 1:
#             search_start = nrem_period_start
#         else:
#             search_start = nrem_period_start + min_nrem_epochs
        
#         for i in range(search_start, search_end):
#             if hypno_smooth[i] == 4:
#                 # Check if REM period is long enough
#                 rem_length = 0
#                 rem_start_temp = i
#                 for j in range(i, len(hypno_smooth)):
#                     if hypno_smooth[j] == 4:
#                         rem_length += 1
#                     else:
#                         break
                
#                 if rem_length >= min_rem_epochs or (cycle_num == 1 and rem_length > 0):
#                     rem_start = rem_start_temp
#                     rem_end = rem_start_temp + rem_length - 1
#                     break
        
#         # If no REM found but sufficient NREM, still count as cycle (esp. for last cycle)
#         if rem_start is None:
#             # Check if this is potentially the last cycle
#             remaining_sleep = np.sum(hypno_smooth[nrem_period_start:] > 0)
#             if remaining_sleep > min_nrem_epochs:
#                 cycles.append({
#                     'cycle': cycle_num,
#                     'start': nrem_period_start,
#                     'end': len(hypno_smooth) - 1,
#                     'nrem_start': nrem_period_start,
#                     'rem_start': None,
#                     'rem_end': None
#                 })
#                 cycle_hypno[nrem_period_start:] = cycle_num
#                 break
#             else:
#                 current_pos = search_end
#                 continue
        
#         # We have a complete cycle
#         cycle_end = rem_end
        
#         # Extend cycle end to include any trailing N1
#         for i in range(rem_end + 1, len(hypno_smooth)):
#             if hypno_smooth[i] == 1:
#                 cycle_end = i
#             else:
#                 break
        
#         cycles.append({
#             'cycle': cycle_num,
#             'start': nrem_period_start,
#             'end': cycle_end,
#             'nrem_start': nrem_period_start,
#             'rem_start': rem_start,
#             'rem_end': rem_end
#         })
        
#         cycle_hypno[nrem_period_start:cycle_end + 1] = cycle_num
        
#         cycle_num += 1
#         current_pos = cycle_end + 1
    
#     # Post-processing: Handle short cycles at the end of night
#     if len(cycles) >= 2:
#         # Check if last cycle is suspiciously short
#         last_cycle = cycles[-1]
#         last_duration = last_cycle['end'] - last_cycle['start'] + 1
        
#         if last_duration < merge_threshold_epochs:
#             # Option 1: Merge with previous cycle if close enough
#             prev_cycle = cycles[-2]
#             gap = last_cycle['start'] - prev_cycle['end']
            
#             if gap < merge_rem_epochs:  # If cycles are close
#                 # Merge the cycles
#                 prev_cycle['end'] = last_cycle['end']
#                 if last_cycle['rem_start'] is not None:
#                     # Update REM end to include the last REM
#                     prev_cycle['rem_end'] = last_cycle['rem_end']
                
#                 # Update cycle_hypno
#                 cycle_hypno[cycle_hypno == last_cycle['cycle']] = prev_cycle['cycle']
                
#                 # Remove the last cycle
#                 cycles.pop()
            
#             # Option 2: If can't merge, check if it's just a REM fragment
#             elif last_cycle['rem_start'] is not None and last_cycle['nrem_start'] == last_cycle['rem_start']:
#                 # This is just a REM fragment, not a real cycle
#                 cycle_hypno[cycle_hypno == last_cycle['cycle']] = 0
#                 cycles.pop()
    
#     # Additional check: Remove any cycles that are too short
#     cycles_to_keep = []
#     for cycle in cycles:
#         duration = cycle['end'] - cycle['start'] + 1
#         if duration >= min_cycle_epochs or cycle['cycle'] == 1:  # Keep first cycle even if short
#             cycles_to_keep.append(cycle)
#         else:
#             # Remove from cycle_hypno
#             cycle_hypno[cycle_hypno == cycle['cycle']] = 0
    
#     # Renumber cycles if any were removed
#     if len(cycles_to_keep) < len(cycles):
#         cycles = cycles_to_keep
#         # Renumber in cycle_hypno
#         for i, cycle in enumerate(cycles):
#             old_num = cycle['cycle']
#             new_num = i + 1
#             cycle['cycle'] = new_num
#             cycle_hypno[cycle_hypno == old_num] = new_num
    
#     # Fill gaps between cycles with sleep stages
#     if len(cycles) > 1:
#         for i in range(len(cycles) - 1):
#             current_cycle = cycles[i]
#             next_cycle = cycles[i + 1]
            
#             gap_start = current_cycle['end'] + 1
#             gap_end = next_cycle['start'] - 1
            
#             if gap_start <= gap_end:
#                 # Check what's in the gap
#                 gap_stages = hypno_smooth[gap_start:gap_end + 1]
                
#                 # If there's any sleep in the gap, assign it
#                 if np.any(gap_stages > 0):
#                     # Assign to the cycle with more similar stages
#                     # or to the previous cycle by default
#                     cycle_hypno[gap_start:gap_end + 1] = current_cycle['cycle']
                    
#                     # Update cycle end time
#                     current_cycle['end'] = gap_end
    
#     # Apply gap filling if requested
#     if fill_gaps:
#         cycles, cycle_hypno = fill_cycle_gaps(cycles, cycle_hypno, hypno_smooth, gap_assignment)
    
#     return cycles, cycle_hypno    

def summarize_cycles(cycles, hypno, sf_hypno=1/30):
    """
    Create summary statistics for detected cycles.
    
    Parameters
    ----------
    cycles : list
        Output from detect_sleep_cycles
    hypno : array
        Sleep stage array
    sf_hypno : float
        Sampling frequency
    
    Returns
    -------
    summary : pd.DataFrame
        Summary statistics for each cycle
    """
    summaries = []
    
    for cycle in cycles:
        summary = {
            'cycle': cycle['cycle'],
            'duration_min': (cycle['end'] - cycle['start'] + 1) / (sf_hypno * 60),
            'nrem_duration_min': 0,
            'rem_duration_min': 0,
            'n1_min': 0,
            'n2_min': 0,
            'n3_min': 0,
            'rem_latency_min': None
        }
        
        # Calculate stage durations within cycle
        cycle_stages = hypno[cycle['start']:cycle['end'] + 1]
        for stage in [1, 2, 3, 4]:
            duration = np.sum(cycle_stages == stage) / (sf_hypno * 60)
            if stage == 1:
                summary['n1_min'] = duration
            elif stage == 2:
                summary['n2_min'] = duration
            elif stage == 3:
                summary['n3_min'] = duration
            elif stage == 4:
                summary['rem_duration_min'] = duration
        
        summary['nrem_duration_min'] = summary['n1_min'] + summary['n2_min'] + summary['n3_min']
        summary['rem_nrem_ratio'] = summary['rem_duration_min'] / (summary['n1_min'] + summary['n2_min'] + summary['n3_min'])
        summary['midtime'] = cycle['start'] + (cycle['end'] - cycle['start'])/2
        
        # REM latency (from cycle start to REM onset)
        if cycle['rem_start'] is not None:
            summary['rem_latency_min'] = (cycle['rem_start'] - cycle['start']) / (sf_hypno * 60)
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)

def fill_cycle_gaps(cycles, cycle_hypno, hypno_smooth, gap_assignment='previous'):
    """
    Fill gaps between cycles with appropriate cycle assignments.
    
    Parameters
    ----------
    cycles : list
        List of cycle dictionaries
    cycle_hypno : array
        Current cycle assignments
    hypno_smooth : array
        Smoothed hypnogram
    gap_assignment : str
        How to assign gaps: 'previous', 'next', 'split', 'closest_rem'
    
    Returns
    -------
    cycles : list
        Updated cycle list
    cycle_hypno : array
        Updated cycle assignments
    """
    if len(cycles) <= 1:
        return cycles, cycle_hypno
    
    for i in range(len(cycles) - 1):
        current_cycle = cycles[i]
        next_cycle = cycles[i + 1]
        
        gap_start = current_cycle['end'] + 1
        gap_end = next_cycle['start'] - 1
        
        if gap_start <= gap_end:
            gap_stages = hypno_smooth[gap_start:gap_end + 1]
            
            # Only process if there's sleep in the gap
            if np.any(gap_stages > 0):
                if gap_assignment == 'previous':
                    # Assign all to previous cycle
                    cycle_hypno[gap_start:gap_end + 1] = current_cycle['cycle']
                    current_cycle['end'] = gap_end
                    
                elif gap_assignment == 'next':
                    # Assign all to next cycle
                    cycle_hypno[gap_start:gap_end + 1] = next_cycle['cycle']
                    next_cycle['start'] = gap_start
                    
                elif gap_assignment == 'split':
                    # Split the gap in the middle
                    mid_point = (gap_start + gap_end) // 2
                    cycle_hypno[gap_start:mid_point + 1] = current_cycle['cycle']
                    cycle_hypno[mid_point + 1:gap_end + 1] = next_cycle['cycle']
                    current_cycle['end'] = mid_point
                    next_cycle['start'] = mid_point + 1
                    
                elif gap_assignment == 'closest_rem':
                    # Assign based on proximity to REM periods
                    if current_cycle['rem_end'] is not None and next_cycle['rem_start'] is not None:
                        # Find the point closest to equidistant from both REM periods
                        for idx in range(gap_start, gap_end + 1):
                            dist_to_prev_rem = idx - current_cycle['rem_end']
                            dist_to_next_rem = next_cycle['rem_start'] - idx
                            
                            if dist_to_prev_rem <= dist_to_next_rem:
                                cycle_hypno[idx] = current_cycle['cycle']
                            else:
                                cycle_hypno[idx] = next_cycle['cycle']
                        
                        # Update cycle boundaries
                        current_end = np.where(cycle_hypno[gap_start:gap_end + 1] == current_cycle['cycle'])[0]
                        if len(current_end) > 0:
                            current_cycle['end'] = gap_start + current_end[-1]
                        
                        next_start = np.where(cycle_hypno[gap_start:gap_end + 1] == next_cycle['cycle'])[0]
                        if len(next_start) > 0:
                            next_cycle['start'] = gap_start + next_start[0]
                    else:
                        # Fallback to previous assignment
                        cycle_hypno[gap_start:gap_end + 1] = current_cycle['cycle']
                        current_cycle['end'] = gap_end
    
    # Handle gap after last cycle
    last_cycle = cycles[-1]
    final_sleep = np.where(hypno_smooth[last_cycle['end'] + 1:] > 0)[0]
    if len(final_sleep) > 0:
        final_sleep_start = last_cycle['end'] + 1 + final_sleep[0]
        final_sleep_end = last_cycle['end'] + 1 + final_sleep[-1]
        cycle_hypno[final_sleep_start:final_sleep_end + 1] = last_cycle['cycle']
        last_cycle['end'] = final_sleep_end
    
    return cycles, cycle_hypno


# Update the main function to use gap filling
def detect_sleep_cycles(hypno, sf_hypno=1/30, min_nrem_duration=15, min_rem_duration=1, 
                       awakening_threshold=5, merge_close_rem=10, first_rem_window=120,
                       min_cycle_duration=60, last_cycle_merge_threshold=45,
                       fill_gaps=True, gap_assignment='previous',
                       min_nrem_between_cycles=10):
    """
    Detect sleep cycles from a hypnogram using published criteria.
    
    Based on Feinberg & Floyd (1979) and updated by Jenni & Carskadon (2004),
    with modifications for micro-awakening handling and end-of-night short cycles.
    
    Parameters
    ----------
    hypno : array
        Sleep stage array (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    sf_hypno : float
        Sampling frequency of hypnogram (default 1/30 Hz for 30s epochs)
    min_nrem_duration : int
        Minimum NREM duration in minutes to start a cycle (default 15)
    min_rem_duration : float
        Minimum REM duration in minutes to be counted (default 1)
    awakening_threshold : int
        Maximum awakening duration in minutes to not break cycle (default 5)
    merge_close_rem : int
        Merge REM periods separated by less than this many minutes (default 10)
    first_rem_window : int
        Window in minutes to look for first REM period (default 120)
    min_cycle_duration : int
        Minimum cycle duration in minutes (default 60)
    last_cycle_merge_threshold : int
        Merge last cycle if shorter than this many minutes (default 45)
    fill_gaps : bool
        Whether to fill gaps between cycles (default True)
    gap_assignment : str
        How to assign gaps: 'previous', 'next', 'split', 'closest_rem' (default 'previous')
    min_nrem_between_cycles : int
        Minimum NREM duration in minutes between REM periods to start new cycle (default 10)
    
    Returns
    -------
    cycles : list of dicts
        Each dict contains 'start', 'end', 'nrem_start', 'rem_start', 'rem_end'
    cycle_hypno : array
        Same length as hypno, with cycle number (0 = no cycle)
    """
    
    # Convert parameters from minutes to epochs
    epochs_per_min = int(60 * sf_hypno)
    min_nrem_epochs = min_nrem_duration * epochs_per_min
    min_rem_epochs = int(min_rem_duration * epochs_per_min)
    wake_threshold_epochs = awakening_threshold * epochs_per_min
    merge_rem_epochs = merge_close_rem * epochs_per_min
    first_rem_epochs = first_rem_window * epochs_per_min
    min_cycle_epochs = min_cycle_duration * epochs_per_min
    merge_threshold_epochs = last_cycle_merge_threshold * epochs_per_min
    min_nrem_between_epochs = min_nrem_between_cycles * epochs_per_min
    
    # Step 1: Handle micro-awakenings by filling short wake periods
    hypno_smooth = hypno.copy()
    wake_mask = hypno == 0
    
    # Label continuous wake periods
    wake_labeled, n_wake_periods = label(wake_mask)
    
    # Fill short wake periods (less than threshold)
    for i in range(1, n_wake_periods + 1):
        wake_period = wake_labeled == i
        period_length = np.sum(wake_period)
        
        if period_length < wake_threshold_epochs:
            # Find what sleep stage surrounds this wake period
            indices = np.where(wake_period)[0]
            start_idx = indices[0]
            end_idx = indices[-1]
            
            # Look at stages before and after
            before_stage = hypno_smooth[start_idx - 1] if start_idx > 0 else 0
            after_stage = hypno_smooth[end_idx + 1] if end_idx < len(hypno_smooth) - 1 else 0
            
            # Fill with most common surrounding stage (excluding wake)
            if before_stage > 0 and after_stage > 0:
                fill_stage = before_stage if before_stage == after_stage else min(before_stage, after_stage)
                hypno_smooth[wake_period] = fill_stage
    
    # Step 2: Merge close REM periods (but only if insufficient NREM between them)
    rem_mask = hypno_smooth == 4
    rem_labeled, n_rem_periods = label(rem_mask)
    
    # Identify all REM periods first
    rem_periods = []
    for i in range(1, n_rem_periods + 1):
        rem_indices = np.where(rem_labeled == i)[0]
        if len(rem_indices) >= min_rem_epochs:
            rem_periods.append({
                'label': i,
                'start': rem_indices[0],
                'end': rem_indices[-1]
            })
    
    # Check each pair of consecutive REM periods
    if len(rem_periods) > 1:
        i = 0
        while i < len(rem_periods) - 1:
            rem1 = rem_periods[i]
            rem2 = rem_periods[i + 1]
            
            gap_start = rem1['end'] + 1
            gap_end = rem2['start'] - 1
            gap_length = gap_end - gap_start + 1
            
            if gap_length < merge_rem_epochs:
                # Check NREM content in the gap
                gap_stages = hypno_smooth[gap_start:gap_end + 1]
                nrem_in_gap = np.sum((gap_stages >= 1) & (gap_stages <= 3))
                
                # Only merge if insufficient NREM between REM periods
                if nrem_in_gap < min_nrem_between_epochs:
                    # Merge the REM periods
                    hypno_smooth[gap_start:gap_end + 1] = 4
                    # Update the first REM period and remove the second
                    rem_periods[i]['end'] = rem2['end']
                    rem_periods.pop(i + 1)
                    continue
            
            i += 1
    
    # Step 3: Identify sleep cycles
    cycles = []
    cycle_hypno = np.zeros_like(hypno)
    
    # Find sleep onset (first N2 or N3)
    sleep_onset = np.where((hypno_smooth == 2) | (hypno_smooth == 3))[0]
    if len(sleep_onset) == 0:
        return cycles, cycle_hypno
    
    sleep_onset = sleep_onset[0]
    current_pos = sleep_onset
    cycle_num = 1
    
    while current_pos < len(hypno_smooth):
        # Find start of NREM period (N2 or N3)
        nrem_start = None
        for i in range(current_pos, len(hypno_smooth)):
            if hypno_smooth[i] in [2, 3]:
                nrem_start = i
                break
        
        if nrem_start is None:
            break
        
        # Find continuous NREM period of sufficient length
        nrem_period_start = nrem_start
        nrem_length = 0
        
        for i in range(nrem_start, len(hypno_smooth)):
            if hypno_smooth[i] in [1, 2, 3]:  # Any NREM
                nrem_length += 1
            else:
                break
        
        if nrem_length < min_nrem_epochs:
            current_pos = nrem_period_start + nrem_length + 1
            continue
        
        # Look for REM period
        rem_start = None
        search_end = min(nrem_period_start + first_rem_epochs, len(hypno_smooth))
        
        # For first cycle, look within first REM window
        if cycle_num == 1:
            search_start = nrem_period_start
        else:
            search_start = nrem_period_start + min_nrem_epochs
        
        for i in range(search_start, search_end):
            if hypno_smooth[i] == 4:
                # Check if REM period is long enough
                rem_length = 0
                rem_start_temp = i
                for j in range(i, len(hypno_smooth)):
                    if hypno_smooth[j] == 4:
                        rem_length += 1
                    else:
                        break
                
                if rem_length >= min_rem_epochs or (cycle_num == 1 and rem_length > 0):
                    rem_start = rem_start_temp
                    rem_end = rem_start_temp + rem_length - 1
                    break
        
        # If no REM found but sufficient NREM, still count as cycle (esp. for last cycle)
        if rem_start is None:
            # Check if this is potentially the last cycle
            remaining_sleep = np.sum(hypno_smooth[nrem_period_start:] > 0)
            if remaining_sleep > min_nrem_epochs:
                cycles.append({
                    'cycle': cycle_num,
                    'start': nrem_period_start,
                    'end': len(hypno_smooth) - 1,
                    'nrem_start': nrem_period_start,
                    'rem_start': None,
                    'rem_end': None
                })
                cycle_hypno[nrem_period_start:] = cycle_num
                break
            else:
                current_pos = search_end
                continue
        
        # We have a complete cycle
        cycle_end = rem_end
        
        # Check if there's significant NREM after this REM that would start a new cycle
        next_nrem_start = None
        for i in range(rem_end + 1, min(rem_end + 1 + min_nrem_between_epochs * 2, len(hypno_smooth))):
            if hypno_smooth[i] in [2, 3]:  # N2 or N3
                # Check for continuous NREM
                nrem_length = 0
                for j in range(i, len(hypno_smooth)):
                    if hypno_smooth[j] in [1, 2, 3]:
                        nrem_length += 1
                    else:
                        break
                
                if nrem_length >= min_nrem_between_epochs:
                    # This NREM period is substantial enough to start a new cycle
                    next_nrem_start = i
                    cycle_end = i - 1
                    break
        
        # Extend cycle end to include any trailing N1 (only if no new cycle starting)
        if next_nrem_start is None:
            for i in range(rem_end + 1, len(hypno_smooth)):
                if hypno_smooth[i] == 1:
                    cycle_end = i
                elif hypno_smooth[i] in [2, 3]:
                    # This might be the start of a new cycle
                    break
                else:
                    break
        
        cycles.append({
            'cycle': cycle_num,
            'start': nrem_period_start,
            'end': cycle_end,
            'nrem_start': nrem_period_start,
            'rem_start': rem_start,
            'rem_end': rem_end
        })
        
        cycle_hypno[nrem_period_start:cycle_end + 1] = cycle_num
        
        cycle_num += 1
        current_pos = cycle_end + 1
    
    # Post-processing: Handle short cycles at the end of night
    if len(cycles) >= 2:
        # Check if last cycle is suspiciously short
        last_cycle = cycles[-1]
        last_duration = last_cycle['end'] - last_cycle['start'] + 1
        
        if last_duration < merge_threshold_epochs:
            # Option 1: Merge with previous cycle if close enough
            prev_cycle = cycles[-2]
            gap = last_cycle['start'] - prev_cycle['end']
            
            if gap < merge_rem_epochs:  # If cycles are close
                # Merge the cycles
                prev_cycle['end'] = last_cycle['end']
                if last_cycle['rem_start'] is not None:
                    # Update REM end to include the last REM
                    prev_cycle['rem_end'] = last_cycle['rem_end']
                
                # Update cycle_hypno
                cycle_hypno[cycle_hypno == last_cycle['cycle']] = prev_cycle['cycle']
                
                # Remove the last cycle
                cycles.pop()
            
            # Option 2: If can't merge, check if it's just a REM fragment
            elif last_cycle['rem_start'] is not None and last_cycle['nrem_start'] == last_cycle['rem_start']:
                # This is just a REM fragment, not a real cycle
                cycle_hypno[cycle_hypno == last_cycle['cycle']] = 0
                cycles.pop()
    
    # Additional check: Remove any cycles that are too short
    cycles_to_keep = []
    for cycle in cycles:
        duration = cycle['end'] - cycle['start'] + 1
        if duration >= min_cycle_epochs or cycle['cycle'] == 1:  # Keep first cycle even if short
            cycles_to_keep.append(cycle)
        else:
            # Remove from cycle_hypno
            cycle_hypno[cycle_hypno == cycle['cycle']] = 0
    
    # Renumber cycles if any were removed
    if len(cycles_to_keep) < len(cycles):
        cycles = cycles_to_keep
        # Renumber in cycle_hypno
        for i, cycle in enumerate(cycles):
            old_num = cycle['cycle']
            new_num = i + 1
            cycle['cycle'] = new_num
            cycle_hypno[cycle_hypno == old_num] = new_num
    
    # Fill gaps between cycles with sleep stages
    if len(cycles) > 1:
        for i in range(len(cycles) - 1):
            current_cycle = cycles[i]
            next_cycle = cycles[i + 1]
            
            gap_start = current_cycle['end'] + 1
            gap_end = next_cycle['start'] - 1
            
            if gap_start <= gap_end:
                # Check what's in the gap
                gap_stages = hypno_smooth[gap_start:gap_end + 1]
                
                # If there's any sleep in the gap, assign it
                if np.any(gap_stages > 0):
                    # Assign to the cycle with more similar stages
                    # or to the previous cycle by default
                    cycle_hypno[gap_start:gap_end + 1] = current_cycle['cycle']
                    
                    # Update cycle end time
                    current_cycle['end'] = gap_end
    
    # Apply gap filling if requested
    if fill_gaps:
        cycles, cycle_hypno = fill_cycle_gaps(cycles, cycle_hypno, hypno_smooth, gap_assignment)
    
    return cycles, cycle_hypno

def smooth_hypno_custom(hypno, window=10):
    """
    Smooth a hypnogram array with custom logic:
    - Use the mode of the window
    - If any REM present in window → assign REM
    - Else if any N3 present → assign N3
    """
    hypno = np.asarray(hypno)
    n = len(hypno)
    half_win = window // 2
    smoothed = np.empty(n, dtype=hypno.dtype)

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        window_stages = hypno[start:end]

        if 'REM' in window_stages:
            smoothed[i] = 4
        elif 'N3' in window_stages:
            smoothed[i] = 3
        else:
            smoothed[i] = mode(window_stages, keepdims=False).mode

    return smoothed

def compute_rolling_rem_propensity(hypno, start_time, epoch_sec=30, sol=0, window_min=10, stages=[4], lat_rem=0):
    """
    Compute rolling REM propensity (density) over time from YASA hypnogram data.
    This estimates circadian phase offset by finding the peak REM density time relative to sleep onset latency (SOL).
    
    Parameters:
    - hypno: numpy array of sleep stages (e.g., from YASA: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    - start_time_str: string of recording start time in 'HH:MM' or 'YYYY-MM-DD HH:MM:SS' format
    - epoch_sec: seconds per epoch (default 30)
    - window_min: rolling window size in minutes (default 10, matching your slow wave/spindle plots)
    - stages: list of stages for propensity (default [4] for REM)
    - channels: number of channels (default 1; adjust if multi-channel averaging needed, but your setup uses F7/F8 for REM)
    
    Returns:
    - phase_offset_hr: float, time from SOL to peak REM density in hours (circadian phase marker)
    - peak_time: datetime, absolute clock time of peak
    - sol_time: datetime, absolute clock time of SOL
    - fig: matplotlib figure for visualization (similar to your density plots)
    """
    
    # hypno = session['hypnos_adj']
    # epoch_sec = 1/sf_hypno
    # start_time = session['dts']
    # stages = [4]
    # window_min = 10
    # sol = session['sleep_stats']['SOL_ADJ']
    
    epoch_min = epoch_sec / 60.0
    num_epochs = len(hypno)
        
    # # Binary indicator for target stages (REM=4)
    is_target = np.isin(hypno, stages).astype(int)
    
    # DataFrame for rolling calculations
    df = pd.DataFrame({'is_target': is_target})
    
    # Window in epochs
    window_epochs = int(window_min / epoch_min)
    
    # Raw count: number of target epochs in rolling window
    df['raw_count'] = df['is_target'].rolling(window=window_epochs, center=True, min_periods=1).sum()
    
    # Density: propensity as fraction (0-1), equivalent to REM time fraction in window
    # For multi-channel, could divide by channels, but your REM detection uses F7/F8 effectively as one
    df['density'] = df['raw_count'] / window_epochs
    
    # Time array: minutes since start
    times_min = np.arange(num_epochs) * epoch_min
    df['time_min'] = times_min
    
    # Mean density for reference line
    mean_density = df['density'].mean()
    
    # Find peak density
    peak_idx = df['density'].idxmax()
    peak_time_min = times_min[peak_idx]
    peak_time = start_time + timedelta(minutes=peak_time_min)
    
    # Phase offset: time from SOL to peak in hours
    phase_offset_min = peak_time_min - sol
    phase_offset_hr = phase_offset_min / 60.0
    
    # Plot similar to your slow wave/spindle density plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Upper plot: Density with shading and mean line
    ax1.plot(times_min / 60, df['density'], color='blue', linewidth=1.5)
    ax1.fill_between(times_min / 60, df['density'], color='lightblue', alpha=0.5)
    ax1.axhline(mean_density, color='red', linestyle='--', label=f'Mean: {mean_density:.2f}')
    ax1.set_ylabel('REM Density (fraction/channel)')
    ax1.set_title(f'Rolling {window_min}-Minute REM Propensity Over Time\nPeak at {peak_time.strftime("%H:%M")} (Offset: {phase_offset_hr:.1f}h from SOL), REM Latency: {(lat_rem/60):.1f}h')
    ax1.legend()
    ax1.grid(True)
    
    # Lower plot: Raw count
    ax2.plot(times_min / 60, df['raw_count'], color='red', linewidth=1.5)
    ax2.set_ylabel('Raw REM Count (per 10-min window)')
    ax2.set_xlabel('Time (hours since start)')
    ax2.grid(True)
    
    # Vertical line at peak
    ax1.axvline(peak_time_min / 60, color='green', linestyle='--')
    ax2.axvline(peak_time_min / 60, color='green', linestyle='--')
    ax1.axvline(lat_rem / 60, color='cyan', linestyle='--')
    ax2.axvline(lat_rem / 60, color='cyan', linestyle='--')
    
    plt.tight_layout()
    
    return phase_offset_hr, peak_time, fig

def compute_cumulative_rem_phase(hypno, start_time, epoch_sec=30, smooth_window=15, min_slope_threshold=0.5, sol=0):
    """
    Compute rolling REM propensity (density) over time from YASA hypnogram data.
    This estimates circadian phase offset by finding the peak REM density time relative to sleep onset latency (SOL).
    
    Parameters:
    - hypno: numpy array of sleep stages (e.g., from YASA: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    - start_time_str: string of recording start time in 'HH:MM' or 'YYYY-MM-DD HH:MM:SS' format
    - epoch_sec: seconds per epoch (default 30)
    - smooth_window: window size for Savitzky-Golay smoothing (in epochs, default 15 ~7.5 min)
    - min_slope_threshold: fraction of max slope for inflection (default 0.5)
    
    Returns:
    - phase_offset_hr: float, time from SOL to peak REM density in hours (circadian phase marker)
    - peak_time: datetime, absolute clock time of peak
    - sol_time: datetime, absolute clock time of SOL
    - fig: matplotlib figure for visualization (similar to your density plots)
    """
    
    # hypno = session['hypnos_adj']
    # epoch_sec = 1/sf_hypno
    # start_time = session['dts']
    # sol = session['sleep_stats']['SOL_ADJ']
    
    epoch_min = epoch_sec / 60.0
    num_epochs = len(hypno)
        
    # Cumulative REM duration (minutes)
    is_rem = (hypno == 4).astype(int)  # REM = 4
    cum_rem = np.cumsum(is_rem) * epoch_min
    
    # Time array in minutes since start
    times_min = np.arange(num_epochs) * epoch_min
    
    # Smooth the cumulative REM curve (Savitzky-Golay filter for better trend)
    from scipy.signal import savgol_filter  # For better smoothing
    cum_rem_smooth = savgol_filter(cum_rem, window_length=smooth_window, polyorder=2)
    
    # Compute first derivative (rate of change)
    deriv1 = np.gradient(cum_rem_smooth)
    
    # Find the maximum slope
    max_slope = np.max(deriv1)
    threshold_slope = max_slope * min_slope_threshold
    
    # Inflection point: where slope exceeds threshold after initial rise
    # Start search after first REM to avoid early noise
    first_rem_idx = np.where(is_rem == 1)[0][0]
    relevant_deriv = deriv1[first_rem_idx:]
    relevant_times = times_min[first_rem_idx:]
    inflection_idx = first_rem_idx + np.where(relevant_deriv >= threshold_slope)[0][0] if len(np.where(relevant_deriv >= threshold_slope)[0]) > 0 else first_rem_idx
    
    inflection_time_min = times_min[inflection_idx]
    inflection_time = start_time + timedelta(minutes=inflection_time_min)
    phase_offset_min = inflection_time_min - sol
    phase_offset_hr = phase_offset_min / 60.0
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times_min / 60, cum_rem, color='blue', label='Cumulative REM (min)', alpha=0.3)
    ax.plot(times_min / 60, cum_rem_smooth, color='blue', label='Smoothed Cumulative REM', linewidth=2)
    ax.plot(times_min[inflection_idx] / 60, cum_rem_smooth[inflection_idx], 'ro', label='Inflection Point')
    ax.axvline(sol / 60, color='green', linestyle='--', label=f'SOL {sol}m')
    ax.set_xlabel('Time (hours since start)')
    ax.set_ylabel('Cumulative REM Duration (minutes)')
    ax.set_title(f'Cumulative REM Over Time\nInflection at {inflection_time.strftime("%H:%M")} (Offset: {phase_offset_hr:.1f}h from SOL)')
    ax.legend()
    ax.grid(True)
    
    return phase_offset_hr, inflection_time, fig