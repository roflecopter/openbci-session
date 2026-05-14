import csv
import datetime
import json
import math
import numpy as np
import os
import pandas as pd
import re
import sqlite3
import yaml

from contextlib import closing
from pyedflib import highlevel
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator

# =====================================================================
# %CKPT parsing + intra-file gap inference (library functions).
#
# These two functions are the canonical implementation of the firmware's
# `%CKPT` heartbeat line format and the algorithm that infers SD-recovery
# sample-drop windows from consecutive heartbeats. Importable from this
# module; private py-qs-data/openbci_functions.py imports them via path
# injection so both pipelines stay in lock-step on the firmware's evolving
# counter set (`t/b/e/r/n/o/x` as of 2026-05-13).
#
# Field semantics + firmware reference:
#   github.com/roflecopter/OpenBCI_Cyton_Library_SD/README.md
#   "SD reliability and observability" section.
# =====================================================================

def parse_ckpt_line(line):
    """Parse a `%CKPT t=<ms> b=<block> e=<errs> r=<retries> n=<reinits> o=<over> x=<extretries>` line.

    Returns dict with int values for any of {t, b, e, r, n, o, x} actually
    present. Returns None if the line is malformed (no `%CKPT ` prefix or
    no parseable key=value pairs). Tolerant of trailing/extra whitespace
    and unknown keys (silently ignored — forward-compat with future
    additions like a tunable-summary `T=<hash>` key per the firmware
    ROADMAP). Pre-2026-05-08 firmware emits no `%CKPT` lines so this
    function is naturally a no-op on stock-firmware recordings.
    """
    if not line.startswith('%CKPT '):
        return None
    payload = line[len('%CKPT '):].strip()
    if not payload:
        return None
    out = {}
    for kv in payload.split():
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        if k not in ('t', 'b', 'e', 'r', 'n', 'o', 'x'):
            continue  # unknown key — skip, forward-compat
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out if out else None


def compute_intra_file_gaps(ckpts, sf, jitter_tolerance_samples=5):
    """From a list of %CKPT dicts (with `sample_idx` set by process_file),
    infer intra-file sample gaps caused by SD-recovery sample drops.

    Each consecutive (ckpt_k, ckpt_{k+1}) pair gives us:
      * Δt_ms = ckpt_{k+1}['t'] - ckpt_k['t']  — wall-clock elapsed
      * Δsamples = ckpt_{k+1}['sample_idx'] - ckpt_k['sample_idx']  — recorded
      * expected = round(Δt_ms * sf / 1000)
      * missing  = expected - Δsamples

    If `missing > jitter_tolerance_samples` a gap is recorded at
    `sample_idx = ckpt_{k+1}['sample_idx']` — i.e. JUST BEFORE the
    post-recovery sample stream resumes.

    Returns a list of dicts:
      [{
        'sample_idx': int,        # where to insert zero rows (in pre-gap-fill coords)
        'n_samples': int,         # how many zero rows to insert
        'gap_ms': int,            # implied DURATION OF THE GAP (= missing*1000/sf), NOT the full CKPT-to-CKPT Δt
        't_start_ms': int,        # leading-CKPT t_ms
        't_end_ms': int,          # trailing-CKPT t_ms (when gap was discovered)
        'errs_delta': int,        # b['e'] - a['e']  (informational)
        'extretries_delta': int,  # b['x'] - a['x']  (informational; 0 on pre-2026-05-13 firmware)
      }, ...]

    Defensive — returns [] on:
      * sf <= 0 (no sample period to compare against)
      * fewer than 2 ckpts
      * any pair with missing keys, non-monotonic t/sample_idx, or missing within jitter
    """
    gaps = []
    try:
        sf_f = float(sf)
    except (TypeError, ValueError):
        return gaps
    if sf_f <= 0:
        return gaps
    try:
        tol = int(jitter_tolerance_samples)
    except (TypeError, ValueError):
        tol = 0
    if tol < 0:
        tol = 0
    if not ckpts or len(ckpts) < 2:
        return gaps
    for k in range(len(ckpts) - 1):
        a, b = ckpts[k], ckpts[k + 1]
        if 't' not in a or 't' not in b:
            continue
        if 'sample_idx' not in a or 'sample_idx' not in b:
            continue
        dt_ms = b['t'] - a['t']
        if dt_ms <= 0:
            continue
        ds = b['sample_idx'] - a['sample_idx']
        if ds < 0:
            continue
        expected = int(round(dt_ms * sf_f / 1000.0))
        missing = expected - ds
        if missing <= tol:
            continue

        def _delta(key):
            if key in a and key in b:
                try:
                    d = int(b[key]) - int(a[key])
                    return d if d >= 0 else 0
                except (TypeError, ValueError):
                    return 0
            return 0

        gaps.append({
            'sample_idx': b['sample_idx'],
            'n_samples': missing,
            'gap_ms': int(round(missing * 1000.0 / sf_f)),
            't_start_ms': a['t'],
            't_end_ms': b['t'],
            'errs_delta': _delta('e'),
            'extretries_delta': _delta('x'),
        })
    return gaps


def sc_interp1d_nan(y, m = 'pchip', extrapolate = False):
  y = np.array(y)
  x = np.arange(len(y))
  nan_indices = np.isnan(y); y_interp = []
  if m == 'akima':
    akima_interp = Akima1DInterpolator(x[~nan_indices], y[~nan_indices])
    y_interp = akima_interp(x, extrapolate = extrapolate)
  elif m == 'pchip':
    pchip_interp = PchipInterpolator(x[~nan_indices], y[~nan_indices])
    y_interp = pchip_interp(x, extrapolate = extrapolate)
  elif m == 'cubic':
    cubic_interp = CubicSpline(x[~nan_indices], y[~nan_indices])
    y_interp = cubic_interp(x, extrapolate = extrapolate)
  elif m == 'natural':
    cubic_interp = CubicSpline(x[~nan_indices], y[~nan_indices], bc_type = 'natural')
    y_interp = cubic_interp(x, extrapolate = extrapolate)
  elif m == 'np_linear':
    y_interp = np.interp(x, x[~nan_indices], y[~nan_indices])
  elif m == 'sc_linear':
    f = interp1d(x[~nan_indices], y[~nan_indices], bounds_error=False, kind='linear',assume_sorted=True,copy=False)
    y_interp = f(np.arange(y.shape[0]))
  else:
    f = interp1d(x[~nan_indices], y[~nan_indices], bounds_error=False, kind=m,assume_sorted=True,copy=False)
    y_interp = f(np.arange(y.shape[0]))
  return(y_interp)

ADS1299_BITS = (2**23-1)
ADS1299_GAIN = 24
V_Factor = 1000000 # uV
def adc_v_bci(signal, ADS1299_VREF = 4.5):
    k = ADS1299_VREF / ADS1299_BITS / ADS1299_GAIN * V_Factor
    return signal * k

def interpret24bitAsInt32(hex_str):
    if len(hex_str) == 6:
        # Convert the hex string to a byte array
        byte_array = bytes.fromhex(hex_str)
    
        # Convert the bytes to a 24-bit integer
        new_int = (byte_array[0] << 16) | (byte_array[1] << 8) | byte_array[2]
    
        # Check if the 24th bit is set (negative number in 2's complement)
        if new_int & 0x00800000:
            new_int |= 0xFF000000  # Extend the sign bit to 32 bits
        else:
            new_int &= 0x00FFFFFF  # Ensure the number is positive
    
        # Adjust for Python's handling of integers larger than 32 bits
        if new_int & 0x80000000:
            new_int -= 0x100000000
    
        return new_int
    else:
        return np.nan

def interpret16bitAsInt32(hex_str):
    if len(hex_str) == 4:
        # Convert the hex string to a byte array
        byte_array = bytes.fromhex(hex_str)
    
        # Convert the bytes to a 16-bit integer
        new_int = (byte_array[0] << 8) | byte_array[1]
    
        # Check if the 16th bit is set (negative number in 2's complement)
        if new_int & 0x00008000:
            new_int |= 0xFFFF0000  # Extend the sign bit to 32 bits
        else:
            new_int &= 0x0000FFFF  # Ensure the number is positive
    
        # Adjust for Python's handling of integers larger than 32 bits
        if new_int & 0x80000000:
            new_int -= 0x100000000
    
        return new_int
    else:
        return np.nan


def process_line(split_line, n_ch, n_acc):    
    values_array = []
    for i in range(1, len(split_line)):
        value = split_line[i]
        if i <= n_ch:
            value = interpret24bitAsInt32(value)
        else:
            value = interpret16bitAsInt32(value)        
        values_array.append(value)
    return values_array

def process_file(file_path, n_ch = 8, n_acc = 3, sf = 250, verbose=False,
                 return_ckpts=False):
    """Parse one OBCI_*.TXT into a list of `[ch1..ch_n, acc_x, acc_y, acc_z]`
    sample rows, plus stop-marker bookkeeping.

    Default return: `(result, stops, stops_at)` — unchanged from pre-2026-05-14.

    When `return_ckpts=True`: returns `(result, stops, stops_at, ckpts)` where
    `ckpts` is a list of dicts (one per `%CKPT` line found), each containing
    parsed `t/b/e/r/n/o/x` int fields plus a `sample_idx` key holding the
    count of data samples in `result` AT THE MOMENT the `%CKPT` line was
    encountered (i.e. the post-CKPT sample stream starts at index sample_idx).

    The `sample_idx` field is what makes downstream `compute_intra_file_gaps`
    possible: between two consecutive `%CKPT`s we know both the wall-clock
    delta (Δt_ms) and the data-sample delta (Δsample_idx) — divergence beyond
    sample-period jitter implies samples were dropped during inline SD
    recovery and lets us zero-pad the gap to keep wall-clock alignment stable.

    Pre-2026-05-08 firmware emits no `%CKPT` lines — `ckpts` returns [] and
    downstream gap inference is a no-op. Post-2026-05-08 firmware emits one
    `%CKPT` line ~every 60 s at sample boundaries.
    """
    with open(file_path, 'r') as file:
        file = open(file_path, 'r')
        result = []
        ckpts = [] if return_ckpts else None
        i = 0
        stops_n = 0
        stops = []
        stops_at = []
        while True:
            line = file.readline()
            if (i == 0) and (len(line) > 30) and not line.startswith('%'):
                print(f'File seems to be corrupted, line len {len(line)}')
                break  # End of file
            if not line:
                print(f'EOF, no line at {i}')
                break  # End of file
            if line.startswith('%META '):  # firmware-embedded metadata, skip from data path
                i += 1
                continue
            if ckpts is not None and line.startswith('%CKPT '):
                # Record sample_idx BEFORE the generic single-%-line branch
                # below also bumps `stops`. parse_ckpt_line returns None on a
                # truncated/garbage %CKPT line — we still add the stop entry
                # in the branch below so existing stops-list consumers don't
                # see a hole, we just skip the ckpts record.
                parsed = parse_ckpt_line(line)
                if parsed is not None:
                    parsed['sample_idx'] = len(result)
                    ckpts.append(parsed)
                # fall through to stops bookkeeping
            split_line = line.strip().split(',')
            if split_line[0].startswith('%Total time'):
                print(f'recording full at {i} / {line}')
                break # SD recording complete
            if len(split_line) == 1 and split_line[0].startswith('%'):
                stops_n += 1
                stops.append(i)
            elif len(split_line) == 1 and not split_line[0].startswith('%'):
                if stops and stops[-1] == i - 1:
                    print(f'stopped at {i} / {line}')
                    stops_at.append(interpret24bitAsInt32('00' + line))
            elif (len(split_line) > 3) and (len(split_line) <= n_ch + n_acc + 1):
                values = process_line(split_line, n_ch, n_acc)
                if len(values) == (n_ch + n_acc):
                    to_add = values
                elif len(values) == (n_ch):
                    to_add = values + [np.nan, np.nan, np.nan]
                else:
                    to_add = [0] * (n_ch + n_acc)
                result.append(to_add)
            elif (len(split_line) > 3) and (len(split_line) <= 16 + n_acc + 1):
                # with 8 ch recorded with daisy on file contains 16ch + 3acc
                values = process_line(split_line, 16, n_acc)
                if len(values) >= 16: del values[8:16] # remove unused daisy channels 9-16
                if len(values) == (n_ch + n_acc):
                    to_add = values
                elif len(values) == (n_ch):
                    to_add = values + [np.nan, np.nan, np.nan]
                else:
                    to_add = [0] * (n_ch + n_acc)
                result.append(to_add)
            i += 1
            if i % (sf*60)== 0:
                if verbose:
                    print(f"Processing... {i/(sf*60)}m, n_samples: {len(result)}, last:{result[-1]}")
            if i % (sf*600)== 0:
                print(f'Processing... {round(i/(sf*60))}m, n_samples: {len(result)}, last:{result[-1]}')
        if return_ckpts:
            return result, stops, stops_at, ckpts
        return result, stops, stops_at

def obci_bdf(bci_signals, sf, channels, user, gender, dts, birthday, gain, electrode, activity, device):
    header = highlevel.make_header(patientname=user, gender=gender, equipment=device + ', ' + activity, 
      startdate = dts, birthdate = datetime.datetime.strptime(birthday, cfg['sql_dt_format']))
    total_samples = math.floor(len(bci_signals) / sf)
    signals = []; signal_headers = []
    bci_signals = np.array(bci_signals)
    for channel in channels:
        ch_i = channels[channel]
        channel_data = bci_signals[:,ch_i]
        channel_data = channel_data[range(0,total_samples*sf)]
        if re.search('ACC',channel) is not None:
            # ACC
            acc_dig_min = -4096; acc_dig_max = 4095
            acc_ph_min = -4; acc_ph_max = 4
            acc_interpolated = sc_interp1d_nan(channel_data, m = 'np_linear')
            accelScale = 0.002 / (pow (2, 4));
            signals.append(acc_interpolated * accelScale)
            signal_headers.append({"label": channel, "dimension": "g", "sample_frequency": sf, 'physical_max': acc_ph_max, 'physical_min': acc_ph_min, 'digital_max': acc_dig_max, 'digital_min': acc_dig_min, 'transducer': 'MEMS', 'prefilter': ''})
        else: 
            # EEG
            # https://openbci.com/forum/index.php?p=/discussion/comment/8122
            ch_dig_min = -8388608; ch_dig_max = 8388607
            ch_ph_min = -187500; ch_ph_max = 187500
            channel_data = np.vectorize(adc_v_bci)(channel_data)
            channel_data[channel_data > ch_ph_max] = ch_ph_max
            channel_data[channel_data < ch_ph_min] = ch_ph_min
            signals.append(channel_data)
            signal_headers.append({"label": channel, "dimension": "uV", "sample_frequency": sf, 'physical_max': ch_ph_max, 'physical_min': ch_ph_min, 'digital_max': ch_dig_max, 'digital_min': ch_dig_min, 'transducer': electrode, 'prefilter': ''})
        processed = len(channel_data)
    return([header, signal_headers, signals, processed])

def read_txt_meta(file_path, max_bytes=4096):
    """Scan the first max_bytes of TXT for the firmware-embedded %META {...} JSON line.
    Returns dict or None. Cheap (single bounded read), tolerant of any number of
    leading %-prefixed control lines, robust against partial-block-write artefacts:
    if multiple %META candidates exist, returns the first one that parses as JSON
    (so a corrupted partial fragment doesn't shadow a clean retry attempt)."""
    try:
        with open(file_path, 'rb') as f:
            head = f.read(max_bytes)
    except (IOError, OSError):
        return None
    head_str = head.decode('utf-8', errors='ignore')
    metas = re.findall(r'^%META (.+)$', head_str, re.MULTILINE)
    if not metas:
        return None
    if len(metas) > 1:
        print(f'WARNING: {len(metas)} %META candidates in {file_path}, picking first valid JSON')
    for meta_str in metas:
        try:
            return json.loads(meta_str.strip())
        except json.JSONDecodeError:
            continue
    return None

def session_lookup(file, sessions=None, defaults=None, file_path=None):
    electrode_type = 'Gold Cup OpenBCI, Ten20'
    activity = 'sleep'
    ch_n = 8
    dts = datetime.datetime.now()
    if defaults is None:
        channels = {
            'F8-T5':0, 'F7-T5':1, 'O2-T5':2, 'O1-T5':3,
            'T8-T5':4, 'T7-T5':5, 'AFz-T5':6, 'T6-T5':7}
        emg_channels = {}
        sf = 500
        device = 'cyton'
        ground = 'Fpz'
        settings = {
            'gain':24, 'channels':channels, 'sf': sf,
            'ground': ground, 'electrode': electrode_type,
            'emg_ch': emg_channels,
            'ch_n': ch_n, 'activity': activity, 'device': device
            }
    else:
        settings = defaults

    # Priority 1 — firmware-embedded %META line in the TXT itself
    if file_path is not None:
        meta = read_txt_meta(file_path)
        if meta is not None:
            print(f'using %META line from {file}')
            for k in ('gain', 'channels', 'sf', 'ground', 'electrode',
                      'emg_ch', 'ch_n', 'activity', 'device'):
                if k in meta:
                    settings[k] = meta[k]
            if 'dts' in meta:
                try:
                    dts = datetime.datetime.strptime(meta['dts'], cfg['sql_dt_format'])
                except (ValueError, TypeError):
                    pass
            return dts, settings

    # Priority 2 — sqlite Sessions row (legacy fallback for pre-firmware-patch recordings)
    if sessions is not None:
        session_file = sessions.loc[sessions['file'] == file]
        if len(session_file) > 0:
            session = session_file.loc[session_file['dt'].idxmax()]
            settings = json.loads(session['settings'])
            if 'electrode' not in settings:
                settings['electrode'] = electrode_type
            if 'activity' not in settings:
                settings['activity'] = activity
            if 'ch_n' not in settings:
                settings['ch_n'] = ch_n
            dts = datetime.datetime.strptime(session['dts'], cfg['sql_dt_format'])
            print(f'using sqlite Sessions row for {file}')
            return dts, settings

    # Priority 3 — defaults
    print(f'using defaults for {file}')
    return dts, settings
        
def get_sessions(session_db):
    sessions = None
    with closing(sqlite3.connect(session_db, timeout=10)) as con:
        with con:
            with closing(con.cursor()) as cur:
                sql = 'SELECT * FROM Sessions'
                cur.execute(sql)
                sessions = pd.DataFrame(cur.fetchall(), columns=['dts','file','settings'])
                sessions['dt'] = pd.to_datetime(sessions['dts'])
    return sessions

if __name__ == "__main__":
    # config, if relative path not working then use explicit path to working dir (repo dir with scripts and yml) or modify working directory in IDE/GUI settings
    # working_dir = '/path/to/openbci-session'
    working_dir = os.getcwd()
    cfg_file = os.path.join(working_dir, "sd_convert.yml")

    # rename sleep_analysis.yml.sample to sleep_analysis.yml and set directories
    with open(cfg_file, "r") as yamlfile:
        cfg_base = yaml.load(yamlfile, Loader=yaml.FullLoader)
        cfg = cfg_base['default']

    # info will be embedded into output BDF
    user = cfg['user']
    gender = cfg['gender']
    birthday = cfg['birthday']
    brand = cfg['brand']

    save_csv = cfg['save_csv'] # set to True to additionally export data to CSV (big size for long session / sleep)

    sessions = get_sessions(os.path.join(cfg['session_dir'],cfg['session_file']))

    files = [file for file in os.listdir(cfg['sd_dir']) if file.endswith('.TXT')]
    print(f'sd: {files}')

    if files:
        files.sort(reverse=True)
        file_name = files[0] # process only single last file
        print(f'Process {file_name}')
        file_path = os.path.join(cfg['sd_dir'], file_name)
        print(f'converting: {file_name}')
        dts, settings = session_lookup(file_name, sessions, file_path=file_path)
        print(f'dt: {dts}, settings: {settings}')
        settings['channels']['ACC_X'] = settings['ch_n']
        settings['channels']['ACC_Y'] = settings['ch_n'] + 1
        settings['channels']['ACC_Z'] = settings['ch_n'] + 2

        bci_signals, stops, stops_at = process_file(file_path, n_ch=settings['ch_n'],
                                                    n_acc=3, sf=settings['sf'])
        bci_signals = np.array(bci_signals)

        # set proper gain for correct ADC conversion
        ADS1299_GAIN = settings['gain'] # might not work
        header, signal_headers, signals, processed = obci_bdf(bci_signals, settings['sf'], settings['channels'], user, gender, dts, birthday, settings['gain'], settings['electrode'], settings['activity'], settings['device'])
        file_bdf = os.path.join(cfg['data_dir'],
           file_name + '_' + dts.strftime(cfg['file_dt_format']) + '.bdf')
        res = highlevel.write_edf(file_bdf, signals, signal_headers, header)
        if res:
            print(f'BDF saved to {file_bdf}')

        header = ['ts'] + [channel for channel in settings['channels']]

        if save_csv:
            # create timestamps for csv and append to signals
            ts = dts + pd.to_timedelta(list(np.arange(len(signals[0]))), unit='ms')*1000/settings['sf']
            ts = ts.astype(np.int64)/1000000000
            signals = np.insert(signals, 0, ts, axis=0)

            # Writing to a CSV file
            file_csv = os.path.join(cfg['data_dir'], file_name + '_' + str(settings['sf']) + 'Hz_' + dts.strftime(cfg['file_dt_format']) + '.csv')
            with open(file_csv, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for row in zip(*signals):
                    writer.writerow(row)
                print(f'CSV saved to {file_csv}')
