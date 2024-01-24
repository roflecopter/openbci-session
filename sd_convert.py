import numpy as np
import scipy as sp
import datetime
import math
import os
import json
import csv    
import pandas as pd
import sqlite3
import re

from pyedflib import highlevel
from contextlib import closing
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator


working_dir = '/path/to/openbci-psg'
working_dir = '/Volumes/Data/Storage/Dev/openbci-psg'
sd_dir = '/Volumes/OBCI'
sd_dir = '/Volumes/Data/Storage/Dev/openbci-psg/sample'

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


# # https://github.com/miladinovic/OpenBCI_SD_HEX2EEGLAB/tree/master
def adc_test():
    # https://docs.openbci.com/Cyton/CytonSDCard/#data-logging-format
    
    # https://openbci.com/forum/index.php?p=/discussion/3146/question-on-external-trigger-for-openbci-cyton-location-in-recording
    # The normal data channels are 24 bits in size, and shown as 6 hex digits. 
    # The three 'Aux' channels are only 16 bits, and shown as 4 hex digits. When you press and release the button, you should see a bit turn on and off in the Aux digits.
    
    # https://openbci.com/forum/index.php?p=/discussion/1653/script-to-convert-sd-card-file
    # F260C9, this is converted as: -19954.006, HOW?
    eeg_hex = 'F260C9'; eeg_adc_value = interpret24bitAsInt32(eeg_hex); eeg_uv = adc_v_bci(eeg_adc_value)
    print(f'{eeg_hex} > {eeg_adc_value} > {eeg_uv}, {round(eeg_uv,3) == -19954.006}')
    
    accelScale = 0.002 / (pow (2, 4));
    aux_hex = 'FAE0'; aux_adc_value = interpret16bitAsInt32(aux_hex); aux_g = aux_adc_value * accelScale 
    print(f'{aux_hex} > {aux_adc_value} > {aux_g}')

def processLine(split_line, n_ch, n_acc):    
    values_array = []
    for i in range(1, len(split_line)):
        value = split_line[i]
        if i <= n_ch:
            value = interpret24bitAsInt32(value)
        else:
            value = interpret16bitAsInt32(value)        
        values_array.append(value)
    return values_array

def process_file(file_path, n_ch = 8, n_acc = 3, sf = 250):
    with open(file_path, 'r') as file:
        result = []
        i = 0
        stops_n = 0
        stops = []
        stops_at = []
        while True:
            line = file.readline()
            if not line:
                print(f'EOF, no line at {i}')
                break  # End of file
            split_line = line.strip().split(',')
            if split_line[0].startswith('%Total time'):
                print(f'recording full at {i} / {line}')
                break # SD recording complete
            if len(split_line) == 1 and split_line[0].startswith('%'):
                stops_n += 1
                stops.append(i)
            elif len(split_line) == 1 and not split_line[0].startswith('%'):
                if stops[-1] == i - 1:
                    print(f'stopped at {i} / {line}')
                    stops_at.append(interpret24bitAsInt32('00' + line))
            elif (len(split_line) > 3) and (len(split_line) <= n_ch + n_acc + 1):
                values = processLine(split_line, n_ch, n_acc)
                if len(values) == (n_ch + n_acc):
                    to_add = values
                elif len(values) == (n_ch):
                    to_add = values + [np.nan, np.nan, np.nan]
                else:
                    to_add = [0] * (n_ch + n_acc)
                result.append(to_add)
            i += 1
            if i % (sf*60)== 0:
                print(f"Processing... {i/(sf*60)}m")
            if i % (sf*600)== 0:
                print(f'Processing... {round(i/(sf*60))}m')
        return result, stops, stops_at

def obci_bdf(bci_signals, sf, channels, user, gender, dts, birthday, gain):
    header = highlevel.make_header(patientname=user, gender=gender, equipment=device, 
      startdate = dts, birthdate = datetime.datetime.strptime(birthday, '%Y_%m_%d-%H_%M_%S'))
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
            signal_headers.append({"label": channel, "dimension": "g", "sample_rate": sf, "sample_frequency": sf, 'physical_max': acc_ph_max, 'physical_min': acc_ph_min, 'digital_max': acc_dig_max, 'digital_min': acc_dig_min, 'transducer': 'MEMS', 'prefilter': ''})
        else: 
            # EEG
            # https://openbci.com/forum/index.php?p=/discussion/comment/8122
            ch_dig_min = -8388608; ch_dig_max = 8388607
            ch_ph_min = -187500; ch_ph_max = 187500
            channel_data = np.vectorize(adc_v_bci)(channel_data)
            channel_data[channel_data > ch_ph_max] = ch_ph_max
            channel_data[channel_data < ch_ph_min] = ch_ph_min
            signals.append(channel_data)
            signal_headers.append({"label": channel, "dimension": "uV", "sample_rate": sf, "sample_frequency": sf, 'physical_max': ch_ph_max, 'physical_min': ch_ph_min, 'digital_max': ch_dig_max, 'digital_min': ch_dig_min, 'transducer': 'Gold-Cup Electrode', 'prefilter': ''})
        processed = len(channel_data)
    return([header, signal_headers, signals, processed])

# ADC to uV conversion for ADS1299
ADS1299_BITS = (2**23-1)
ADS1299_GAIN = 24
V_Factor = 1000000 # uV
def adc_v_bci(signal, ADS1299_VREF = 4.5):
    k = ADS1299_VREF / ADS1299_BITS / ADS1299_GAIN * V_Factor
    return signal * k

sessions = None
session = None
with closing(sqlite3.connect(os.path.join(working_dir, 'data','sessions.db'), timeout=10)) as con:
    with con:
        with closing(con.cursor()) as cur:
            sql = 'SELECT * FROM Sessions'
            cur.execute(sql)
            sessions = cur.fetchall()

files = [file for file in os.listdir(sd_dir) if file.endswith('.TXT')]
print(f'sd: {files}')
if files:
    files.sort(reverse=True)
    file_name = files[0]
    file_path = os.path.join(sd_dir, file_name)
    print(f'converting: {file_name}')

    # default session settings
    channels = {'F8-T3':0,'F7-T3':1,'O2-T3':2, 'O1-T3':3, 'T4-T3':4}
    sf = 500
    gain = 24
    dts = datetime.datetime.now()
    
    # look for previously saved session start and parameters
    if sessions is not None:
        session = [session for session in sessions 
                   if len(session) > 2 and session[1] == file_name]
        if len(session) > 0:
            settings = json.loads(session[0][2])
            sf = settings['sf']
            gain = settings['gain']
            channels = settings['channels']
            dts = datetime.datetime.strptime(session[0][0], '%Y-%m-%d %H:%M:%S')
    
    n_ch = 8
    channels['ACC_X'] = n_ch
    channels['ACC_Y'] = n_ch + 1
    channels['ACC_Z'] = n_ch + 2
    
    # BDF info
    user = 'User'
    gender = 'Male'
    birthday = '1980_01_01-12_00_00'
    brand = 'OpenBCI'
    device = 'Cyton'

    bci_signals, stops, stops_at = process_file(file_path)
    bci_signals = np.array(bci_signals)

    # set proper gain for correct ADC conversion
    ADS1299_GAIN = gain
    header, signal_headers, signals, processed = obci_bdf(bci_signals, sf, channels, user, gender, dts, birthday, gain)
    file_bdf = os.path.join(working_dir, 'data', 
       file_name + '_' + dts.strftime('%Y-%m-%d %H-%M-%S') + '.bdf')
    res = highlevel.write_edf(file_bdf, signals, signal_headers, header)
    if res:
        print(f'BDF saved to {file_bdf}')
    
    header = ['ts'] + [channel for channel in channels]

    # create timestamps for csv and append to signals
    ts = dts + pd.to_timedelta(list(np.arange(len(signals[0]))), unit='ms')*1000/sf
    ts = ts.astype(np.int64)/1000000000
    signals = np.insert(signals, 0, ts, axis=0)
    
    # Writing to a CSV file
    file_csv = os.path.join(working_dir, 'data', file_name + '_' + str(sf) + 'Hz_' + dts.strftime('%Y-%m-%d %H-%M-%S') + '.csv')
    with open(file_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in zip(*signals):
            writer.writerow(row)
        print(f'CSV saved to {file_csv}')

