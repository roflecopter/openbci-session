import numpy as np
import datetime
import math
import os
import json
from pyedflib import highlevel
import sqlite3
from contextlib import closing

working_dir = '/path/to/openbci-psg'
sd_dir = '/Volumes/BCI'

ADS1299_BITS = (2**23-1)
V_Factor = 1000000 # uV
def adc_v_bci(signal, ADS1299_VREF = 4.5):
    k = ADS1299_VREF / ADS1299_BITS / ADS1299_GAIN * V_Factor
    return signal * k

def parseInt24Hex(hex_value):
    if len(hex_value) < 16:
        value_dec = int(hex_value, 16) if hex_value[0] != 'F' else int(hex_value, 16) - 2**32
        return value_dec
    return 0

def processLine(split_line):    
    values_array = []
    for i in range(1, len(split_line)):
        value = split_line[i]
        if i <= 16:
            channel_value = 'FF' + value if value[0] > '7' else '00' + value
            value = parseInt24Hex(channel_value)
        else:
            aux_value = 'FF' + value if value[0] == 'F' else '00' + value
            value = parseInt24Hex(aux_value)        
        values_array.append(value)
    return values_array

def process_file(file_path, n_ch = 8, n_acc = 3):
    with open(file_path, 'r') as file:
        result = []
        i = 0
        stops_n = 0
        stops = []
        while True:
            line = file.readline()
            if not line:
                break  # End of file
            split_line = line.strip().split(',')
            if len(split_line) == 1 and split_line[0].startswith('%'):
                stops_n += 1
                stops.append(i)
            elif (len(split_line) > 3) and (len(split_line) <= n_ch + n_acc + 1):
                values = processLine(split_line)
                if len(values) == (n_ch + n_acc):
                    to_add = values
                elif len(values) == (n_ch):
                    to_add = values + [0, 0, 0]
                else:
                    to_add = [0] * (n_ch + n_acc)
                result.append(to_add)
            i += 1
            if i % 1000000 == 0:
                print(f"Processing... {i}")
        return result, stops

def obci_create_edf_adc(bci_signals, sf, channels, user, gender, dts, birthday):
    header = highlevel.make_header(patientname=user, gender=gender, equipment=device, 
      startdate = dts, birthdate = datetime.datetime.strptime(birthday, '%Y_%m_%d-%H_%M_%S'))
    total_samples = math.floor(len(bci_signals) / sf)
    signals = []; signal_headers = []
    bci_signals = np.array(bci_signals)
    for channel in channels:
        ch_i =channels[channel]
        channel_data = bci_signals[:,ch_i]
        channel_data = channel_data[range(0,total_samples*sf)]
        signals.append(channel_data)
        ch_ph_min = -8388608; ch_ph_max = 8388607; ch_dig_min = -8388608; ch_dig_max = 8388607
        signal_headers.append({"label": channel, "dimension": "mV", "sample_rate": sf, "sample_frequency": sf, 'physical_max': ch_ph_max, 'physical_min': ch_ph_min, 'digital_max': ch_dig_max, 'digital_min': ch_dig_min, 'transducer': 'Gold-Cup Electrode', 'prefilter': ''})
        processed = len(channel_data)
    return([header, signal_headers, signals, processed])

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
    if sessions is not None:
        session = [session for session in sessions 
                   if len(session) > 2 and session[1] == file_name]
        settings = json.loads(session[0][2])
        sf = settings['sf']
        gain = settings['gain']
        channels = settings['channels']
        dts = datetime.datetime.strptime(session[0][0], '%Y-%m-%d %H:%M:%S')
    else:
        sf = 250
        gain = 24
        channels = {'F7-Fpz':0,'F8-Fpz':1}
        dts = datetime.datetime.now()
    
    user = 'User'
    gender = 'Male'; 
    birthday = '1980_01_01-12_00_00'
    brand = 'OpenBCI'
    device = 'Cyton'

    bci_signals, stops = process_file(file_path)
    bci_signals = np.array(bci_signals)
    header, signal_headers, signals, processed = obci_create_edf_adc(bci_signals, sf, channels, user, gender, dts, birthday)
    file_bdf = os.path.join(working_dir, 'data', 
       file_name + '_' + dts.strftime('%Y-%m-%d %H-%M-%S') + '.bdf')
    res = highlevel.write_edf(file_bdf, signals, signal_headers, header)
    if res:
        print(f'BDF saved to {file_bdf}')
    ADS1299_GAIN = gain
    signals_V = np.vectorize(adc_v_bci)(signals)
    
    import csv
    header = ['ts'] + [channel for channel in channels]
    
    import pandas as pd
    ts = dts + pd.to_timedelta(list(np.arange(len(signals[0]))), unit='ms')*1000/sf
    ts = ts.astype(np.int64)/1000000000
    
    signals_V = np.insert(signals_V, 0, ts, axis=0)
    signals_V[1]
    # Writing to a CSV file
    file_csv = os.path.join(working_dir, 'data', file_name + '_' + str(sf) + 'Hz_' + dts.strftime('%Y-%m-%d %H-%M-%S') + '.csv')
    with open(file_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in zip(*signals_V):
            writer.writerow(row)
        print(f'CSV saved to {file_csv}')

