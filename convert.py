import numpy as np
import datetime
import math
import os
from pyedflib import highlevel

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

working_dir = '/Volumes/Data/Storage/Dev/openbci-sd'
# working_dir = '/path/to/openbci-sd'
os.chdir(working_dir)

file_name = os.path.join('data','OBCI-01.TXT')
sf = 250
user = 'User'
gender = 'Male'; 
birthday = '1980_01_01-12_00_00'
brand = 'OpenBCI'
device = 'Cyton'
dts = datetime.datetime.now()
bci_signals, stops = process_file(file_name)
bci_signals = np.array(bci_signals)
channels = {'F7-Fpz':2,'F8-Fpz':3}

header, signal_headers, signals, processed = obci_create_edf_adc(bci_signals, sf, channels, user, gender, dts, birthday)
res = highlevel.write_edf(file_name + '.bdf', signals, signal_headers, header)
