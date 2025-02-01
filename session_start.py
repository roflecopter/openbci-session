#!/Users/bob/anaconda3/envs/env-311/bin/python3.11
import datetime 
import json
import os
import pty
import pyOpenBCI
import re
import serial
import sqlite3
import sys
import time
import yaml
from contextlib import closing

# config, if relative path not working then use explicit path to working dir (repo dir with scripts and yml) or modify working directory in IDE/GUI settings
# working_dir = '/path/to/openbci-session'
working_dir = os.getcwd()
cfg_file = os.path.join(working_dir, "session_start.yml")

# rename sleep_analysis.yml.sample to sleep_analysis.yml and set directories
with open(cfg_file, "r") as yamlfile:
    cfg_base = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg = cfg_base['default']

# for simplicity channels must be attached sequentially, 
# starting from the first and without gaps / skips
# so for 3 channels setup always use board channels 1, 2, 3
# otherwise you have to modify script for you specific needs

# here you can see different montages I've tried. You can reproduce and use one of these
# or add your own; you can delete uneeded montages
# but make sure activity_choice contains at least single montage

sleep_channels = {
    'F8-T5':0, 'F7-T5':1, 'O2-T5':2, 'O1-T5':3, 
    'T8-T5':4, 'T7-T5':5, 'AFz-T5':6, 'T6-T5':7}
sleep_om_channels = {
    'F8-T5':0, 'F7-T5':1, 'O2-T5':2, 'O1-T5':3, 
    'T8-T5':4, 'T7-T5':5, 'AFz-T5':6, 'T6-T5':7}
sleep_ext_channels = {
    'F8-T5':0, 'F7-T5':1, 'O2-T5':2, 'O1-T5':3, 
    'T8-T5':4, 'T7-T5':5, 'AFz-T5':6, 'T6-T5':7,
    'Cz': 8}
sleep_all_channels = {
    'F8-T7':0, 'F7-T7':1, 'O2-T7':2, 'O1-T7':3, 
    'T8-T7':4, 'AFz-T7':5}
# emg_channels = {'EMG-N':8, 'EOG-RL':9}

sleep_full_channels = {
    'F8-T7':0, 'F7-T7':1, 'O2-T7':2, 'O1-T7':3, 
    'T8-T7':4, 'AFz-T7':5}
sleep_af_channels = {
    'AF4-T7':0, 'AF3-T7':1, 'O2-T7':2, 'O1-T7':3, 
    'T8-T7':4, 'AFz-T7':5}
sleep_full_short_channels = {
    'F8-T7':0, 'T8-T7':1, 'O2-T7':2, 'O1-T7':3, 
    'F7-T7':4, 'AFz-T7':5}
# emg_channels = {'EOG-RL':6, 'EMG-N':7}
# emg_channels = {'EOG-RL':6}
emg_channels = {}

cyton_cap_channels = {
    'F4-M1':0, 'F3-M1':1, 'O2-M1':2, 'O1-M1':3, 
    'T8-M1':4, 'T7-M1':5, 'Cz-M1':6, 'M2-M1':7}
daisy_cap_channels = {
    'O2-M1':0, 'O1-M1':1, 'P4-M1':2, 'P3-M1':3, 
    'C4-M1':4, 'C3-M1':5, 'Cz-M1':6, 'Fpz-M1':7,
    'F4-M1':8, 'F3-M1':9, 'F8-M1':10, 'F7-M1':11,
    'T8-M1':12, 'T7-M1':13, 'Fz-M1':14, 'M2-M1':15}
daisy_ultracortex_channels = {
    'O2-M1':0, 'O1-M1':1, 'P4-M1':2, 'P3-M1':3, 
    'C4-M1':4, 'C3-M1':5, 'Cz-M1':6, 'M2-M1':7,
    'F4-M1':8, 'F3-M1':9, 'F8-M1':10, 'F7-M1':11,
    'T8-M1':12, 'T7-M1':13, 'Fz-M1':14, 'AFz-M1':15}

# emg_channels = {'E2-Fpz':5, 'E1-Fpz':6}
# emg_channels = {'EOG-RL':5, 'ECG-RA-V2':6}
# emg_channels = {'ECG-AI':6}
# emg_channels = {}

sleep_fo_channels = {
    'F8-AFz':0, 'F7-AFz':1, 'O2-AFz':2, 'O1-AFz':3, 
    }
emg_channels = {'ECG-AI':4}

electrode_choice = {'1':'Gold Cup OpenBCI, Ten20', 
                    '2': 'Premium Ag/AgCl FRI, Sigma Gel',
                    '3':'Premium Ag/AgCl FRI', 
                    '4':'Ag/AgCl FRI disposable, Sigma Gel', 
                    '5':'Gold Cup Grass, Ten20',
                    '6':'Gold Cup OpenBCI, Sigma Gel',
                    '7': 'Ambu Neuroline Cup, Sigma Gel'}

# final list of full montage settings
# type: sleep or anything else for custom processing
# dur: for sleep use 12H, for shorter sessions 5M 15M 30M 1H 2H 4H is available
# sf: sampling frequency, use 250 or 500 for sleep, for shorter sessions 1000 seems fine. in Hz
# gain: 24
# e: electrode type description, added to BDF desc. for example 'Gold Cup OpenBCI, Ten20'. Use electrode_choice array and modify if needed
# ch: array electrode montage description, dict with 'channel name':board_slot_number, e.g. {'F7-Fpz':1, 'F8-Fpz': 2}
# dev: daisy or cyton (device)
# notice: emg_channels are set separately and not contained here. set them separately if you want EOG / ECG. Format is same is ch

activity_choice = {
    '0': {'type': 'sleep', 'dur': '12H', 'sf': 500, 'g': 'FT7', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_om_channels, 'dev': 'daisy'},
    '1': {'type': 'sleep', 'dur': '12H', 'sf': 500, 'g': 'Fp2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_fo_channels, 'dev': 'cyton'},
    '2': {'type': 'nsdr', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['4'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '3': {'type': 'swaroopa isha', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '4': {'type': 'meditation', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['6'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '5': {'type': 'rest', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '6': {'type': 'rest-eyeopen', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '7': {'type': 'dantian breath', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '8': {'type': 'meditation', 'dur': '1H', 'sf': 1000, 'g': 'Fp2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_channels, 'dev': 'cyton'},
    '9': {'type': 'unknown', 'dur': '1H', 'sf': 500, 'g': 'Fp2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_ext_channels, 'dev': 'daisy'},
    '10': {'type': 'dantian breath', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '11': {'type': 'dantian breath', 'dur': '1H', 'sf': 1000, 'g': 'M2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_ultracortex_channels, 'dev': 'daisy'},
    '12': {'type': 'sleep', 'dur': '12H', 'sf': 500, 'g': 'Fp2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_af_channels, 'dev': 'cyton'},
    }

activity_chosen = '1' # choose the montage you want to apply, add note if needed
# note = '2m rest before, seated'
note = ''

# full example of sleep EEG (F8, F7, O2, O1) with AFz as ref and FT7 as ground, with ECG (A-I lead) montage using daisy board, OpenBCI Gold cups with Ten-20 and 500Hz sampling frequency
# activity_choice = {
#    '0': {'type': 'sleep', 'dur': '12H', 'sf': 500, 'g': 'FT7', 'gain': 24,
#          'e': 'Gold Cup OpenBCI, Ten20', 'ch': {'F8-AFz':0, 'F7-AFz':1, 'O2-AFz':2, 'O1-AFz':3, }, 'dev': 'daisy'},}
# emg_channels = {'ECG-AI':4}
# note = 'my note'
# activity_chosen = '0'
# here you have set montage, thats all, dont need to modify anything below


# extract settings from chosen montage
activity = activity_choice[activity_chosen]['type'];
device = activity_choice[activity_chosen]['dev']
ch_n = 8 if device == 'cyton' else 16
channels = activity_choice[activity_chosen]['ch'];
electrode_type = activity_choice[activity_chosen]['e'];
duration = activity_choice[activity_chosen]['dur'];
ground = activity_choice[activity_chosen]['g'];
sampling_rate = activity_choice[activity_chosen]['sf'];
gain = activity_choice[activity_chosen]['gain'];

print(f'{device}: {activity}, {electrode_type}, g{gain}, {sampling_rate}Hz, {duration}')
print(f'{channels}, ground: {ground}, emg: {emg_channels}')

# format sd card on mac with terminal command:
# sudo diskutil eraseDisk FAT32 OBCI MBRFormat /dev/disk6

t_sleep = 2
dbg = False
board = pyOpenBCI.OpenBCICyton(port=cfg['port'], daisy=False)
time.sleep(t_sleep)
res = board.ser.read_all().decode()

if dbg:
    board.write_command('?')
    time.sleep(t_sleep * 3)
    registers = board.ser.read_all().decode()
    registers2 = board.ser.read_all().decode()

def print_raw(sample):
    print(sample.channels_data)

# command list is here https://docs.openbci.com/Cyton/CytonSDK/
# check if board in default mode

board.write_command('//')
time.sleep(t_sleep)
res = board.ser.read_all().decode()
if dbg: print(res)
if res == 'Success: default$$$':
    print(f'mode is default')
else:
    sys.exit(f'mode is not default')

# BLOCK_DIV in firmware code seems to reduce real size by 2 times due to wrong block size
# this results in half session time, so do not detach daisy until it fixed in firmware
# if you want 8 channels with daisy - you have to detach it physically
# # remove daisy if it unused
# if (device == 'daisy') and (len(channels) > 8):
#     board.write_command('C')
#     time.sleep(t_sleep)
#     res = board.ser.read_all().decode()
#     print(f'attached daisy')
# elif (device == 'daisy') and (len(channels) < 9):
#     board.write_command('c')
#     time.sleep(t_sleep)
#     res = board.ser.read_all().decode()
#     if dbg: print(res)
#     if res == 'daisy removed$$$':
#         print(f'daisy is removed')
#         ch_n = 8
#         device = 'daisy-off'
#         print(f'Device changed to {device}: n_channels to {ch_n}')
#     else:
#         sys.exit(f'error, daisy should be removed, but it wasnt')
# elif (len(channels) > 8) and (device != 'daisy'):
#     sys.exit("device is not set to 'daisy' but n_channels > 8")

# set sampling rate
sampling_rates = {16000:0,8000:1,4000:2,2000:3,1000:4,500:5,250:6}
board.write_command('~' + str(sampling_rates[sampling_rate]))
time.sleep(t_sleep)
res = board.ser.read_all().decode()
if dbg: print(res)
if len(re.findall('Sample rate is ' + str(sampling_rate) + 'Hz', res)) > 0:
    print(f'sampling rate set to {sampling_rate}')
else:
    sys.exit(f'sampling rate not set')

# channel configuration
gains = {1:0,2:1,4:2,6:3,8:4,12:5,24:6}
on = '0' + str(gains[gain]) + '0110X'; off = '160000X'
emg = '0' + str(gains[gain]) + '0000X'
chs = [on] * len(channels) + [emg] * len(emg_channels) + [off] * (ch_n - len(channels)-len(emg_channels))
ch_cmd = ('x1' + chs[0] + 'x2' + chs[1] + 
    'x3' + chs[2] + 'x4' + chs[3] + 
    'x5' + chs[4]+ 'x6' + chs[5] + 
    'x7' + chs[6] + 'x8' + chs[7])
if len(chs) == 16:
    ch_cmd = ch_cmd  + ('xQ' + chs[8] + 'xW' + chs[9] + 
        'xE' + chs[10] + 'xR' + chs[11] + 
        'xT' + chs[12]+ 'xY' + chs[13] + 
        'xU' + chs[14] + 'xI' + chs[15])
    
print(ch_cmd)
board.write_command(ch_cmd)
time.sleep(t_sleep*5)
res = board.ser.read_all().decode()
if dbg: print(res)
if (len(re.findall('Success', res)) > 0):
    channels.update(emg_channels)
    print(f'channels set: {channels}')
else:
    sys.exit(f'channels not set')

if dbg: 
    board.write_command('?')
    time.sleep(t_sleep*3)
    new_registers = board.ser.read_all().decode()
    new_registers2 = board.ser.read_all().decode()
    # print(new_registers + registers2)

# enable sdcard writing
durations = {'5M':'A','15M':'S','30M':'F','1H':'G','2H':'H','4H':'J','12H':'K','24H':'L'}
board.write_command(durations[duration])
time.sleep(t_sleep * 10)

res = board.ser.read_all().decode()
print(res)
time.sleep(t_sleep * 5)
res2 = board.ser.read_all().decode()
print(res2)
res = res + res2

match = re.search(r' \d+ ', res)
if match is not None:
    matched = match.group(0)
    if(len(matched) > 0):
        blocks = int(re.sub(" ","", matched))
        BLOCK_5MIN = 16890
        sd_duration = (blocks * 250 / sampling_rate) / (BLOCK_5MIN / 5)
        print(f'SD blocks: {blocks} and max duration: {round(sd_duration)} minutes', )
        if duration in ['1H','2H','4H','12H','24H']:
            if f'{round(sd_duration / 60)}H' != duration:
                sys.exit(f'board init wrong duration for sd file: {duration} requested {round(sd_duration/60)}H returned')
        elif f'{round(sd_duration)}M' != duration:
            sys.exit(f'board init wrong duration for sd file: {duration} requested {round(sd_duration)}M returned')

if dbg: print(res)
if len(re.findall('correct', res)) > 0:
    re_file = re.findall(r'I\_.*\.T', res)
    if len(re_file) > 0:
        re_file = 'OBC' + re_file[0] + 'XT'
        print(f'SD file init success {re_file}')
        sd_file = re_file
        # board.start_stream(print_raw)
        board.write_command('~~')
        time.sleep(t_sleep)
        res = board.ser.read_all().decode()
        print(res)
        board.write_command('b')
        dts = datetime.datetime.now()
        print(f'Session started at {dts}')
        time.sleep(t_sleep)
        with closing(sqlite3.connect(os.path.join(cfg['session_dir'],cfg['session_file']), timeout=10)) as con:
            with con:
                with closing(con.cursor()) as cur:
                    sql = 'CREATE TABLE IF NOT EXISTS Sessions(dts datetime NOT NULL PRIMARY KEY, file VARCHAR(256), settings TEXT NOT NULL)'
                    cur.execute(sql)
                    settings = {
                        'gain':gain, 'channels':channels, 'sf': sampling_rate, 
                        'ground': ground, 'electrode': electrode_type, 'emg_ch': emg_channels,
                        'ch_n': ch_n, 'activity': activity, 'device': device, 'note': note}
                    json_settings = json.dumps(settings)
                    sql = 'REPLACE INTO Sessions (dts,file,settings) VALUES (\'' + dts.strftime(cfg["sql_dt_format"]) + '\', \'' + sd_file + '\', \'' + json_settings + '\')'
                    cur.execute(sql)            
    else:
        print(res)
        sys.exit(f'SD file not found. Please restart board, dongle & check sd card')
else:
    print(res)
    sys.exit(f'SD init failed. Please restart board, dongle & check sd card')
