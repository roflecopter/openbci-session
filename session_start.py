import os, time, datetime, sys, re, pty, serial
import sqlite3, json
from contextlib import closing
import pyOpenBCI

# config
working_dir = '/path/to/openbci-session' 

sleep_channels = {
    'F8-T5':0, 'F7-T5':1, 'O2-T5':2, 'O1-T5':3, 
    'T8-T5':4, 'T7-T5':5, 'Fpz-T5':6, 'T6-T5':7}
cyton_cap_channels = {
    'F4-M1':0, 'F3-M1':1, 'O2-M1':2, 'O1-M1':3, 
    'T8-M1':4, 'T7-M1':5, 'Cz-M1':6, 'M2-M1':7}
daisy_cap_channels = {
    'F8-M1':0, 'F7-M1':1, 'F4-M1':2, 'F3-M1':3, 
    'Fpz-M1':4, 'Fz-M1':5, 'T8-M1':6, 'T7-M1':7,
    'C4-M1':8, 'C3-M1':9, 'Cz-M1':10, 'M2-M1':11,
    'P4-M1':12, 'P3-M1':13, 'O2-M1':14, 'O1-M1':15}
electrode_choice = {'1':'Gold Cup OpenBCI, Ten20', 
                    '2': 'Premium Ag/AgCl FRI, Sigma Gel',
                    '3':'Premium Ag/AgCl FRI', 
                    '4':'Ag/AgCl FRI disposable, Sigma Gel', 
                    '5':'Gold Cup Grass, Ten20'}
activity_choice = {
    '1': {'type': 'sleep', 'dur': '12H', 'sf': 500, 'g': 'Fp2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_channels, 'dev': 'cyton'},
    '2': {'type': 'nsdr', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '3': {'type': 'swaroopa isha', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '4': {'type': 'meditation', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '5': {'type': 'rest', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '6': {'type': 'rest-eyeopen', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '7': {'type': 'dantian breath', 'dur': '1H', 'sf': 1000, 'g': 'AFz', 'gain': 24,
          'e': electrode_choice['1'], 'ch': daisy_cap_channels, 'dev': 'daisy'},
    '8': {'type': 'meditation', 'dur': '1H', 'sf': 1000, 'g': 'Fp2', 'gain': 24,
          'e': electrode_choice['1'], 'ch': sleep_channels, 'dev': 'cyton'},
    }

activity_chosen = '8'; 
activity = activity_choice[activity_chosen]['type'];
device = activity_choice[activity_chosen]['dev']
ch_n = 8 if device == 'cyton' else 16
channels = activity_choice[activity_chosen]['ch'];
electrode_type = activity_choice[activity_chosen]['e'];
duration = activity_choice[activity_chosen]['dur'];
ground = activity_choice[activity_chosen]['g'];
sampling_rate = activity_choice[activity_chosen]['sf'];
gain = activity_choice[activity_chosen]['gain'];

# emg_channels = {'E2-Fpz':5, 'E1-Fpz':6}
# emg_channels = {'EOG-RL':5, 'ECG-RA-V2':6}
# emg_channels = {'EOG-RL':5}
# emg_channels = {'ECG-AS':5}
emg_channels = {}

print(f'{activity}, {electrode_type}, g{gain}, {sampling_rate}Hz, {duration}')
print(f'{channels}, ground: {ground}, emg: {emg_channels}')

# format sd card on mac with terminal command:
# sudo diskutil eraseDisk FAT32 OBCI MBRFormat /dev/disk6

t_sleep = 1
dbg = False
# board = pyOpenBCI.OpenBCICyton(port='/dev/cu.usbserial-D200PMQM', daisy=False)
board = pyOpenBCI.OpenBCICyton(port='/dev/cu.usbserial-DP04WFVJ', daisy=False)
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
chs = [on] * len(channels) + [emg] * len(emg_channels) + [off] * (ch_n -len(channels)-len(emg_channels))
ch_cmd = ('x1' + chs[0] + 'x2' + chs[1] + 
    'x3' + chs[2] + 'x4' + chs[3] + 
    'x5' + chs[4]+ 'x6' + chs[5] + 
    'x7' + chs[6] + 'x8' + chs[7])
if len(chs) == 16:
    ch_cmd = ch_cmd  + ('xQ' + chs[8] + 'xW' + chs[9] + 
        'xE' + chs[2] + 'xR' + chs[3] + 
        'xT' + chs[4]+ 'xY' + chs[5] + 
        'xU' + chs[6] + 'xI' + chs[7])
    
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
        sd_duration = (blocks * 250 / sampling_rate) / (BLOCK_5MIN * 12)
        print(f'SD blocks: {blocks} and max duration: {round(sd_duration,1)}h', )

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
        with closing(sqlite3.connect(os.path.join(working_dir, 'data','sessions.db'), timeout=10)) as con:
            with con:
                with closing(con.cursor()) as cur:
                    sql = 'CREATE TABLE IF NOT EXISTS Sessions(dts datetime NOT NULL PRIMARY KEY, file VARCHAR(256), settings TEXT NOT NULL)'
                    cur.execute(sql)
                    settings = {
                        'gain':gain, 'channels':channels, 'sf': sampling_rate, 
                        'ground': ground, 'electrode': electrode_type, 'emg_ch': emg_channels,
                        'ch_n': ch_n, 'activity': activity, 'device': device}
                    json_settings = json.dumps(settings)
                    sql = 'REPLACE INTO Sessions (dts,file,settings) VALUES (\'' + dts.strftime('%Y-%m-%d %H:%M:%S') + '\', \'' + sd_file + '\', \'' + json_settings + '\')'
                    cur.execute(sql)            
                    # sql = 'SELECT * FROM Sessions'
                    # cur.execute(sql)
                    # cur.fetchall()
    else:
        print(res)
        sys.exit(f'SD file not found. Please restart board, dongle & check sd card')
else:
    print(res)
    sys.exit(f'SD init failed. Please restart board, dongle & check sd card')

# old setups
# channels = {'F8-T3':0} # for sd processing at the session end, Label:%N [0-7] 
# channels = {'F8-T3':0,'F7-T3':1,'O2-T3':2, 'O1-T3':3, 'T4-T3':4} # for sd processing at the session end, Label:%N [0-7] 
# channels = {'F8-Pz':0,'F7-Pz':1,'O2-Pz':2, 'O1-Pz':3, 'T8-Pz':4, 'T7-Pz':5, 'Fz-Pz':6, 'Cz-Pz':7} # for sd processing at the session end, Label:%N [0-7] 
# channels = {'F8-Pz':0,'F7-Pz':1,'O2-Pz':2, 'O1-Pz':3, 'T8-Pz':4, 'T7-Pz':5, 'Fz-Pz':6} # for sd processing at the session end, Label:%N [0-7] 
# channels = {'F8-Oz':0,'F7-Oz':1,'O2-Oz':2, 'O1-Oz':3, 'T8-Oz':4, 'T7-Oz':5, 'Fz-Oz':6, 'Cz-Oz': 7} # for sd processing at the session end, Label:%N [0-7] 
# channels = {'F8-Oz':0, 'F7-Oz':1, 'O2-Oz':2, 'O1-Oz':3, 
#             'T8-Oz':4, 'T7-Oz':5, 'Fz-Oz':6} # for sd processing at the session end, Label:%N [0-7] 
# channels = {'F8-M1':0,'F7-M1':1,'O2-M1':2, 'O1-M1':3, 'T8-M1':4, 'T7-M1':5, 'Fz-M1':6, 'Cz-M1':7 } # for sd processing at the session end, Label:%N [0-7] 
