import os, time, datetime, sys, re, pty, serial
import sqlite3
from contextlib import closing
import pyOpenBCI

working_dir = '/Volumes/Data/Storage/Dev/openbci-psg' 
t_sleep = 1
dbg = False
board = pyOpenBCI.OpenBCICyton(port='/dev/cu.usbserial-D200PMQM', daisy=False)
time.sleep(t_sleep)
res = board.ser.read_all().decode()
res
if dbg:
    board.write_command('?')
    time.sleep(t_sleep * 3)
    registers = board.ser.read_all().decode()
    registers2 = board.ser.read_all().decode()
    # print(registers + registers2)

def print_raw(sample):
    print(sample.channels_data)

# command list is here https://docs.openbci.com/Cyton/CytonSDK/
# check if board in default mode

board.write_command('//')
time.sleep(t_sleep)
res = board.ser.read_all().decode()
if dbg: print(res)
if res == 'Success: default$$$':
    print(f'mode is ok')
else:
    sys.exit(f'mode is not default')

# set sampling rate
board.write_command('~5')
time.sleep(t_sleep)
res = board.ser.read_all().decode()
if dbg: print(res)
if len(re.findall('Sample rate is 500Hz', res)) > 0:
    print(f'sampling rate set')
else:
    sys.exit(f'sampling rate not set')

# channel configuration
eeg_ch = '060110X'; no_ch = '160000X'
board.write_command('x1' + eeg_ch + 'x2' + eeg_ch + 
                    'x3' + eeg_ch + 'x4' + no_ch + 
                    'x5' + no_ch + 'x6' + no_ch + 
                    'x7' + no_ch + 'x8' + no_ch)
time.sleep(t_sleep*5)
res = board.ser.read_all().decode()
if dbg: print(res)
if (len(re.findall('Success', res)) > 0):
    print(f'channels set')
else:
    sys.exit(f'channels not set')

if dbg: 
    board.write_command('?')
    time.sleep(t_sleep*3)
    new_registers = board.ser.read_all().decode()
    new_registers2 = board.ser.read_all().decode()
    # print(new_registers + registers2)

# enable sdcard writing with 12H
board.write_command('A')
time.sleep(t_sleep * 5)

res = board.ser.read_all().decode()
if dbg: print(res)
if len(re.findall('correct', res)) > 0:
    re_file = re.findall(r'OBCI\_.*\.TXT', res)
    if len(re_file) > 0:
        print(f'SD file init success {re_file}')
        sd_file = re_file[0]
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
                    sql = 'CREATE TABLE IF NOT EXISTS Sessions(dts datetime NOT NULL PRIMARY KEY, file VARCHAR(256))'
                    cur.execute(sql)
                    
                    sql = 'REPLACE INTO Sessions (dts,file) VALUES (\'' + dts.strftime('%Y-%m-%d %H:%M:%S') + '\', \'' + sd_file + '\')'
                    cur.execute(sql)
            
                    # sql = 'SELECT * FROM Sessions'
                    # cur.execute(sql)
                    # rows = cur.fetchall()
                    # print(rows)

else:
    sys.exit(f'SD init failed')
    
# dte = datetime.datetime.now(); print(dte)
