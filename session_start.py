#!/home/lst/Storage/Dev/openbci-session/venv/bin/python3.11
import argparse
import datetime
import json
import os
import pyOpenBCI
import re
import serial.tools.list_ports
import sqlite3
import struct
import sys
import time
import yaml
from contextlib import closing


# CLI override for `activity` so you don't have to edit session_start.yml to
# switch between e.g. 16 (12H sleep) and 18 (5-min burn-in test). When --activity
# is omitted, fall back to the yml's `activity:` key (current behavior).
_ap = argparse.ArgumentParser(add_help=True)
_ap.add_argument('--activity', '-a', type=int, default=None,
                 help='Activity number from session_start.yml activities map '
                      '(overrides yml-configured activity)')
_args, _ = _ap.parse_known_args()


def find_openbci_port():
    """Find OpenBCI dongle (FTDI FT231X, VID:0x0403 PID:0x6015) serial port."""
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x0403 and p.pid == 0x6015:
            return p.device
    return None

# config, if relative path not working then use explicit path to working dir (repo dir with scripts and yml) or modify working directory in IDE/GUI settings
# working_dir = '/path/to/openbci-session'
working_dir = os.getcwd()
cfg_file = os.path.join(working_dir, "session_start.yml")

# rename sleep_analysis.yml.sample to sleep_analysis.yml and set directories
with open(cfg_file, "r") as yamlfile:
    cfg_base = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg = cfg_base['default']

# open sleep_analysis.yml and setup desired configuration: montage, electrode and activity
# activity consists of activity settings, chosen eeg and emg montages and electrode type

# activity structure
# type: sleep or anything else for custom processing
# dur: for sleep use 12H, for shorter sessions 5M 15M 30M 1H 2H 4H is available
# sf: sampling frequency, use 250 or 500 for sleep, for shorter sessions 1000 seems fine. in Hz
# gain: 24
# electrode: electrode type description from electrodes section, to be added into BDF description. for example 'Gold Cup OpenBCI, Ten20'.
# channels: eeg montage from montages section which contains electrode montage description in form of channel_name: board_pin, e.g. F7-Fpz:1 etc
# emg_channels: eog/ecg/emg montage from montages section
# dev: daisy or cyton (device)
# note: additional information

# full example of sleep EEG (F8, F7, O2, O1) with AFz as ref and FT7 as ground, with ECG (A-I lead) montage using daisy board, OpenBCI Gold cups with Ten-20 and 500Hz sampling frequency
# add to session_start.yml below default section
# activity: 1
# activities:
#   1:
#     type: sleep
#     dur: 12H
#     sf: 500
#     ground: Fp2
#     gain: 24
#     electrode: 1
#     ch: sleep_channels
#     dev: cyton
#     emg: emg_channels
#     note: my usual sleep session
# electrodes:
#   1: Gold Cup OpenBCI, Ten20
# montages:
#   sleep_channels:
#     F8-AFz: 0
#     F7-AFz: 1
#     O2-AFz: 2
#     O1-AFz: 3    
#   emg_channels:
#     ECG-AI: 4

# for simplicity board channels must be attached sequentially, 
# starting from the first and without gaps / skips
# so for 3 channels setup always use board channels 1, 2, 3
# otherwise you have to modify script for you specific needs

# format sd card on mac with terminal command:
# find disk number in dev with 'diskutil list'
# sudo diskutil zeroDisk /dev/disk4
# sudo diskutil eraseDisk FAT32 OBCI MBRFormat /dev/disk6
# sudo diskutil mountDisk /dev/disk4
# diskutil list:
#    /dev/disk4 (internal, physical):
#       #:                       TYPE NAME                    SIZE       IDENTIFIER
#       0:     FDisk_partition_scheme                        *31.9 GB    disk4
#       1:                 DOS_FAT_32 OBCI                    31.9 GB    disk4s1
# read 100mb of data after 50mb and confirm zeros (press q after command executed)
# sudo dd if=/dev/disk4 bs=1m skip=50 count=100 | hexdump -C | less
#     100+0 records in
#     100+0 records out
#     104857600 bytes transferred in 3.251993 secs (32244104 bytes/sec)
#     00000000  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
#     *
#     06400000

montages = cfg_base['montages']
electrodes = cfg_base['electrodes']
activities = cfg_base['activities']
activity_chosen = _args.activity if _args.activity is not None else cfg_base['activity']  # CLI override > yml default

# extract settings from chosen montage
activity = activities[activity_chosen]['type'];
device = activities[activity_chosen]['dev']
ch_n = 8 if device == 'cyton' else 16
channels = montages[activities[activity_chosen]['ch']];
emg_channels = montages[activities[activity_chosen]['emg']];
if emg_channels is None:
    emg_channels = {}
electrode_type = electrodes[activities[activity_chosen]['electrode']];
duration = activities[activity_chosen]['dur'];
ground = activities[activity_chosen]['ground'];
sampling_rate = activities[activity_chosen]['sf'];
gain = activities[activity_chosen]['gain'];
note = activities[activity_chosen]['note'];

print(f'{device}: {activity}, {electrode_type}, g{gain}, {sampling_rate}Hz, {duration}')
print(f'{channels}, ground: {ground}, emg: {emg_channels}')

# format sd card on mac with terminal command:
# sudo diskutil eraseDisk FAT32 OBCI MBRFormat /dev/disk6

t_sleep = 0.5
dbg = False
port = find_openbci_port() or cfg['port']
print(f'Port: {port}')
board = pyOpenBCI.OpenBCICyton(port=port, daisy=False)
time.sleep(1.5)  # one-time post-port-open settle for dongle RFduino startup banner
res = board.ser.read_all().decode()

def drain_serial(quiet=0.05, max_total=0.5):
    """Drain the serial buffer until it stays quiet for `quiet` seconds (default 50ms)
    or `max_total` elapses. Prevents leftover $$$ from a prior response satisfying
    the next wait_for_response() call early."""
    deadline_total = time.time() + max_total
    deadline_quiet = time.time() + quiet
    while time.time() < deadline_total:
        chunk = board.ser.read_all()
        if chunk:
            deadline_quiet = time.time() + quiet  # reset quiet-window
        elif time.time() >= deadline_quiet:
            return
        time.sleep(0.01)

def send_cmd(cmd):
    """Drain stale buffer, then send command (str). Use this for protocol commands
    where wait_for_response() will follow."""
    drain_serial()
    board.write_command(cmd)

def wait_for_response(end_marker='$$$', timeout=5.0, poll=0.02, settle=0.05):
    """Poll the serial buffer until end_marker is seen or timeout elapses.
    Returns whatever was accumulated, decoded. Replaces sleep+read_all with a
    bounded wait that returns as soon as the firmware finishes its response."""
    accumulated = b''
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = board.ser.read_all()
        if chunk:
            accumulated += chunk
            if end_marker.encode() in accumulated:
                time.sleep(settle)
                accumulated += board.ser.read_all()
                return accumulated.decode(errors='ignore')
        time.sleep(poll)
    return accumulated.decode(errors='ignore')

if dbg:
    board.write_command('?')
    time.sleep(t_sleep * 3)
    registers = board.ser.read_all().decode()
    registers2 = board.ser.read_all().decode()

def print_raw(sample):
    print(sample.channels_data)

# command list is here https://docs.openbci.com/Cyton/CytonSDK/
# check if board in default mode

send_cmd('//')
res = wait_for_response(timeout=2.0)
if dbg: print(repr(res))
if 'Success: default' in res:
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
send_cmd('~' + str(sampling_rates[sampling_rate]))
res = wait_for_response(timeout=2.0)
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
send_cmd(ch_cmd)
res = wait_for_response(timeout=5.0)
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

# Write SESSION.TXT via the firmware 'P' protocol BEFORE opening the SD slot.
# Contents = the exact byte stream the firmware would need to replay this
# session after a silent halt: channel config, sample rate, board mode,
# slot character, streaming start. NO META — the auto-resumed continuation
# file's %BOOT chains back to this session's first OBCI file via prev=, and
# the post-processor follows that to find the original %META.
#
# Order (must match firmware pre-scan in replaySessionFile):
#   1. xNGSIBPnX channel-config commands (no ordering constraint among them
#      themselves, but must precede 'b')
#   2. ~N sample-rate (must precede slot, since setupSDcard's BLOCK_COUNT
#      computation reads getSampleRate())
#   3. /N board mode
#   4. slot char (K/A/etc.) — opens new OBCI_*.TXT
#   5. b — streamStart
#
# P is gated on !board.streaming in the firmware, so this MUST happen before
# we send 'b'. We send it now (before the slot too) — the SD is still
# uncontended (no recording file open), so the SESSION.TXT write is safe.
#
# 2026-05-11 — on by default after firmware fix landed (the dispatch-order
# bug where 'P' bytes embedded in an M META payload were being hijacked by
# sdPersistProcess was resolved by gating P state-0 entry on sdMetaState==0).
# Set WR_SESSION_PERSIST=0 to disable the SESSION.TXT write if a regression
# ever shows up — the underlying session will still record without it.
if os.environ.get('WR_SESSION_PERSIST', '1') != '0':
    session_txt_lines = [ch_cmd,
                         '~' + str(sampling_rates[sampling_rate]),
                         '/0',
                         durations[duration],
                         'b']
    session_txt_payload = ('\n'.join(session_txt_lines) + '\n').encode('ascii')
    if len(session_txt_payload) > 1024:
        print(f'WARNING: SESSION.TXT payload {len(session_txt_payload)}B exceeds 1024-byte firmware cap; skipping auto-resume setup')
    else:
        expected_sum = sum(session_txt_payload) & 0xFFFF
        persist_verified = False
        p_ack = ''
        for attempt in range(2):
            drain_serial()
            board.ser.write(b'P' + struct.pack('<H', len(session_txt_payload)) + session_txt_payload)
            p_ack = wait_for_response(timeout=5.0)
            m = re.search(r'PERSIST OK (\d+) (\d+)', p_ack)
            if m and int(m.group(1)) == len(session_txt_payload) and int(m.group(2)) == expected_sum:
                print(f'SESSION.TXT written: {len(session_txt_payload)} bytes, sum {expected_sum}'
                      + (f' (retry {attempt})' if attempt else ''))
                persist_verified = True
                break
            if attempt == 0:
                time.sleep(1.1)  # firmware P state-machine timeout (1s) + margin
        if not persist_verified:
            # Don't sys.exit — auto-resume is a safety net, not a hard requirement.
            # Continue without it; the session will still record fine, just without
            # silent-halt recovery on next boot.
            print(f'WARNING: SESSION.TXT not confirmed after retry. response: {p_ack!r}. '
                  f'Continuing without auto-resume capability.')
else:
    print('SESSION.TXT auto-resume setup disabled (WR_SESSION_PERSIST != 1). '
          'Recording will proceed normally; silent-halt recovery is not armed for this session.')

send_cmd(durations[duration])
# SD pre-erase + allocation: ms for 5M, ~6-25 s for 12H/24H (depends on SD card speed).
# Generous timeout — wait_for_response returns as soon as the $$$ EOT arrives.
res = wait_for_response(timeout=60.0)
print(res)

match = re.search(r' \d+ ', res)
if match is not None:
    matched = match.group(0)
    if(len(matched) > 0):
        blocks = int(re.sub(" ","", matched))
        BLOCK_5MIN = 16890
        # firmware auto-picks BLOCK_DIV=2 for Cyton-only (daisy absent), =1 for daisy
        block_div = 1 if device == 'daisy' else 2
        sd_duration = (blocks * 250 / sampling_rate) * block_div / (BLOCK_5MIN / 5)
        print(f'SD blocks: {blocks} and max duration: {round(sd_duration)} minutes', )
        if duration in ['1H','2H','4H','12H','24H']:
            if f'{round(sd_duration / 60)}H' != duration:
                sys.exit(f'board init wrong duration for sd file: {duration} requested {round(sd_duration/60)}H returned')
        elif f'{round(sd_duration)}M' != duration:
            sys.exit(f'board init wrong duration for sd file: {duration} requested {round(sd_duration)}M returned')

if dbg: print(res)
# Detect successful SD slot open via the "Size N SD file OBCI_NN.TXT" line —
# emitted by setupSDcard on success regardless of whether the card was already
# initialised (the "Wiring and sdcard is correct." pre-roll only prints on
# the very first card.init, which we may consume earlier in the P command).
if len(re.findall('Size ', res)) > 0:
    re_file = re.findall(r'I\_.*\.T', res)
    if len(re_file) > 0:
        re_file = 'OBC' + re_file[0] + 'XT'
        print(f'SD file init success {re_file}')
        sd_file = re_file
        # board.start_stream(print_raw)
        send_cmd('~~')
        res = wait_for_response(timeout=0.5)
        if dbg: print(res)
        # Build settings + send META line via firmware 'M' protocol BEFORE 'b'
        # so the line lands at the start of the SD TXT file (self-describing).
        dts = datetime.datetime.now()
        settings = {
            'gain':gain, 'channels':channels, 'sf': sampling_rate,
            'ground': ground, 'electrode': electrode_type, 'emg_ch': emg_channels,
            'ch_n': ch_n, 'activity': activity, 'device': device, 'note': note}
        json_settings = json.dumps(settings)
        meta_obj = {'dts': dts.strftime(cfg["sql_dt_format"]), 'file': sd_file}
        meta_obj.update(settings)
        meta_payload = ('%META ' + json.dumps(meta_obj) + '\n').encode('utf-8')
        # defense-in-depth: payload must be single-line ASCII (json.dumps with default
        # ensure_ascii=True guarantees this; the asserts catch a future regression).
        assert b'\n' not in meta_payload[:-1], 'meta payload has embedded newline'
        assert b'\x00' not in meta_payload, 'meta payload has NUL byte'
        if len(meta_payload) <= 1024:
            expected_sum = sum(meta_payload) & 0xFFFF
            verified = False
            ack = ''
            for attempt in range(2):
                drain_serial()  # clear stale buffer (incl. any prior META ERR)
                board.ser.write(b'M' + struct.pack('<H', len(meta_payload)) + meta_payload)
                ack = wait_for_response(timeout=2.0)
                m = re.search(r'META OK (\d+) (\d+)', ack)
                if m and int(m.group(1)) == len(meta_payload) and int(m.group(2)) == expected_sum:
                    print(f'meta verified: {len(meta_payload)} bytes, sum {expected_sum}'
                          + (f' (retry {attempt})' if attempt else ''))
                    verified = True
                    break
                if attempt == 0:
                    time.sleep(1.1)  # firmware META timeout (1s) + margin → drop any stuck state
            if not verified:
                m = re.search(r'META OK (\d+) (\d+)', ack)
                if m:
                    print(f'WARNING: meta mismatch after retry — expected len={len(meta_payload)} sum={expected_sum}, firmware reported len={m.group(1)} sum={m.group(2)}. Continuing without verified meta.')
                else:
                    print(f'WARNING: meta not confirmed after retry. response: {ack!r}. Continuing without verified meta.')
        else:
            print(f'meta payload {len(meta_payload)}B exceeds 1024 cap, skipped')
        send_cmd('b')
        print(f'Session started at {dts}')
        time.sleep(0.2)  # brief settle so stream actually begins before we close serial
        with closing(sqlite3.connect(os.path.join(cfg['session_dir'],cfg['session_file']), timeout=10)) as con:
            with con:
                with closing(con.cursor()) as cur:
                    sql = 'CREATE TABLE IF NOT EXISTS Sessions(dts datetime NOT NULL PRIMARY KEY, file VARCHAR(256), settings TEXT NOT NULL)'
                    cur.execute(sql)
                    sql = 'REPLACE INTO Sessions (dts,file,settings) VALUES (\'' + dts.strftime(cfg["sql_dt_format"]) + '\', \'' + sd_file + '\', \'' + json_settings + '\')'
                    cur.execute(sql)
    else:
        print(res)
        sys.exit(f'SD file not found. Please restart board, dongle & check sd card')
else:
    print(res)
    sys.exit(f'SD init failed. Please restart board, dongle & check sd card')
