I run these scripts at my **macbook** as part of my home personal sleep research to start session and collect data from OpenBCI. 
OpenBCI allows for gold-standard PSG (EEG, ECG etc) data collection. 
Device is good enought for daily use, setting up montage (diy headband) with a session start takes 4-5 minutes. Read more [here](https://blog.kto.to/hypnodyne-zmax-vs-openbci-eeg-psg)

# session_start.py
Script to start OpenBCI Cyton session in a single click, usually for sleep EEG acquisiton purposes. 
* Used to start session with data saved on sd with desired sampling frequency (for greater than 250Hz modded [firmware](https://github.com/roflecopter/OpenBCI_Cyton_Library_SD) need to be flashed, otherwise it will always write with default 250Hz).
* Saves session start timestamp and settings into sqlite file (data/sessions.db). Session info will be used in sd_convert.py script

# sd_convert.py
Script to convert OpenBCI SD card .TXT files to 
* 24-bit BDF with calibrated values. Accelerometer data is upsampled to match ADS sampling rate.
* Recording timestamp and settings taken from sqlite db which automatically created and updated by session_start.py script

# session_analyse.py
Script to analyses recorded sessions. Suited for short sessions (like meditations etc).
Periods (and other settings) are defined for each session in sessions variable inside script.
* Reads raw BDF file from sd_convert and filter it, split into epochs and rejects bad ones (autoreject)
* Plot Multitaper Spectrogram for each channel, highlights bad epochs (autoreject) and periods
* Plot Amplitude topomaps based on good epochs passed into yasa.bandpower for each period defined in sessions
* Plot PSD / Frequency plot for each channel and period computed with mne.compute_psd(method='welch')
* Plot Band Power (delta, theta, alpha, beta, gamma) vs time and highlings band epochs (autoreject) and periods

# sleep_analyse.py
Script to analyses recored sleep session.
* Reads raw BDF file from sd_convert and filter it
* Builds hypnograms with YASA and make plots (for each channel, max probablity and adjusted consensus)
* Plot Multitaper Spectrogram
* PLto Amplitude topomaps grouped by sleep stage
* Plot PSD / Frequency plot (easy to see bad channels)

# Quickstart
* install / setup Python 3.11 environment or use global
* pull repo and cd into it
* run pip3 install -r requirements.txt
* optional: flash modded [firmware](https://github.com/roflecopter/OpenBCI_Cyton_Library_SD) if you want 500Hz sampling rate and flashing LED during sd card write session
* put sd card into macbook and set desired name, for example BCI. Full path should be /Volumes/BCI (default in script) and need to be set into sd_convert.py to sd_dir variable
* inside board and insert Cyton USB dongle with switch at position GPIO6
* run ls /dev and look for something like /dev/cu.usbserial-D200PMQM, exact name is required to start session
* turn on board into PC position
* optional: run OpenBCI_GUI for visual inspection of signal / montage and for impendance check. Stop session before running session_start.py
* it is recommended to verify/modify session_start.py and sd_convert default settings inside script, according to montage and repo location: working_dir (absolute repo path), duration, channels, gain, sampling rate, sd_dir. The default montage is {'F7-T3':0,'F8-T3':1,'O2-T3':2} with sf = 250Hz, gain = 24 and 24H duration, sd_dir is /Volumes/BCI (used only in sd_convert.py)
* verify line for correct device port location: board = pyOpenBCI.OpenBCICyton(port='/dev/cu.usbserial-D200PMQM', daisy=False)
* run python3 session_start.py
* watch for any error messages to appear. Script will use serial interface to check board mode, will set sf, gain, setup channels and allocate space for recording on sd card, will save allocated file name, re-check sf and start session if everything is fine. Immediately after start it will save recording filename, current time and all settings into sqlite db
* board with modded [firmware](https://github.com/roflecopter/OpenBCI_Cyton_Library_SD) will start turn on / off LED every 5s to confirm SD recording is started
* to stop session just turn off board, pull out sd card and insert it into macbook
* run sd_convert.py
* script will list sd_dir files, and then tries to find session start information for that file from sqlite db. If nothing found default settings will be used.
* as a result .BDF and .CSV with uV for EEG and g for ACCEL values will be created inside 'data' directory (create it if is not exist before starting a session)
* it is recommended to move .TXT file out of SD card to make it empty, and backup it somewhere else for possible re-processing in a future.
* i didnt test scripts on windows / linux, but i assume that with no / little modifications they will work, because based on standard libraries.
* if something is not working and you want it to work / or want to add functionality - feel free to send pull request with fix / functionality. You can open issue and ask questions.
* P.S. i'm not a python developer so my code might doesnt look well.
