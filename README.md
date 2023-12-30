I run these scripts at my **macbook** as part of my home personal sleep research to start session and collect data from OpenBCI. 
OpenBCI allows for gold-standard PSG (EEG, ECG etc) data collection and i use it every day. Read more [here](https://blog.kto.to/hypnodyne-zmax-vs-openbci-eeg-psg)

# session_start.py
Script to start OpenBCI session in a single click, usually for sleep EEG acquisiton purposes. 
* Used to starts session with data saved on sd with desired sampling frequency (for greater than 250Hz modded [firmware](https://github.com/roflecopter/OpenBCI_Cyton_Library_SD) need to be flashed, otherwise it will always write with default 250Hz).
* Saves session start timestamp and settings into sqlite file. Session info will be joined in sd_convert.py script

# sd_convert.py
Script to convert OpenBCI SD card .TXT files to 
* 24-bit BDF with raw ADC values. (I prefer to convert from ADC to mV after reading BDF)
* Recording timestamp and settings taken from sqlite db which automatically created and updated by session_start.py script

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
* as a result .BDF with raw ADC values and .CSV with uV values will be created inside 'data' directory (create it if is not exist before starting a session)
* it is recommended to move .TXT file out of SD card to make it empty, and backup it somewhere else for possible re-processing in a future.
* i didnt test scripts on windows / linux, but i assume that with no / little modifications they will work, because based on standard libraries.
