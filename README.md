I use these scripts as part of my home personal sleep research. OpenBCI allows for gold-standard PSG (EEG / ECG etc) data collection and i use it every day. Read more [here](https://blog.kto.to/hypnodyne-zmax-vs-openbci-eeg-psg)

# sd_convert.py
Script to convert OpenBCI SD card .TXT files to 
* 24-bit BDF with raw ADC values. (I prefer to convert to mV befor analysis)

TODO:
* CSV
* ADC to mV conversion

Since there are no clock inside device, recording timestamp is set to current time and need to be manually updated.

# session_start.py
Script to start OpenBCI session in a single click, usually for sleep EEG acquisiton purposes. 
* Tries to starts session with data saved on sd with 500Hz sampling frequency (need to flash modded [firmware](https://github.com/roflecopter/OpenBCI_Cyton_Library_SD) otherwise it will write with default 250Hz).
* Saves session start timestamp into sqlite file for a future use

TODO: 
* automate sd_convert to take session start from sqlite file and add it to result file
