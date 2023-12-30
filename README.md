I use these scripts as part of my home personal sleep research. OpenBCI allows for gold-standard PSG (EEG / ECG etc) data collection and i use it every day. Read more [here](https://blog.kto.to/hypnodyne-zmax-vs-openbci-eeg-psg)

# sd_convert.py
Script to convert OpenBCI SD card .TXT files to 
* 24-bit BDF with raw ADC values. (I prefer to convert to mV befor analysis)
* CSV with ADC to mV conversion
* Recording timestamp and settings taken from sqlite db which automatically created and updated by session_start.py script

# session_start.py
Script to start OpenBCI session in a single click, usually for sleep EEG acquisiton purposes. 
* Tries to starts session with data saved on sd with 500Hz sampling frequency (need to flash modded [firmware](https://github.com/roflecopter/OpenBCI_Cyton_Library_SD) otherwise it will write with default 250Hz).
* Saves session start timestamp and settings into sqlite file. Session info will be joined in sd_convert.py script

