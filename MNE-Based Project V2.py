#MNE-Based Project for McGil Application

import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

# loads the sample dataset from the mne library
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60).load_data()

# perform ICA
rawF = raw.copy().filter(5,35) #low pass filter
ica = mne.preprocessing.ICA(n_components=20, random_state=0)
ica.fit(rawF)
ica.plot_components(outlines="head")
ica.exclude = [0, 4, 14, 16] #exclude the ones that i think as bad

# to compare the raw, filtered, and filtered+excluded brain waves
raw.plot()
rawF.plot()
ica.apply(rawF, exclude = ica.exclude).plot()

# get the events & epochs
events = mne.find_events(raw)
mne.viz.plot_events(events[:100])

event_ids = {"standard/stimulus": 1, "target/stimulus": 5}
epochs = mne.Epochs(rawF, events, event_id = event_ids, preload =True)
epochs.plot() 

epochs = ica.apply(epochs, exclude = ica.exclude) 
epochs.apply_baseline((None, 0))
epochs.info # to get info about the events and time rage for myself


epochs["target"].plot_image(picks =[2]) #I made the illustration on channel 2, could be any from the 64
epochs.equalize_event_counts(event_ids) #to even out no. of standad to target stimulus

#to save in the local directory since this sample dataset is imported from the library itself
epochs.save("MNEbasedProject-epo.fif") 
