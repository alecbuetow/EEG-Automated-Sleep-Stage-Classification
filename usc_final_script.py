import pandas as pd
import numpy as np
import os
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import spkit as sp
from scipy.signal import savgol_filter
import csv

raw_df = pd.read_csv("combined.csv")
raw_test_df = pd.read_csv("combined_results.csv")


eeg_data = raw_df.iloc[:,:-1]
eeg_test_data = raw_df.iloc[:,:-1]

label_data = raw_df.iloc[:,-1]



#flatter for the butterworth filter
train_data = eeg_data.to_numpy().flatten()
test_data = eeg_test_data.to_numpy().flatten()

#apply butterworth filter
train_data_filtered = sp.filter_X(train_data,band=[0.5,200], btype='bandpass',fs=500,verbose=0)
test_data_filtered = sp.filter_X(test_data,band=[0.5,200], btype='bandpass',fs=500,verbose=0)

#reshape to iterate through epoch (1 row = 1 epoch = 5000 observations (500 observations/s for 10s))
train_data_reshape = train_data_filtered.reshape(-1,5000)
test_data_reshape = test_data_filtered.reshape(-1,5000)

#create an array of frequencies for the component waves that a Fourier transform splits the filtered wave into
fs = 500     
epoch_len = 5000
wave_frequency = np.fft.rfftfreq(epoch_len, 1.0/fs)

#biologicially significant intervals for rat brainwaves
rat_eeg_bins = {'Delta': (1, 4),
                'Theta': (5, 9),
                'Alpha': (10, 15),
                'Beta': (16, 30),
                'Gamma': (31, 48),
                'High Gamma': (52, 95),
                'HFO': (105, 200)}


#create an array of frequencies for the component waves that a Fourier transform splits the filtered wave into
fs = 500     
epoch_len = 5000
wave_frequency = np.fft.rfftfreq(epoch_len, 1.0/fs)

mean_amplitudes = []
for epoch in train_data_reshape:
    #split smoothed wave into its component waves with a Fourier transform and get amplitude of each component wave (performing this operation epoch by epoch)
    wave_amplitudes = np.absolute(np.fft.rfft(epoch))

    #record mean amplitude of all rat brainwaves which fall into each biologically significant bin
    epoch_mean_amplitudes = []
    for bin in rat_eeg_bins:  
        #select bin interval
        bin_lower_limit = rat_eeg_bins[bin][0]
        bin_upper_limit = rat_eeg_bins[bin][1]
        #select the amplitudes for all waves that fall in this frequency bin
        freq_ix = np.where((wave_frequency >= bin_lower_limit) & 
                        (wave_frequency <= bin_upper_limit))
        freq_ix = freq_ix[0]
        #average amplitudes and record
        avg = np.mean(wave_amplitudes[freq_ix])
        epoch_mean_amplitudes.append(avg) 
    
    #record each bin's mean amplitude for every epoch
    mean_amplitudes.append(epoch_mean_amplitudes)

#perform the same steps as above with the testing data
mean_test_amplitudes = []
for epoch in test_data_reshape:
    wave_amplitudes = np.absolute(np.fft.rfft(epoch))

    epoch_mean_test_amplitudes = []
    epoch_sum_test_amplitudes = []
    for bin in rat_eeg_bins:
        bin_lower_limit = rat_eeg_bins[bin][0]
        bin_upper_limit = rat_eeg_bins[bin][1]
        freq_ix = np.where((wave_frequency >= bin_lower_limit) &
                        (wave_frequency <= bin_upper_limit))
        freq_ix = freq_ix[0]
        avg = np.mean(wave_amplitudes[freq_ix])
        epoch_mean_test_amplitudes.append(avg)
        total = np.sum(wave_amplitudes[freq_ix])
        epoch_sum_test_amplitudes.append(total)

    mean_test_amplitudes.append(epoch_mean_test_amplitudes)



#create a feature dataframe: 1 column for each bin of mean amplitdues
mean_amplitudes_df = pd.DataFrame(mean_amplitudes, columns = ['dm', 'tm', 'am', 'bm', 'gm', 'hgm', 'hfom'])
mean_test_amplitudes_df = pd.DataFrame(mean_test_amplitudes, columns = ['dm', 'tm', 'am', 'bm', 'gm', 'hgm', 'hfom'])

#the following chunk of code is redundant, as data processing and model fitting were performed separately 
def eeg_decomposition(eeg_data, sampling_rate=500):
    #repeat EEG wave decomposition with a Fourier transform 
    fft_values = np.abs(np.fft.fft(eeg_data, axis=1))
    n = eeg_data.shape[1]
    freqs = fftfreq(n, d=1/sampling_rate)
    #select 5 waves with greatest amplitudes and their frequencies
    top_peaks_idx = np.argsort(fft_values)[:, -5:]
    top_peaks_freq = freqs[top_peaks_idx]
    top_peaks_amp = fft_values[np.arange(len(eeg_data))[:, None], top_peaks_idx]
    
    return top_peaks_freq, top_peaks_amp, freqs, fft_values

#get frequencies of 5 most significant component waves, their amplitudes, frequencies for all component waves and their amplitudes
eeg_freqs, eeg_amps, freqs, fft_values = eeg_decomposition(eeg_data)
eeg_test_freqs, eeg_test_amps, freqs_test, fft_test_values = eeg_decomposition(eeg_test_data)

#convert this information into a dataframe and combine it with the previous features
combined_data = pd.DataFrame(np.hstack((eeg_freqs, eeg_amps)),
                             columns=[f'freq_{i+1}' for i in range(5)] +
                                     [f'amp_{i+1}' for i in range(5)])
combined_data = pd.concat([combined_data, mean_amplitudes_df, label_data], axis=1)

#repeat for testing data
combined_test_data = pd.DataFrame(np.hstack((eeg_test_freqs, eeg_test_amps)),
                             columns=[f'freq_{i+1}' for i in range(5)] +
                                     [f'amp_{i+1}' for i in range(5)])
combined_test_data = pd.concat([combined_test_data, mean_test_amplitudes_df], axis=1)


#split provided training data into 20% validation 80% true training to assess model performance
X_train, X_test, y_train, y_test = train_test_split(combined_data.iloc[:, :-1], label_data, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

#assess trained Random Forest Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


#make predictions on true testing data
combined_test_data['class'] = model.predict(combined_test_data)

combined_test_data['animal_id'] = (combined_test_data.index // 17280) + 8
combined_test_data['recording_id'] = (combined_test_data.index // 8640) % 2
combined_test_data['sample_index'] = combined_test_data.index % 8640

combined_test_data = combined_test_data[['animal_id', 'recording_id', 'sample_index', 'class']]

# Write the DataFrame to the final CSV file
final_output_file = "test.csv"
combined_test_data.to_csv(final_output_file, index=False)

print("Final output file created successfully.")
