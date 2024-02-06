# EEG-Automated-Sleep-Stage-Classification

This repository contains the code and resources used for automating sleep stage classification with single-channel Electroencephalography (EEG) data from rats.

## Key Takeaways

- **Wavelet Smoothing**: A Butterworth filter was implemented to smooth raw EEG data and reduce noise, isolating waves between 0.5 and 200 Hz, the known range of rat brain wave frequencies.
- **Feature Selection**: Utilizing a Fourier transformation, we extracted the individual component waves of the smoothed EEG reading. We retained the waves with the five greatest amplitudes along with their frequencies. We also binned waves into biologically significant groups by their frequency (e.g. alpha, beta, detla, etc.) and averaged their amplitudes.
- **Random Forest Model**: A Random Forest Model was utilized to differentiate the three stages of sleep, chosen for its resilience to overfitting and ease of interpretation. 

## Overview

Electroencephalography (EEG) provides insight into brain activity by recording electrical signals via scalp electrodes. Automating the processing of classifying EEG data into sleep stages offers potentially greater efficiency and accuracy in an otherwise timely and subjective process.

## Methodology & Findings

We processed 48 hours of 500 Hz EEG readings from 8 rodents using a Butterworth filter to isolate brain wave frequencies from 0.5 Hz to 200 Hz. Fourier transformation separated the filtered wave into individual frequencies, with the top five amplitudes recorded. The Random Forest model predicted sleep states, achieving high accuracy with F1 scores for each stage as follows: 
- REM=0.72
- Slow-wave Sleep=0.94
- Wakefulness=0.93

## Conclusion & Future Directions

Automated EEG analysis using the Butterworth filter and Random Forest model reliably predicts sleep states in rats. This research holds potential for pharmaceutical development and improved diagnosis of sleep disorders, allowing patterns which may have otherwise gone unnoticed to shine. In order to further expand on this research, we suggest establishing a baseline level of concordance between annotators. It is unknown whether F1 scores of 0.94 are outperforming humans, as we don't know how often humans agree; the gold standard used to measure our model may itself be imperfect. We also recommend occassional confirmation by human annotators to prevent unnoticed model drift, as small changes in the environment may lead to alteration of EEG readings and faulty predictions. 

## Dependencies

- Python 3.7
- NumPy
- SciPy
- Scikit-learn

