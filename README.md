# Background Noise Suppression with U-Nets

This is a deep learning project that trains U-Nets to remove background noise from human speech. The network learns to predict time-frequency masks over magnitude spectrograms, which are multiplied with the noisy input to produce a denoised spectrogram. The models were trained and evaluated on VoiceBank-DEMAND.

The best model improveed SI-SDR by 9.466 dB over the noisy baseline on the full 824-file test set

## How the Model Denoises

<img width="944" height="265" alt="image" src="https://github.com/user-attachments/assets/63b17062-e150-405e-8440-d97fce7f553f" />

The noisy waveforms are converted to complex spectrograms via STFT. The magnitudes are log-compressed and passed to a U-Net, which outputs masks the same shape as the spectrograms with values between 0 and 1. The masks are multiplied elementwise with the noisy magnitudes, recombined with the original noisy phases, and converted back to waveforms. The network is trained with MSE between the predicted and clean magnitudes.

## Architecture

The main model is a four-stage U-Net with skip connections, operating on single-channel magnitude spectrograms. Two variants were trained for comparison: a no-skip version of the same depth, and a shallow version with two stages instead of four. Each was trained for 35 epochs with a learning rate selected from a short sweep over 1e-3, 1e-4, and 1e-5.

<img width="532" height="314" alt="image" src="https://github.com/user-attachments/assets/1fcfab6f-fef0-4b89-9958-333fcc7632c6" />

<br>

| Model          | SI-SDR Improvement |
| -------------- | ------------------ |
| Standard U-Net | 9.466 dB           |
| No-Skip        | 9.269 dB           |
| Shallow        | 9.089 dB           |

It was found that less than 0.4 dB separates the three models, which suggests the encoder-decoder structure is doing the majority of the work. Skip connections still contributed, but by less than the literature suggested.

## Running it

Requirements:

```
pip install "torch<2.7.1" "torchaudio<2.7.1" soundfile torchmetrics matplotlib numpy tqdm
jupyter lab audio_denoising.ipynb
```

Most sections of the notebook require only the data included in the repo to run. The only exception is Section III: Training. Section III requires the full training set to run. To download the VoiceBank-DEMAND dataset, go to datashare.ed.ac.uk/handle/10283/2791 and download `clean_trainset_28spk_wav` and `noisy_trainset_28spk_wav` and place them next to the notebook.

## Limitations

The model uses a magnitude-only mask and reuses the noisy phase during reconstruction, which puts a hard ceiling on the reconstruction quality. The MSE training objective is also not perfectly aligned with the SI-SDR evaluation metric. State-of-the-art systems on this dataset reach 15 dB or more, mostly by addressing both issues with complex-valued masks, SI-SDR losses, and recurrent or transformer architectures that handle long-range temporal context better than CNNs.

## Author

William Shaw. Built for Comp 432 Machine Learning, April 2026.
