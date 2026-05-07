# Speech Denoising with U-Net

A PyTorch project that trains a U-Net to remove background noise from human speech. The network learns to predict time-frequency masks over magnitude spectrograms, which are multiplied with the noisy input to produce a denoised estimate. Trained and evaluated on VoiceBank-DEMAND.

On the full 824-file test set, the final model improves SI-SDR by 9.466 dB over the noisy baseline.

## How it works

A noisy waveform is converted to a complex spectrogram via STFT. The magnitude is log-compressed and passed to a U-Net, which outputs a mask the same shape as the spectrogram with values between 0 and 1. The mask is multiplied elementwise with the noisy magnitude, recombined with the original noisy phase, and inverse-STFT'd back to a waveform. The network is trained with MSE between the predicted and clean magnitudes.

## Architecture and ablation

The main model is a four-stage U-Net with skip connections, operating on single-channel magnitude spectrograms. Two variants were trained for comparison: a no-skip version of the same depth, and a shallow version with two stages instead of four. Each was trained for 35 epochs with a learning rate selected from a short sweep over 1e-3, 1e-4, and 1e-5.

| Model          | SI-SDR Improvement |
| -------------- | ------------------ |
| Standard U-Net | 9.466 dB           |
| No-Skip        | 9.269 dB           |
| Shallow        | 9.089 dB           |

Less than 0.4 dB separates the three, which suggests the encoder-decoder structure is doing most of the work. Skip connections still helped, but by less than the literature led me to expect.

## Running it

The notebook runs end-to-end on the included 8-sample test subset using the bundled pretrained weights, no dataset download required.

```
pip install "torch<2.7.1" "torchaudio<2.7.1" soundfile torchmetrics matplotlib numpy tqdm
jupyter lab audio_denoising.ipynb
```

To reproduce training, download VoiceBank-DEMAND from datashare.ed.ac.uk/handle/10283/2791 and place `clean_trainset_28spk_wav` and `noisy_trainset_28spk_wav` next to the notebook.

## Limitations

The model uses a magnitude-only mask and reuses the noisy phase during reconstruction, which puts a hard ceiling on reconstruction quality. The MSE training objective is also not perfectly aligned with the SI-SDR evaluation metric. State-of-the-art systems on this dataset reach 15 dB or more, mostly by addressing both issues with complex-valued masks, SI-SDR losses, and recurrent or transformer architectures that handle long-range temporal context better than CNNs.

## Author

William Shaw. Built for Comp 432 Machine Learning, April 2026.
