import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Denoiser(torch.nn.Module):
    def __init__(self, thr=1, redRate=4): #0.3 5 - тоже неплохо
        super(Denoiser, self).__init__()
        self.noise_profile = None
        self.thr = thr
        self.redRate = redRate
        self.n_fft = 1024
        self.hop_length = 256
    
    def fit(self, noiseSample):
        stft_noise = torch.stft(noiseSample, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, return_complex=True)
    
        # Оценка профиля шума
        self.noise_profile = torch.mean(torch.abs(stft_noise), dim=-1)

        # Нормализация профиля шума в диапазоне [0, 1]
        #self.noise_profile = (self.noise_profile - torch.min(self.noise_profile)) / (torch.max(self.noise_profile) - torch.min(self.noise_profile))
        #self.noise_profile = self.noise_profile.unsqueeze(-1)


    def forward(self, audioWav):
        # Вычисляем STFT
        stft_audio = torch.stft(audioWav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, return_complex=True)
        amplitude = torch.abs(stft_audio)

        # Денойзинг
        noise_profile_broad = self.noise_profile.unsqueeze(-1).expand_as(amplitude)

        # Вычисление маски шума
        mask = (1 - noise_profile_broad * self.thr / (amplitude + 1e-10)) / self.redRate

        # Считаем F'
        F_hatch = amplitude * mask

        # Получение фазы исходного сигнала
        phase = stft_audio.angle()

        # Объединение измененной амплитуды с фазой
        processed_stft = torch.polar(F_hatch, phase)

        # iSTFT для восстановления аудио сигнала
        cleaned_audio = torch.istft(processed_stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, length=audioWav.size(-1))

        return cleaned_audio
    
    def visualize(self, original_audio, denoised_audio):
        # Обрезаем оригинальное аудио до размера очищенного аудио
        original_audio = original_audio[:denoised_audio.shape[0]]

        fig, axes = plt.subplots(3, 2, figsize=(12, 8))

        # Original Audio
        axes[0, 0].plot(original_audio.numpy())
        axes[0, 0].set_title('Waveform of Original Audio')

        # Spectrogram of original audio
        D = np.abs(np.fft.rfft(original_audio.numpy()))
        axes[0, 1].specgram(original_audio.numpy(), NFFT=self.n_fft, Fs=2, Fc=0, noverlap=self.hop_length, cmap='inferno', aspect='auto', vmin=-40, vmax=0)
        axes[0, 1].set_title('Spectrogram of Original Audio')

        # Removed noise
        noise = original_audio - denoised_audio
        axes[1, 0].plot(noise.numpy())
        axes[1, 0].set_title('Waveform of Removed Noise')

        # Spectrogram of removed noise
        D = np.abs(np.fft.rfft(noise.numpy()))
        axes[1, 1].specgram(noise.numpy(), NFFT=self.n_fft, Fs=2, Fc=0, noverlap=self.hop_length, cmap='inferno', aspect='auto', vmin=-40, vmax=0)
        axes[1, 1].set_title('Spectrogram of Removed Noise')
        
        # Denoised audio
        axes[2, 0].plot(denoised_audio.numpy())
        axes[2, 0].set_title('Waveform of Denoised audio')

        #  Spectrogram of denoised audio
        D = np.abs(np.fft.rfft(denoised_audio.numpy()))
        axes[2, 1].specgram(denoised_audio.numpy(), NFFT=self.n_fft, Fs=2, Fc=0, noverlap=self.hop_length, cmap='inferno', aspect='auto', vmin=-40, vmax=0)
        axes[2, 1].set_title('Spectrogram of Denoised audio')

        plt.tight_layout()
        plt.show()

        
    def plot_windows(self, audio):
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),

            # Мы не хотим дополнять входной сигнал, используя паддинг
            center=False,

            # Берем первые (n_fft // 2 + 1) частоты
            onesided=True,

            # Применяем torch.view_as_real на все окна
            return_complex=False, 
        )
        
        spectrogram = spectrum.norm(dim=-1).pow(2)
        spectrogram.shape
        plt.figure(figsize=(20, 5))
        plt.imshow(spectrogram.squeeze().log())
        plt.xlabel('Window number', size=20)
        plt.ylabel('Frequency (Hz)', size=20)
        plt.title('spectrogram', size=20)
        plt.show()