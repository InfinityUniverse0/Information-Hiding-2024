% Fast Fourier Transform (FFT)
clc;
clear;
close all;

% Read Audio
[audio, Fs] = audioread('audio/hello.wav');

% Play the audio
% sound(audio, Fs);

figure;
plot(audio);
title('Time Domain Audio Signal');

audioFFT = fft(audio); % Fast Fourier Transform (FFT)
figure;
plot(abs(fftshift(audioFFT)));
title('FFT Analysis Result');
