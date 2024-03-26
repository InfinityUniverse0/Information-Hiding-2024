% Discrete Consine Transform (DCT)
clc;
clear;
close all;

% Read Audio
[audio, Fs] = audioread('audio/hello.wav');

audioDCT = dct(audio(:,1)); % DCT
reconstruct_audio = idct(audioDCT); % Inverse DCT

% Display Results
subplot(3, 1, 1); plot(audio(:,1)); title('Original Waveform');
subplot(3, 1, 2); plot(audioDCT); title('DCT Processed Signal');
subplot(3, 1, 3); plot(reconstruct_audio); title('Reconstructed Signal');
