% Discrete Wavelet Transform (DWT)
clc;
clear;
close all;

% Read Audio
[audio, Fs] = audioread('audio/hello.wav');

[ca1, cd1] = dwt(audio(:,1), 'db4'); % DWF
reconstruct_audio = idwt(ca1, cd1, 'db4', length(audio(:,1))); % Inverse DWT

% Display Results
figure;
subplot(2, 2, 1); plot(audio(:,1)); title('Original Waveform');
subplot(2, 2, 2); plot(ca1); title('Approxmation Component');
subplot(2, 2, 3); plot(cd1); title('Detail Component');
subplot(2, 2, 4); plot(reconstruct_audio); title('Reconstructed Signal');

[coefs, levels] = wavedec(audio(:,2), 3, 'db4'); % DWT
reconstruct_audio = waverec(coefs, levels, 'db4'); % Inverse DWT

ca3 = appcoef(coefs, levels, 'db4', 3); % 提取第3层的近似系数
cd3 = detcoef(coefs, levels, 3); % 提取第3层的细节系数
cd2 = detcoef(coefs, levels, 2); % 提取第2层的细节系数
cd1 = detcoef(coefs, levels, 1); % 提取第1层的细节系数

% Display Results
figure;
subplot(2, 3, 1); plot(audio(:,2)); title('Original Waveform (Channel 2)');
subplot(2, 3, 2); plot(ca3); title('Approxmation Component (Level 3)');
subplot(2, 3, 3); plot(reconstruct_audio); title('Reconstructed Signal');
subplot(2, 3, 4); plot(cd1); title('Detail Component (Level 1)');
subplot(2, 3, 5); plot(cd2); title('Detail Component (Level 2)');
subplot(2, 3, 6); plot(cd3); title('Detail Component (Level 3)');
