% Thomas S. Murray
% 2014/09/24
%
% This script investigates the scaling of FFTs and spectrograms.

clc;
clear all;
close all;

num_frames = 12500; % 12500 frames in a 2 minute file

fc = 20E3;
Fs = 100E3;
n  = 960*num_frames;
t  = (0:n-1)/Fs;
f  = (-1/2:1/n:1/2-1/n)*Fs;

nfft = 960*num_frames/5;
noverlap = nfft/2;
w = hamming(nfft);

% 1.0V signal
V1p0.a = (2^16)/5 * 1.0;                                % amplitude (counts)
V1p0.c = V1p0.a * sin(2*pi*fc*t) + 2^15;                % test signal (counts)
V1p0.v = (V1p0.c-2^15)*5/2^16;                          % test signal (volts)
V1p0.V = abs(fftshift(fft(V1p0.v)*2/length(V1p0.v)));   % test signal FFT
V1p0.dBV = 20*log10(V1p0.V);                            % test signal FFT (dBV)
V1p0.W = abs(fftshift(fft(V1p0.v)*2/length(V1p0.v)));
[S,F,T,P] = spectrogram(V1p0.v,nfft,noverlap,nfft,Fs);
V1p0.S = abs(S*2/nfft);
V1p0.dBS = 20*log10(V1p0.S);

display(['Max FFT (V) = ' num2str(max(V1p0.V))]);
display(['Max FFT (dB) = ' num2str(max(V1p0.dBV))]);
display(['Max Spec (V) = ' num2str(max(V1p0.S(:)))]);
display(['Max Spec (dB) = ' num2str(max(V1p0.dBS(:)))]);
