% Thomas S. Murray
% 2013/04/01
%
% Script to compute ideal spectrogram max/min for image scaling purposes.

clc;
clear all;
close all;

xmax = 2.5 * ones(12500*960,1);
xmin = eps * ones(12500*960,1);

Fs = 100E3;
nfft = 2^13;
noverlap = nfft/2;

Smax = spectrogram(xmax,nfft,noverlap,nfft,Fs);
Smin = spectrogram(xmin,nfft,noverlap,nfft,Fs);

% SmaxMax = max(max(20*log10(abs(Smax))));    % 80.8736 dB
% SmaxMin = min(min(20*log10(abs(Smax))));    % -Inf
% SminMin = min(min(20*log10(abs(Smin))));    % -Inf
% SminMax = max(max(20*log10(abs(Smin))));    % -240.1564 dB

% Smax = max(max(20*log10(abs(spectrogram(xmax,nfft,noverlap,nfft,Fs)))));    % 80.8736 dB
% Smin = max(max(20*log10(abs(spectrogram(xmin,nfft,noverlap,nfft,Fs)))));    % -240.1564 dB

