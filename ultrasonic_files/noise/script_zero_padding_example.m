%   Thomas S. Murray
%   2014/09/24
%
%   This script investigates the effect of zero-padding an FFT.

% clc;
clear all;
close all;

num_frames = 100;

fc = 20E3;
Fs = 100E3;
n  = 960*num_frames;
t  = (0:n-1)/Fs;
f  = (-1/2:1/n:1/2-1/n)*Fs;

x = sin(2*pi*fc*t);
X = abs(fftshift(fft(x,length(x))*2/length(x)));
pow = 17;
Xz = abs(fftshift(fft(x,2^pow)*2/length(x)));
fz = (-1/2:1/2^pow:1/2-1/2^pow)*Fs;

figure();
hold on;
plot(f,X,'bo-');
plot(fz,Xz,'r.:');
hold off;