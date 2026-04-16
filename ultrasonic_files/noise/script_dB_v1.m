clc;
clear all;
close all;

fc = 20E3;
Fs = 500E3;
n = 1000;
t = (0:n-1)/Fs;
f = (-1/2:1/n:1/2-1/n)*Fs;

% 1V signal
a = 2^16/5*1;
x = a*sin(2*pi*t*fc)+2^15;
x = (x-32768)*5/65536;
% X = 20*log10(abs(fftshift(fft(x)*2/n)));    % normalized
X = 20*log10(abs(fftshift(fft(x))));        % un-normalized (what I've been doing)

% 10 mV noise
b = 2^16/5*10E-3;
y = b*sin(2*pi*t*fc)+2^15;
y = (y-32768)*5/65536;
% Y = 20*log10(abs(fftshift(fft(y)*2/n)));    % normalized
Y = 20*log10(abs(fftshift(fft(y))));        % un-normalized (what I've been doing)

figure(1);
plot(f/1E3,X,'b-',f/1E3,Y,'r-');

figure(2);
plot(t,x,'b.-',t,y,'r.-');