clc;
clear all;
close all;

fc = 20E3;
Fs = 100E3;
n = 1000;
t = (0:n-1)/Fs;
f = (-1/2:1/n:1/2-1/n)*Fs;

% 2.5V signal
a1 = (2^16-1)/5 * 2.5;
x1 = a1*sin(2*pi*t*fc)+2^15;
x1 = (x1-32768)*5/65536;
X1 = 20*log10(abs(fftshift(fft(x1)*2/length(x1))));
dB1 = max(X1);

% 2.5mV signal
a2 = (2^16-1)/5 * 2.5E-3;
x2 = a2*sin(2*pi*t*fc)+2^15;
x2 = (x2-32768)*5/65536;
X2 = 20*log10(abs(fftshift(fft(x2)*2/length(x2))));
dB2 = max(X2);

% 0.25mV signal
a3 = (2^16-1)/5 * 0.25E-3;
x3 = a3*sin(2*pi*t*fc)+2^15;
x3 = (x3-32768)*5/65536;
X3 = 20*log10(abs(fftshift(fft(x3)*2/length(x3))));
dB3 = max(X3);

% 1V signal
a4 = (2^16-1)/5 * 1;
x4 = a4*sin(2*pi*t*fc)+2^15;
x4 = (x4-32768)*5/65536;
X4 = 20*log10(abs(fftshift(fft(x4)*2/length(x4))));
dB4 = max(X4);

figure(1);
plot(f/1E3,X1,'b.-',f/1E3,X2,'r.-',f/1E3,X4,'g.-');

figure(2);
plot(t,x1,'b-',t,x2,'r-',t,x4,'g-');
axis([0 1E-3 -3 3]);