% Thomas S. Murray
% 2014/09/24
%
% This simulation runs a constant amplitude sinusoid through the 
% FFT function.  We look at an exactly represented
% frequency so that almost all the energy falls inside a single FFT bin.
% The hamming window spreads it out a little bit though.  To scale future
% FFTs using this data, set c_min = max @ 2.5 mV and c_max = max @ 2.5V.
%
% Results of running it to steady state:
%
% +---------+----------+----------+
% |  A (V)  | Min (dB) | Max (dB) |
% +---------+----------+----------+
% |  2.5E-4 | -226.43  | -72.0726 |
% |  2.5E-3 | -206.43  | -52.0726 |
% |  1.0    | -154.39  | - 0.0314 |
% |  2.5    | -146.43  |   7.9274 |
% +---------+----------+----------+
%
% So, c_min = -52 dB and c_max = 8 dB.
  
clc;
clear all;
close all;

% a_volts = 2.5E-4;
% a_volts = 2.5E-3;
a_volts = 1.0;
% a_volts = 2.5;

num_frames = 12500;             % 12500 frames in a 2 minute file
nfft = 2^14;
Fs = 100E3;
n  = 960*num_frames;
t  = (0:n-1)/Fs;
nc = 1;                         % number of channels (1 for ultrasound)

f  = (0:1/nfft:1/2)'*Fs;
fc = f(5000);

a_bits = (2^16)/5 * a_volts;
x_bits = a_bits * sin(2*pi*fc*t) + 2^15;
x_volts = (x_bits-2^15)*5/2^16;

f_fft   = (-1/2:1/length(x_volts):1/2-1/length(x_volts))*Fs;
X_volts = abs(fftshift(fft(x_volts)*2/length(x_volts)));
X_dB    = 20*log10(X_volts);

figure(1);
hold on;
plot(f_fft/1E3,X_dB,'b-');
% plot(f_fft/1E3,dB_min*ones(1,length(f_fft)),'k-');
% plot(f_fft/1E3,dB_max*ones(1,length(f_fft)),'k-');
hold off;
xlabel('Frequency (kHz)');
ylabel('Amplitude (dBV)');

dB_min = min(X_dB(:))
dB_max = max(X_dB(:))
