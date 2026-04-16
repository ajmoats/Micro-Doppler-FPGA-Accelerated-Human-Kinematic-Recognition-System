% Thomas S. Murray
% 2014/09/24
%
% This simulation runs a constant amplitude sinusoid through the 
% spectrogram function.  We look at an exactly represented
% frequency so that almost all the energy falls inside a single FFT bin.
% The hamming window spreads it out a little bit though.  To scale future
% FFTs using this data, set c_min = max @ 2.5 mV and c_max = max @ 2.5V.
%
% Results of running it to steady state:
%
% +---------+----------+----------+
% |  A (V)  | Min (dB) | Max (dB) |
% +---------+----------+----------+
% |  2.5E-4 | -371.12  | -77.3938 |
% |  2.5E-3 | -362.57  | -57.3938 |
% |  1.0    | -311.38  | - 5.3526 |
% |  2.5    | -303.41  |   2.6062 |
% +---------+----------+----------+
%
% So, c_min = -58 dB and c_max = 3 dB.
  
clc;
clear all;
close all;

a_volts = 2.5E-4;
% a_volts = 2.5E-3;
% a_volts = 1.0;
% a_volts = 2.5;

num_frames = 12500;             % 12500 frames in a 2 minute file
nfft = 2^14;
Fs = 100E3;
n  = 960*num_frames;
t  = (0:n-1)/Fs;

f  = (0:1/nfft:1/2)'*Fs;
fc = f(5000);
f_min = fc - 1.5E3;
f_max = fc + 1.5E3;
f_window = intersect(find(f>=f_min),find(f<=f_max));

a_bits = (2^16)/5 * a_volts;
x_bits = a_bits * sin(2*pi*fc*t) + 2^15;
x_volts = (x_bits-2^15)*5/2^16;


nhw = 12;
ndw = 960;
nfw = nhw+ndw;
nc = 1;                         % number of channels (1 for ultrasound)

ns = ndw/nc;                    % number of samples per frame
na = round(nfft/ns/4)*ns;       % number of samples to advance 
                                % - integer number of frames
                                % - factor of 4 gets us around 25% advance
no = nfft - na;                 % number of samples to overlap

[S,F,T] = spectrogram(x_volts,nfft,no,nfft,Fs);
P = 20*log10(abs(S)*2/nfft);

figure(1);
imagesc(T,F/1E3,P);
xlabel('Time (sec)');
ylabel('Frequency (kHz)');

dB_min = min(P(:))
dB_max = max(P(:))
