% Thomas S. Murray
% 2014/09/25
  
clc;
clear all;
% close all;
addpath('../file_utilities/');
addpath('../noise/');

[h,x_volts,x_bits] = processFile('dacq_COM04_2015_05_04_15_13_22.dat');   % US25
  
[c_max,c_min,~] = getNoiseLevels('spec');
nfft = 2^14;
Fs = 100E3;
f_spec = (0:1/nfft:1/2)'*Fs;
switch(h.modid(1,:))
    case{'US25','RC25'}
        f_c = 25E3;
    case{'US33','RC33'}
        f_c = 25E3;
    case{'US40','RC40'}
        f_c = 25E3;
    otherwise
        error('script_us_characterization: unexpected module ID.');
end    
f_min = f_c - 1.5E3;
f_max = f_c + 1.5E3;
f_window = intersect(find(f_spec>=f_min),find(f_spec<=f_max));

nhw = 12;
ndw = 960;
nfw = nhw+ndw;
nc = 1;                         % number of channels (1 for ultrasound)

ns = ndw/nc;                    % number of samples per frame
na = round(nfft/ns/4)*ns;       % number of samples to advance 
                                % - integer number of frames
                                % - factor of 4 gets us around 25% advance
no = nfft - na;                 % number of samples to overlap

% delay ~40 seconds to remove motion at the beginning of the file
numSkip = ceil(40/0.0096);
x_volts = x_volts((numSkip-1)*960:end);

[S,F,T] = spectrogram(x_volts,nfft,no,nfft,Fs);
P = 20*log10(abs(S)*2/nfft);

f_fft   = (-1/2:1/length(x_volts):1/2-1/length(x_volts))*Fs;
X_volts = 20*log10(abs(fftshift(fft(x_volts)*2/length(x_volts))));

fg = figure(1);
% im = imagesc(F,T,20*log10(abs(S)),[c_min c_max]);
im = imagesc(F,T,20*log10(abs(S)));
ylabel('Frequency (kHz)');
xlabel('Time (sec)');
colormap('jet');
set(gca,'ydir','normal');