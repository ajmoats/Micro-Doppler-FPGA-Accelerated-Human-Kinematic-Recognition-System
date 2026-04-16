% Thomas S. Murray
% 2014/09/25
%
% This simulation runs a constant amplitude sinusoid through the real-time
% ultrasound monitoring algorithm.  We look at an exactly represented
% frequency so that almost all the energy falls inside a single FFT bin.
% The hamming window spreads it out a little bit though.  To scale future
% FFTs using this data, set c_min = max @ 2.5 mV and c_max = max @ 2.5V.
%
% Results of running it to steady state:
%
% +---------+----------+----------+
% |  A (V)  | Min (dB) | Max (dB) |
% +---------+----------+----------+
% |  2.5E-4 | - 151.09 |   4.9564 |
% |  2.5E-3 | - 151.58 |  24.9564 |
% |  1.0    | - 104.34 |  76.9976 |
% |  2.5    | - 96.399 |  84.9564 |
% +---------+----------+----------+
%
% So, c_min = 24 dB and c_max = 85 dB.
%
% Input Buffer Diagram:
%
%                     nib
% <---------------------------------------->
%
% |                   ib                   |
% +----------------------------------------+
%           |----------|
%               |----------|
%                   |----------|
%                    <----> <->
%                      no    na
%
% nfft = no + na
% no = length of overlap
% na = length of advance (must be a multiple of ns)
% ns = number of samples per frame (960/nc)
% nc = number of channels

clc;
clear all;
close all;
addpath('../pio/');

% configurable parameters
% a_volts = 2.5E-4;
% a_volts = 2.5E-3;
a_volts = 1.0;
% a_volts = 2.5;
nfft = 2^14;                    % length of the fft
nob = 100;                      % number of fft slices in the output buffer

% system parameters
Fs = 100E3;
nhw = 12;
ndw = 960;
nfw = nhw+ndw;
nc = 1;                         % number of channels (1 for ultrasound)

% signal parameters
f = (0:Fs/nfft:Fs/2);
fc = f(5000);
a_bits = (2^16)/5 * a_volts;
f_min = fc - 1.5E3;
f_max = fc + 1.5E3;
f_window = intersect(find(f>=f_min),find(f<=f_max));

% dependent parameters
po = 1/4;                       % percentage overlap (1/4 -> ~25%)
ns = ndw/nc;                    % number of samples per frame
na = round(po*nfft/ns)*ns;      % number of samples to advance (integer number of frames)                                
no = nfft - na;                 % number of samples to overlap
nib = 2*nfft;                   % number of samples in the input buffer

fprintf('FFT Length: %d\n',nfft);
fprintf('Advance: %d\n',na);
fprintf('Overlap: %d\n',no);
fprintf('Input Buffer Length: %d\n',nib);
fprintf('Output Buffer Length: %d\n',nob);

ib = randi([-32767 +32767],1,nib);
ob = nan(numel(f_window),nob);
w = hamming(nfft)'*2/nfft;

% set up figure(s)
fg = figure(1);
im = imagesc((0:nob-1)/Fs*na,f(f_window)/1E3,ob);
ylabel('Frequency (kHz)');
xlabel('Time (sec)');
ht = title(num2str(0));

% loop ad infinitum
last_stamp = 1;
last_frame = 1;
ib_idx = 0;
ob_idx = 1;
offset = 0;
num_iters = 1000;
dB_min = nan(num_iters,1);
dB_max = nan(num_iters,1);
for ii = 1:num_iters

    % make data:
    t = (offset + (0:na-1))/Fs;
    x5 = a_bits * sin(2*pi*fc*t) + 2^15;
    offset = offset+na;
    
    % update the input buffer data
    if(ib_idx+na > nib)
        ib(ib_idx+1:nib) = x5(1:nib-ib_idx);
        ib(1:ib_idx+na-nib) = x5(nib-ib_idx+1:na);
    else
        ib(ib_idx+(1:na)) = x5;
    end
    
    % update the output buffer data
    if(ib_idx+nfft > nib)
        temp = 20*log10(abs(fft([ib(ib_idx+1:end) ib(1:ib_idx+nfft-nib)].*w)));
    else
        temp = 20*log10(abs(fft(ib(ib_idx+(1:nfft)).*w)*2/nfft));
    end
    ob(:,ob_idx) = temp(f_window);
    dB_min(ii) = min(temp(f_window));
    dB_max(ii) = max(temp(f_window));
    
    % update cdata
    set(ht,'string',num2str(offset));
    set(im,'cdata',ob);
    drawnow;
    
    % update input buffer index
    ib_idx = ib_idx + na;
    if(ib_idx>nib)
        ib_idx = ib_idx - nib;  
    end
    
    % update output buffer index
    if(ob_idx==nob)
        ob_idx = 1;
    else
        ob_idx = ob_idx + 1;
    end
    
end

dB_min(end)
dB_max(end)

figure(2);
plot(1:num_iters,dB_min,'b.-',1:num_iters,dB_max,'r.-');
xlabel('iterations');
ylabel('Amplitude Bits (dB)');
title('Max/Min dB Values');