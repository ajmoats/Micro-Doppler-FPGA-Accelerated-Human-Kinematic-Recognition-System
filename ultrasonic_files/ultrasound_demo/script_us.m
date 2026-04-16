% Thomas S. Murray & Daniel R. Mendat
% 2015/06/18
%
% Input Buffer:
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
addpath('..\pio\');
addpath('..\noise\');

% configurable parameters
com_port = '5';                 % com port number
[c_max,~,c_min] = getNoiseLevels('rtc');
nfft = 2^14;                    % length of the fft
nob = 50;                      % number of fft slices in the output buffer
fm = false;

% system parameters
Fs = 100E3;
nhw = 12;
ndw = 960;
nfw = nhw+ndw;
nc = 1;                         % number of channels (1 for ultrasound)

% dependent parameters
po = 1/5;                       % percentage overlap (1/4 -> ~25%)
ns = ndw/nc;                    % number of samples per frame
na = round(po*nfft/ns)*ns;      % number of samples to advance (integer number of frames)                                
no = nfft - na;                 % number of samples to overlap
nib = 2*nfft;                   % number of samples in the input buffer

% read one chunk to get sensor information:
h = pio(0,com_port);
x = pio(3,h,nfw*2);
pio(1,com_port);
x = double(typecast(uint8(x(1:nhw*2)),'uint16'));               % just a single header

% decode header:                                
modid = [char(bitand(bitshift(x(3), 0),bitshift(1,8)-1)); ...   % decode module ID
         char(bitand(bitshift(x(3),-8),bitshift(1,8)-1)); ...
         char(bitand(bitshift(x(4), 0),bitshift(1,8)-1)); ...
         char(bitand(bitshift(x(4),-8),bitshift(1,8)-1))]';
     
% timestamp sanity check:       
stamp = (x(6)-2^15)*2^0  + ...                                  % decode timestamp
        (x(7)-2^15)*2^15 + ...
        (x(8)-2^15)*2^30 + ...
        (x(9)-2^15)*2^45;

unix_time=round(double(stamp)*9.6e-3);
timestring=datestr(addtodate(datenum([1970 1 1 0 0 0]),unix_time,'second'));

% set the sensor specific settings:
fprintf('-> %s\n',modid);
switch(modid)
  case{'US25','RC25'}
    fmin = 25E3 - 1.5E3;
    fmax = 25E3 + 1.5E3;
  case{'US33','RC33'}
    fmin = 33.3333E3 - 1.5E3;
    fmax = 33.3333E3 + 1.5E3;
  case{'US40','RC40'}
    fmin = 40E3 - 1.5E3;
    fmax = 40E3 + 1.5E3;
  otherwise
    error('Unrecognized Module ID.\n');
end

% full ultrasound range
% nchannels = 1;
% fmin = 20E3;
% fmax = 50E3;

fprintf('Module ID: %s\n',modid);
fprintf('FFT Length: %d\n',nfft);
fprintf('Advance: %d\n',na);
fprintf('Overlap: %d\n',no);
fprintf('Input Buffer Length: %d\n',nib);
fprintf('Output Buffer Length: %d\n',nob);

% set up frequency axis
f = (0:Fs/nfft:Fs/2);
f_window = intersect(find(f>=fmin),find(f<=fmax));

ib = randi([-32767 +32767],1,nib);
ob = nan(numel(f_window),nob);
w = hamming(nfft)';

% set up figure(s)
fg = figure(1);
im = imagesc((0:nob-1)/Fs*na,f(f_window)/1E3,ob,[c_min c_max]);
ylabel('Frequency (kHz)');
xlabel('Time (sec)');
title(modid);
colormap('jet');
set(gca,'ydir','normal');

% loop ad infinitum
last_stamp = 1;
last_frame = 1;
ib_idx = 0;
ob_idx = 1;
h = pio(0,com_port);
while(1)

    % get data:
    x0 = pio(3,h,nfw*2*na/ns);
    x1 = double(typecast(uint8(x0),'uint16'));  % convert bytes -> words -> doubles
    x2 = reshape(x1,nfw,na/ns);                 % shape into frame series
    x4 = x2(nhw+1:nfw,:);                       % strip headers
    x5 = reshape(x4,1,na);                      % shape into time series
    
    % decode some diagnostic header information:
    frame = x2(5,:);
    stamp = (x2(6,:)-2^15)*2^0 ...
          + (x2(7,:)-2^15)*2^15 ...
          + (x2(8,:)-2^15)*2^30 ...
          + (x2(9,:)-2^15)*2^45;
    dmin  = x2(11,:);
    dmax  = x2(12,:);
    
    % check for dropped frames/timestamps
    for jj = 1:na/ns
        if(fm)
            if(last_stamp+1~=stamp(jj))
                fprintf('[%d] timestamps dropped = %d (last frame = %d, current frame = %d)\n',jj,stamp(jj)-last_stamp,last_stamp,stamp(jj));
            end
        end
        last_stamp = stamp(jj);
        if(last_frame+1~=frame(jj))
            fprintf('[%d] frames dropped = %d (last frame = %d, current frame = %d)\n',jj,frame(jj)-last_frame,last_frame,frame(jj));
        end
        last_frame = frame(jj);
    end
    
    % update the input buffer data
    if(ib_idx+na > nib)
        ib(ib_idx+1:nib) = x5(1:nib-ib_idx);
        ib(1:ib_idx+na-nib) = x5(nib-ib_idx+1:na);
    else
        ib(ib_idx+(1:na)) = x5;
    end
    
    % update the output buffer data
    if(ib_idx-nfft+1 < 1)
        temp = 20*log10(abs(fft([ib(end-nfft+ib_idx+1:end) ib(1:ib_idx)].*w)*2/nfft));
    else
        temp = 20*log10(abs(fft(ib(ib_idx+(-nfft+1:0)).*w)*2/nfft));
    end
    
    if(dmax==65535)
        ob(:,ob_idx) = c_max*ones(nfft);
    elseif(dmin==1)
        ob(:,ob_idx) = c_min*ones(nfft);
    else
        ob(:,ob_idx) = temp(f_window);
    end

    % update cdata
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
pio(1,com_port);