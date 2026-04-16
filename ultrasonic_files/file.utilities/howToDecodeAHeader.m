% Thomas S. Murray
% 2013/03/05
% how to decode a header. And other useful DACQ related snippets.

% x = <1x12 uint16>

% decode header:
sync = [x(1) x(2)];                       % decode sync                                 
modid = [char(bitshift(x(3), 0,8)); ...   % decode module ID
         char(bitshift(x(3),-8,8)); ...
         char(bitshift(x(4), 0,8)); ...
         char(bitshift(x(4),-8,8))]; ...
index = x(5);                             % decode index
stamp = (x(6)-2^15)*2^0  + ...            % decode timestamp
        (x(7)-2^15)*2^15 + ...
        (x(8)-2^15)*2^30 + ...
        (x(9)-2^15)*2^45;
rssi = bitshift(x(10),0,8);               % decode signal strength
gain = 2.^bitshift(x(10),-12,8);          % decode amplifier gain
mode = bitshift(x(10),-8,4);              % decode amplifier operating mode
vmin = (double(x(11))-32768)*(5/65536);   % decode minimum (volts)
vmax = (double(x(12))-32768)*(5/65536);   % decode maximum (volts)
cmin = double(x(11));
cmax = double(x(12));

% spec max/min
% S_max = 20*log10(abs(spectrogram_lite(+2.5*ones(1,nfft),w,nfft)));
% S_min = 20*log10(abs(spectrogram_lite(-2.5*ones(1,nfft),w,nfft)));