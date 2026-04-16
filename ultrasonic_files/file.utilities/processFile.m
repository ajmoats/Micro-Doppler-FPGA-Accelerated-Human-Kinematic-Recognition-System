function [h,x_volts,x_bits] = processFile(filename)

  % system parameters:
  nHeaderBytes = 24;        % header bytes per frame
  nFrameBytes = 1944;       % bytes per frame
  nFrames = 12500;          % number of frames per file

  % 1) read data:
  fid = fopen(filename,'r');
  x1  = fread(fid,nFrames*nFrameBytes); % read all the frames in the file
  fclose(fid);

  % 3) process data:
  x2 = double(typecast(uint8(x1),'uint16'));     % transform bytes to words
  x3 = reshape(x2,nFrameBytes/2,nFrames);        % separate frames
  x4 = x3(1:nHeaderBytes/2,:);                   % isolate headers

  % 4) check headers:
  h.sync  = [x4(1,:); x4(2,:)]'; 
  h.modid = [char(bitand(bitshift(x4(3,:), 0),bitshift(1,8)-1)); ...
             char(bitand(bitshift(x4(3,:),-8),bitshift(1,8)-1)); ...
             char(bitand(bitshift(x4(4,:), 0),bitshift(1,8)-1)); ...
             char(bitand(bitshift(x4(4,:),-8),bitshift(1,8)-1))]';

  h.index = x4(5,:);
  h.stamp = (x4(6,:)-2^15)*2^0  + ...
            (x4(7,:)-2^15)*2^15 + ...
            (x4(8,:)-2^15)*2^30 + ...
            (x4(9,:)-2^15)*2^45;
  h.rssi   = bitand(bitshift(x4(10,:),0),bitshift(1,8)-1)';
  h.pga_gn = 2.^bitand(bitshift(x4(10,:),-12),bitshift(1,8)-1)';
  h.pga_ch = bitand(bitshift(x4(10,:),-8),bitshift(1,4)-1)';

  h.min_val = x4(11,:)';
  h.max_val = x4(12,:)';
  
  % process data:
  
  switch(h.modid(1,:))
      case 'ASU1'
          nChannels = 4;
      case 'ASU2'
          nChannels = 4;
      case 'GEO1'
          nChannels = 3;
      case 'GEO2'
          nChannels = 3;
      case 'GEO3'
          nChannels = 3;
      case 'VBM1'
          nChannels = 2;
      case 'MONO'
          nChannels = 1;
      case 'US25'
          nChannels = 1;
      case 'US33'
          nChannels = 1;
      case 'US40'
          nChannels = 1;
      case 'RC25'
          nChannels = 1;
      case 'RC33'
          nChannels = 1;
      case 'RC40'
          nChannels = 1;
      otherwise
          error('processFile: unknown module type');
  end
     
  h.nChannels = nChannels;
  
  x5 = x3(nHeaderBytes/2+1:nFrameBytes/2,:);            % isolate data
  x_bits = reshape(x5,nChannels,nFrames*960/nChannels); % channel x data streams
  x_volts  = (x_bits - 32768)*5/65536;                  % ADC counts to volts
  
end