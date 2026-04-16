function [x4] = getGerbilFrame(file,frameIndex)

  % parameters for dacq ultrasound files:
  bytesPerFrame = 1944;
  dataOffset = 24;
  dataBytes = 960*2;
  nChannels = 2;

  fid = fopen(file,'r');
  fseek(fid,(frameIndex-1)*bytesPerFrame+dataOffset,-1);
  x1 = fread(fid,dataBytes);
  fclose(fid);

  % process data:
  x2 = double(typecast(uint8(x1),'uint16'));           % transform bytes to words
  x3 = reshape(x2,nChannels,960/nChannels);            % channel x data streams
  x4 = (x3 - 32768)*5/65536;                          % DAC counts to volts

end