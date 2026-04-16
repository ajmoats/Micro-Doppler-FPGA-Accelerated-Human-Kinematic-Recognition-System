function [x3] = getUltrasoundFrame(file,frameIndex)

  % parameters for dacq ultrasound files:
  bytesPerFrame = 1944;
  dataOffset = 24;
  dataBytes = 960*2;

  fid = fopen(file,'r');
  fseek(fid,(frameIndex-1)*bytesPerFrame+dataOffset,-1);
  x1 = fread(fid,dataBytes);
  fclose(fid);

  % process data:
  x2 = double(typecast(uint8(x1),'uint16'))'; % transform bytes to words
  x3  = (x2 - 32768)*5/65536;                 % DAC counts to volts

end