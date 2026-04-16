function [gain] = getDacqGain(file)
% Thomas S. Murray
% 2014/08/19
%
% This function extracts only the channel gain from a dacq*.dat file.
%
% for dacq files:
% framesPerFile = 12500
% bytesPerFrame = 1944
% stampOffset = 5

%   framesPerFile = 12500;
%   bytesPerFrame = 1944;
  offset = 18;
  bytesToGet = 2;
  
  fid = fopen(file,'r');
  fseek(fid,offset,-1);

  x = fread(fid,bytesToGet);
  x = double(typecast(uint8(x),'uint16'));    % transform bytes to words

  gain = 2^bitand(bitshift(x,-12),bitshift(1,8)-1);
  
  fclose(fid);
  
end