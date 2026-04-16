function [modid] = getDacqModuleID(file)
% Thomas S. Murray
% 2014/08/19
%
% This function extracts only the first module ID from a dacq*.dat file.
%
% for dacq files:
% framesPerFile = 12500
% bytesPerFrame = 1944
% stampOffset = 5

  offset = 4;
  bytesToGet = 4;
  
  fid = fopen(file,'r');
  fseek(fid,offset,-1);
  x = fread(fid,bytesToGet);
  fclose(fid);
  x = double(typecast(uint8(x),'uint16'));    % transform bytes to words

  modid = [char(bitand(bitshift(x(1), 0),bitshift(1,8)-1)) ...
           char(bitand(bitshift(x(1),-8),bitshift(1,8)-1)) ...
           char(bitand(bitshift(x(2), 0),bitshift(1,8)-1)) ...
           char(bitand(bitshift(x(2),-8),bitshift(1,8)-1))];
         
return;