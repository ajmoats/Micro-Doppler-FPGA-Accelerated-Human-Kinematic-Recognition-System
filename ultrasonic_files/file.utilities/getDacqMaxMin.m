function [xMin,xMax] = getDacqMaxMin(file)
% Thomas S. Murray
% 2013/10/18
%
% This function extracts only the max/min values from a dacq*.dat file.
%
% for dacq files:
% framesPerFile = 12500
% bytesPerFrame = 1944
% stampOffset = 5

  framesPerFile = 12500;
  bytesPerFrame = 1944;
  offset = 20;
  bytesToGet = 4;
  
  xMin = nan(1,framesPerFile);
  xMax = nan(1,framesPerFile);
  
  fid = fopen(file,'r');
  fseek(fid,offset,-1);
  for ii = 1:framesPerFile
    x = fread(fid,bytesToGet);
    x = double(typecast(uint8(x),'uint16'));    % transform bytes to words
    xMin(ii) = x(1);
    xMax(ii) = x(2);
    fseek(fid,bytesPerFrame-bytesToGet,0);
  end
  fclose(fid);
  
  xMin = (min(xMin)- 32768)*5/65536;
  xMax = (max(xMax)- 32768)*5/65536;
end