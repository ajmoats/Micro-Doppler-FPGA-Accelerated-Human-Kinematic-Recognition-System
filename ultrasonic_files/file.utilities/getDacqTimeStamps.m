function [stamps] = getDacqTimeStamps(file)
% Thomas S. Murray
% 2013/10/14
%
% This function extracts only the timestamps from a dacq*.dat file.
%
% for dacq files:
% framesPerFile = 12500
% bytesPerFrame = 1944
% stampOffset = 5

  framesPerFile = 12500;
  bytesPerFrame = 1944;
  stampOffset = 10;
  bytesPerStamp = 8;
  
  stamps = nan(1,framesPerFile);
  
  fid = fopen(file,'r');
  fseek(fid,stampOffset,-1);
  for ii = 1:framesPerFile
    x = fread(fid,bytesPerStamp);
    x = double(typecast(uint8(x),'uint16'));    % transform bytes to words
    stamps(ii) = (x(1,:)-2^15)*2^0  + ...
                 (x(2,:)-2^15)*2^15 + ...
                 (x(3,:)-2^15)*2^30 + ...
                 (x(4,:)-2^15)*2^45;
    fseek(fid,bytesPerFrame-bytesPerStamp,0);
  end
  fclose(fid);

end