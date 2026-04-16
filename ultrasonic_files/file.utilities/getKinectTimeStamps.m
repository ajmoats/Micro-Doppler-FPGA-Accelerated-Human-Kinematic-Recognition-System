function [stamps] = getKinectTimeStamps(file)
% Thomas S. Murray
% 2013/10/14
%
% This function extracts only the timestamps from a kinect*.dat file.
%
% Kinect file format:
% framesPerFile = 3400
%
% +------------+----------+--------------+--------+
% |    Item    |  Offset  | Size (Bytes) | Format |
% +------------+----------+--------------+--------+
% | timestamp  | 0        | 8*8          | uint64 |
% | skel2D     | 64       | 20*2*6*8     | double |
% | skelTrack  | 1984     | 20*6*8       | double |
% | skel3D     | 2944     | 20*3*6*8     | double |
% | imageColor | 5824     | 480*640*3    | uint8  |
% | imageDepth | 927424   | 480*640*2    | uint16 |
% +------------+----------+--------------+--------+
% |   Total    | 1541824  |
% +------------+----------+

  framesPerFile = 3400;
  bytesPerFrame = 1541824;
  stampOffset = 0;
  bytesPerStamp = 64;
  
  stamps = nan(1,framesPerFile);
  
  fid = fopen(file,'r');
  fseek(fid,stampOffset,-1);
  for ii = 1:framesPerFile
    x = fread(fid,8,'uint64=>uint64');
%     bitshift(x,k,N) and bitand(bitshift(x,k),2^N-1)
%     stamps(ii) = bitshift(x(1), 0, 8)+bitshift(x(2), 8,16)+...
%                  bitshift(x(3),16,24)+bitshift(x(4),24,32)+...
%                  bitshift(x(5),32,40)+bitshift(x(6),40,48)+...
%                  bitshift(x(7),48,56)+bitshift(x(8),56,60);
    stamps(ii) = bitand(bitshift(x(1), 0), 2^8-1) ...
               + bitand(bitshift(x(2), 8),2^16-1) ...
               + bitand(bitshift(x(3),16),2^24-1) ...
               + bitand(bitshift(x(4),24),2^32-1) ...
               + bitand(bitshift(x(5),32),2^40-1) ...
               + bitand(bitshift(x(6),40),2^48-1) ...
               + bitand(bitshift(x(7),48),2^56-1) ...
               + bitand(bitshift(x(8),56),2^60-1);
    fseek(fid,bytesPerFrame-bytesPerStamp,0);
  end
  fclose(fid);

end