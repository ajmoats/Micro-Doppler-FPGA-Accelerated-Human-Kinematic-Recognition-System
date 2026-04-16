function [imageColor] = getKinectImage(file,frameIndex)
% Thomas S. Murray
% 2013/10/14
%
% This function extracts only the RGB image from a kinect*.dat file.
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

  % parameters for kinect files:
  bytesPerFrame = 1541824;
  imageOffset = 5824;

  fid = fopen(file,'r');
  fseek(fid,(frameIndex-1)*bytesPerFrame+imageOffset,-1);
  imageColor = reshape(fread(fid,480*640*3,'uint8=>uint8'),480,640,3);
  fclose(fid);

end