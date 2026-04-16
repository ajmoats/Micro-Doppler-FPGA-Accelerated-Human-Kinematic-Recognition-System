function [h,d] = processKinectFile(filename)

  % system parameters:
  n = 3400;           % number of frames per file

  % allocate space:
  h.stamp_pc = nan(1,n);
  h.stamp_fm = nan(1,n);
  h.stamp_color = nan(1,n);
  h.stamp_depth = nan(1,n);
  d.JointImageIndices = nan(n,20,2,6);
  d.JointTrackingState = nan(n,20,6);
  d.JointWorldCoordinates = nan(n,20,3,6);
  d.imgColor = zeros(480,640,3,n,'uint8');
  % imgDepth = zeros(480,640,n,'uint16');
  
  % read data:
  fid = fopen(filename,'r');
  for ii = 1:n
    h.stamp_pc(1,ii) = fread(fid,1,'double');
    data = fread(fid,8,'uint64=>uint64');
    tsFM = bitshift(data(1),0,8)+bitshift(data(2),8,16)+...
           bitshift(data(3),16,24)+bitshift(data(4),24,32)+...
           bitshift(data(5),32,40)+bitshift(data(6),40,48)+...
           bitshift(data(7),48,56)+bitshift(data(8),56,60);
    h.stamp_fm(1,ii) = tsFM;
    h.stamp_color(1,ii) = fread(fid,1,'double');
    h.stamp_depth(1,ii) = fread(fid,1,'double');
    d.JointImageIndices(ii,:,:,:) = reshape(fread(fid,20*2*6,'double'),20,2,6);
    d.JointTrackingState(ii,:,:) = reshape(fread(fid,20*6,'double'),20,6);
    d.JointWorldCoordinates(ii,:,:,:) = reshape(fread(fid,20*3*6,'double'),20,3,6);        
    d.imgColor(:,:,:,ii) = reshape(fread(fid,480*640*3,'uint8=>uint8'),480,640,3);
  end
  
  fclose(fid);

end