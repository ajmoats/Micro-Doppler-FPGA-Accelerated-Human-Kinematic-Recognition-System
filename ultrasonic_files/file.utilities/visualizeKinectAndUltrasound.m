function [] = visualizeKinectAndUltrasound(catalog)

  % min/max timestamps:
%   ts0 = min(min(catalog.dacqStamps(:)),min(catalog.kinectStamps(:)));
  ts0 = min(catalog.kinectStamps(:));
  tsE = max(max(catalog.dacqStamps(:)),max(catalog.kinectStamps(:)));
  
  % Example of FFT windowing and overlap scheme:
  % numFramesPerFFT = 9; (NF)
  % numFramesOverlap = 4; (NO)
  % numSpecSlices = 100; (NS)
  % 
  % +--------+-------+-------+-------+-----+------------------+-----+---------+
  % | slices |   1   |   2   |   3   | ... |        n         | ... |   100   |
  % | frames |  1-9  |  6-14 | 11-19 | ... | n*(NF-NO)+(1:NF) | ... | 501-509 |
  % +--------+-------+-------+-------+-----+------------------+-----+---------+
  
  % ultrasound parameters:
  numFramesPerFFT = 9;
  numFramesOverlap = 4;
  numFFT = numFramesPerFFT*960;
  numSpecSlices = 100;
  T = (-(numSpecSlices-1):0)*9.6E-3*(numFramesPerFFT-numFramesOverlap);
  
  fmin = (33+1/3)*1E3 - 1E3;
  fmax = (33+1/3)*1E3 + 1E3;
  Fs = 100E3;
  F = (0:Fs/numFFT:Fs/2)';
  fIdx = intersect(find(F>=fmin),find(F<=fmax));
  F = F(fIdx);
  
  % kinect parameters:
  SkeletonConnectionMap = [[1 2];     % Spine: hip center to spine
                         [2 3];     % spine to shoulder center
                         [3 4];     % shoulder center to head
                         [3 5];     % Left Hand: shoulder center to left shoulder
                         [5 6];     % left shoulder to elbow
                         [6 7];     % left elbow to wrist
                         [7 8];     % left wrist to hand
                         [3 9];     % Right Hand: shoulder center to right shoulder
                         [9 10];    % right shoulder to elbow
                         [10 11];   % right elbow to wrist
                         [11 12];   % right wrist to hand
                         [1 17];    % Right Leg: hip center to right hip
                         [17 18];   % right hip to knee
                         [18 19];   % right knee to ankle
                         [19 20];   % right ankle to foot
                         [1 13];    % Left Leg: hip center to left hip
                         [13 14];   % left hip to knee
                         [14 15];   % left knee to ankle
                         [15 16]];  % left ankle to foot
                       
  % buffer setup:
  xFFT = zeros(1,numFFT);
  S = repmat(abs(fft(ones(numFFT,1).*catalog.xMin.*hamming(numFFT))),1,numSpecSlices);
  S(:,end) = abs(fft(ones(numFFT,1).*catalog.xMax.*hamming(numFFT)));
  S = 20*log10(S(fIdx,:));  
  xSpec = S;
    
  % figure setup:
  figure(1);
  subplot(2,1,1);
  kt = imagesc(zeros(480,640));   % kinect color image
  sk = nan(19,1);                 % kinect skeleton
  for ii = 1:19
      sk(ii) = line([1 1],[1 1], 'LineWidth', 1.5, 'LineStyle', '-', 'Marker', '+', 'Color', 'g');
  end
  ht = title('Timestamp = Init'); % title
  subplot(2,1,2);
  us = imagesc(T,F/1E3,S);        % ultrasound spectrogram
  xlabel('Time (s)');
  ylabel('Frequency (kHz)');
  
  newFrameCount = 0;
  for tt = ts0:tsE
    tic
    % update timestamp
    set(ht,'string',['timestamp = ' num2str(tt)]);
    
    % update kinect
    [I] = find(catalog.kinectStamps==tt);
    if(isempty(I))
      % do nothing
    else
      % image
      [I,J] = ind2sub(size(catalog.kinectStamps),I);
      set(kt,'cdata',getKinectImage(catalog.kinectFiles(J,:),I));
      % skeleton
      [skID,skJoints] = getKinectSkeleton2D(catalog.kinectFiles(J,:),I);
      if(skID>0)
        for kk = 1:19
          x = [skJoints(SkeletonConnectionMap(kk,1),1,skID) skJoints(SkeletonConnectionMap(kk,2),1,skID)];
          y = [skJoints(SkeletonConnectionMap(kk,1),2,skID) skJoints(SkeletonConnectionMap(kk,2),2,skID)];     
          set(sk(kk),'xdata',x,'ydata',y);
        end
      end
    end
    
    % update ultrasound
    [I] = find(catalog.dacqStamps==tt);
    if(isempty(I))
      x = zeros(1,960);
    else
      [I,J] = ind2sub(size(catalog.dacqStamps),I);
      x = getUltrasoundFrame(catalog.dacqFiles(J,:),I);
      x = (x-catalog.xMin)/catalog.xMax;
    end
    
    xFFT = [xFFT(960+(1:(numFramesPerFFT-1)*960)) x];
    newFrameCount = newFrameCount + 1;
    if(newFrameCount == numFramesPerFFT-numFramesOverlap)
      S = 20*log10(abs(fft(xFFT'.*hamming(numFFT))));
      S = S(fIdx);
      xSpec = [xSpec(:,2:numSpecSlices) S];
      set(us,'cdata',xSpec);
      newFrameCount = 0;
    end
    
    drawnow;
%     pause(0.005);
    toc
  end
  
end