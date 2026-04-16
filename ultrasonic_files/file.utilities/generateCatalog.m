function [catalog] = generateCatalog(directoryPath,catalog)
% Thomas S. Murray
% 2013/11/07
%
% Generates a catalog structure for all of the .dat files in the given
% directory.  This function requires a structure that identifies the 
% sensors used in the data collection.

  % Get all the files in the directory:
  files = dir(directoryPath);
  
  % For each sensor in the catalog...
  for ss = 1:numel(catalog)
    % find the associated data files...
    sensorFiles = find(strncmp(catalog(ss).tag,{files.name}',length(catalog(ss).tag)));
    % for each file...
    for ff = 1:numel(sensorFiles)
      % record the file name...
      catalog(ss).files(ff,:) = strcat(directoryPath,files(sensorFiles(ff)).name);
      % if the file is from a dacq box
      if(~isempty(regexp(catalog(ss).tag,'dacq_\w*','once')))
        % record the available time stamps...
        catalog(ss).stamps(:,ff) = getDacqTimeStamps(catalog(ss).files(ff,:));
        % record the global max/min values...
        catalog(ss).xMin = Inf;
        catalog(ss).xMax = -Inf;
        [xMin,xMax] = getDacqMaxMin(catalog(ss).files(ff,:));
        if(xMin<catalog(ss).xMin)
          catalog(ss).xMin = xMin;
        end
        if(xMax>catalog(ss).xMax)
          catalog(ss).xMax = xMax;
        end
      % if the file is from a kinect
      elseif(~isempty(regexp(catalog(ss).tag,'kinect_\w*','once')))
        % record the available time stamps...
        catalog(ss).stamps(:,ff) = getKinectTimeStamps(catalog(ss).files(ff,:));
      else
        fprintf('[generateCatalog] Unknown sensor type.\n');
      end
    end
  end
  
  % save the catalog with the data:
  save(strcat(directoryPath,'catalog','.mat'),'catalog');

end

