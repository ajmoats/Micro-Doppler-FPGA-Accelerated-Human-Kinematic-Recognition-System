function [limbRots,referenceVector] = kinectJointXYZ2LimbRotation(X,limbs)
% Thomas S. Murray & Daniel R. Mendat
% 2014/01/17
%
% Input:
% X - skeletal joint coordinates [numJoints x 3 (x,y,z)]
% limbs - mapping from limb segments to joints [numLimbs x 2 (joint1,joint2)]
%
% Output:
% limbRots - limb rotation representation [numLimbs x 4] vector triple +
% angle
  
  referenceVector = [0 0 1];  % choose positive z-axis
  numLimbs = size(limbs,1);
  limbRots = zeros(numLimbs,4);
  
  for ss = 1:numLimbs
    limbVector = X(limbs(ss,1),:) - X(limbs(ss,2),:);
    if(isequal(limbVector,[0,0,0]))
      % some limb has zero length, exit
      limbRots = [];
      referenceVector = [];
      break;
    else
      limbRots(ss,:) = vrrotvec(referenceVector,limbVector);
    end
  end

end