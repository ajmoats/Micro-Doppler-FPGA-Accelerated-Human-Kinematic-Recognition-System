function [skROT] = skelXYZ2ROT(skXYZ,skMap)
% Thomas S. Murray
% 2014/11/11
%
% Input:
%   skXYZ - 3D skeleton in cartesian coordinates [numJoints x 3]
%   skMap - mapping from 3D skeleton (joints) to limbs (joint pairs) [numLimbs x 2]
%
% Output:
%   skROT - 3D skeleton in limb rotations augmented with hip translation [numLimbs+1 x 3]

  numLimbs   = size(skMap,1);

  % translate skeleton and limbs
  hip_center = skXYZ(1,:);                                % hip is first joint
  skXYZ      = skXYZ - repmat(hip_center,20,1);           % anchor skeleton (remove hip translation)
  limbXYZ    = skXYZ(skMap(:,2),:) - skXYZ(skMap(:,1),:); % limb vectors, all originating from the origin

  % compute rotation representation
  skROT      = nan(numLimbs+1,3);                         % allocate augmented limb rotations
  skROT(1,:) = hip_center;                                % augment w/ hip translation
  for ll = 2:numLimbs+1                                   % loop: through limbs
    if(isequal(limbXYZ(ll-1,:),[0 0 0]))                    %   check: for zero limb vecotrs
      skROT(ll,:) = [0 0 0];                              %     set rotation to zero
    else                                                  %   otherwise:
      limbROT = vrrotvec(limbXYZ(ll-1,:),[0 0 1]);          %     limb rotation referenced to 3rd dimension
      skROT(ll,:) = [limbROT(1) limbROT(2) limbROT(4)];   %     remove the always zero 3rd dimension
    end
  end     

return;