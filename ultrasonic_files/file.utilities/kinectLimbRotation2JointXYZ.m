function [R] = kinectLimbRotation2JointXYZ(limbRots,referenceVector,X,limbs)
% Thomas S. Murray & Daniel R. Mendat
% 2014/01/17
%
% This function is used to check the limb rotation "coordinates" by
% reconstructing the original XYZ joints.
  
  limbLengths = sqrt(sum((X(limbs(:,1),:) - X(limbs(:,2),:)).^2,2));
  
  scaledVector = -(limbLengths * referenceVector)';
  
%   R( 1,:) = X( 1,:);  % hip center, used to rreferenceVectorot skeleton
%   R( 2,:) = R( 1,:) + scaledVector( 1,:)*vrrotvec2mat(limbRots( 1,:));  % Spine = 2, not used
%   R( 3,:) = R( 2,:) + scaledVector( 2,:)*vrrotvec2mat(limbRots( 2,:));  % Shoulder_Center = 3;
%   R( 4,:) = R( 3,:) + scaledVector( 3,:)*vrrotvec2mat(limbRots( 3,:));  % Head = 4;
%   R( 5,:) = R( 3,:) + scaledVector( 4,:)*vrrotvec2mat(limbRots( 4,:));  % Shoulder_Left = 5;
%   R( 6,:) = R( 5,:) + scaledVector( 5,:)*vrrotvec2mat(limbRots( 5,:));  % Elbow_Left = 6;
%   R( 7,:) = R( 6,:) + scaledVector( 6,:)*vrrotvec2mat(limbRots( 6,:));  % Wrist_Left = 7;
%   R( 8,:) = R( 7,:) + scaledVector( 7,:)*vrrotvec2mat(limbRots( 7,:));  % Hand_Left = 8;
%   R( 9,:) = R( 3,:) + scaledVector( 8,:)*vrrotvec2mat(limbRots( 8,:));  % Shoulder_Right = 9;
%   R(10,:) = R( 9,:) + scaledVector( 9,:)*vrrotvec2mat(limbRots( 9,:));  % Elbow_Right = 10;
%   R(11,:) = R(10,:) + scaledVector(10,:)*vrrotvec2mat(limbRots(10,:));  % Wrist_Right = 11;
%   R(12,:) = R(11,:) + scaledVector(11,:)*vrrotvec2mat(limbRots(11,:));  % Hand_Right = 12;
%   R(13,:) = R( 1,:) + scaledVector(12,:)*vrrotvec2mat(limbRots(12,:));  % Hip_Left = 13;
%   R(14,:) = R(13,:) + scaledVector(13,:)*vrrotvec2mat(limbRots(13,:));  % Knee_Left = 14;
%   R(15,:) = R(14,:) + scaledVector(14,:)*vrrotvec2mat(limbRots(14,:));  % Ankle_Left = 15;
%   R(16,:) = R(15,:) + scaledVector(15,:)*vrrotvec2mat(limbRots(15,:));  % Foot_Left = 16; 
%   R(17,:) = R( 1,:) + scaledVector(16,:)*vrrotvec2mat(limbRots(16,:));  % Hip_Right = 17;
%   R(18,:) = R(17,:) + scaledVector(17,:)*vrrotvec2mat(limbRots(17,:));  % Knee_Right = 18;
%   R(19,:) = R(18,:) + scaledVector(18,:)*vrrotvec2mat(limbRots(18,:));  % Ankle_Right = 19;
%   R(20,:) = R(19,:) + scaledVector(19,:)*vrrotvec2mat(limbRots(19,:));  % Foot_Right = 20;
  
  R( 1,:) = X( 1,:);  % hip center, used to rreferenceVectorot skeleton
  R( 2,:) = R( 1,:) + (vrrotvec2mat(limbRots( 1,:))*scaledVector(:, 1))'; % Spine = 2, not used
  R( 3,:) = R( 2,:) + (vrrotvec2mat(limbRots( 2,:))*scaledVector(:, 2))'; % Shoulder_Center = 3;
  R( 4,:) = R( 3,:) + (vrrotvec2mat(limbRots( 3,:))*scaledVector(:, 3))'; % Head = 4;
  R( 5,:) = R( 3,:) + (vrrotvec2mat(limbRots( 4,:))*scaledVector(:, 4))'; % Shoulder_Left = 5;
  R( 6,:) = R( 5,:) + (vrrotvec2mat(limbRots( 5,:))*scaledVector(:, 5))'; % Elbow_Left = 6;
  R( 7,:) = R( 6,:) + (vrrotvec2mat(limbRots( 6,:))*scaledVector(:, 6))'; % Wrist_Left = 7;
  R( 8,:) = R( 7,:) + (vrrotvec2mat(limbRots( 7,:))*scaledVector(:, 7))'; % Hand_Left = 8;
  R( 9,:) = R( 3,:) + (vrrotvec2mat(limbRots( 8,:))*scaledVector(:, 8))'; % Shoulder_Right = 9;
  R(10,:) = R( 9,:) + (vrrotvec2mat(limbRots( 9,:))*scaledVector(:, 9))'; % Elbow_Right = 10;
  R(11,:) = R(10,:) + (vrrotvec2mat(limbRots(10,:))*scaledVector(:,10))'; % Wrist_Right = 11;
  R(12,:) = R(11,:) + (vrrotvec2mat(limbRots(11,:))*scaledVector(:,11))'; % Hand_Right = 12;
  R(13,:) = R( 1,:) + (vrrotvec2mat(limbRots(16,:))*scaledVector(:,16))'; % Hip_Left = 13;
  R(14,:) = R(13,:) + (vrrotvec2mat(limbRots(17,:))*scaledVector(:,17))'; % Knee_Left = 14;
  R(15,:) = R(14,:) + (vrrotvec2mat(limbRots(18,:))*scaledVector(:,18))'; % Ankle_Left = 15;
  R(16,:) = R(15,:) + (vrrotvec2mat(limbRots(19,:))*scaledVector(:,19))'; % Foot_Left = 16; 
  R(17,:) = R( 1,:) + (vrrotvec2mat(limbRots(12,:))*scaledVector(:,12))'; % Hip_Right = 17;
  R(18,:) = R(17,:) + (vrrotvec2mat(limbRots(13,:))*scaledVector(:,13))'; % Knee_Right = 18;
  R(19,:) = R(18,:) + (vrrotvec2mat(limbRots(14,:))*scaledVector(:,14))'; % Ankle_Right = 19;
  R(20,:) = R(19,:) + (vrrotvec2mat(limbRots(15,:))*scaledVector(:,15))'; % Foot_Right = 20;
end