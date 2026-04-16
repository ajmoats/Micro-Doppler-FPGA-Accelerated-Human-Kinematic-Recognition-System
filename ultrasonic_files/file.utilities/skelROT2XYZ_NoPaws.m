function [skXYZ] = skelROT2XYZ_NoPaws(skROT,limbLengths)
% Thomas S. Murray
% 2014/11/11
%
% This version of the code removes hands and feet (i.e. no paws) 2014/12/10
%
% Input:
%   skROT - 3D skeleton in limb rotations augmented with hip translation [numLimbs+1 x 3]
%   limbLengthsgths - length of each limb segment
%
% Output:
%   skXYZ - 3D skeleton in cartesian coordinates [numJoints x 3]

  skROT_aug = [skROT(2:end,1) skROT(2:end,2) zeros(15,1) skROT(2:end,3)];

  skXYZ( 1,:) = skROT( 1,:);                                                           % hip center
  skXYZ( 2,:) = skXYZ( 1,:) + (limbLengths( 1)*[0 0 1]*vrrotvec2mat(skROT_aug( 1,:))); % spine
  skXYZ( 3,:) = skXYZ( 2,:) + (limbLengths( 2)*[0 0 1]*vrrotvec2mat(skROT_aug( 2,:))); % shoulder center
  skXYZ( 4,:) = skXYZ( 3,:) + (limbLengths( 3)*[0 0 1]*vrrotvec2mat(skROT_aug( 3,:))); % head
  skXYZ( 5,:) = skXYZ( 3,:) + (limbLengths( 4)*[0 0 1]*vrrotvec2mat(skROT_aug( 4,:))); % left shoulder
  skXYZ( 6,:) = skXYZ( 5,:) + (limbLengths( 5)*[0 0 1]*vrrotvec2mat(skROT_aug( 5,:))); % left elbow
  skXYZ( 7,:) = skXYZ( 6,:) + (limbLengths( 6)*[0 0 1]*vrrotvec2mat(skROT_aug( 6,:))); % left wrist  
  skXYZ( 8,:) = skXYZ( 3,:) + (limbLengths( 7)*[0 0 1]*vrrotvec2mat(skROT_aug( 7,:))); % right shoulder
  skXYZ( 9,:) = skXYZ( 8,:) + (limbLengths( 8)*[0 0 1]*vrrotvec2mat(skROT_aug( 8,:))); % right elbow
  skXYZ(10,:) = skXYZ( 9,:) + (limbLengths( 9)*[0 0 1]*vrrotvec2mat(skROT_aug( 9,:))); % right wrist  
  skXYZ(11,:) = skXYZ( 1,:) + (limbLengths(10)*[0 0 1]*vrrotvec2mat(skROT_aug(10,:))); % left hip
  skXYZ(12,:) = skXYZ(11,:) + (limbLengths(11)*[0 0 1]*vrrotvec2mat(skROT_aug(11,:))); % left knee
  skXYZ(13,:) = skXYZ(12,:) + (limbLengths(12)*[0 0 1]*vrrotvec2mat(skROT_aug(12,:))); % left ankle  
  skXYZ(14,:) = skXYZ( 1,:) + (limbLengths(13)*[0 0 1]*vrrotvec2mat(skROT_aug(13,:))); % right hip
  skXYZ(15,:) = skXYZ(14,:) + (limbLengths(14)*[0 0 1]*vrrotvec2mat(skROT_aug(14,:))); % right knee
  skXYZ(16,:) = skXYZ(15,:) + (limbLengths(15)*[0 0 1]*vrrotvec2mat(skROT_aug(15,:))); % right ankle
  
return;

%   scaling_coefs = [0 0 1]' * limbLengthsgths';
%   
%   skXYZ( 1,:) = skROT( 1,:);  % hip center, used to translate skeleton
%   skXYZ( 2,:) = skXYZ( 1,:) + (vrrotvec2mat(skROT_aug( 2,:))*scaling_coefs(:, 1))'; % Spine = 2, not used
%   skXYZ( 3,:) = skXYZ( 2,:) + (vrrotvec2mat(skROT_aug( 3,:))*scaling_coefs(:, 2))'; % Shoulder_Center = 3;
%   skXYZ( 4,:) = skXYZ( 3,:) + (vrrotvec2mat(skROT_aug( 4,:))*scaling_coefs(:, 3))'; % Head = 4;
%   skXYZ( 5,:) = skXYZ( 3,:) + (vrrotvec2mat(skROT_aug( 5,:))*scaling_coefs(:, 4))'; % Shoulder_Left = 5;
%   skXYZ( 6,:) = skXYZ( 5,:) + (vrrotvec2mat(skROT_aug( 6,:))*scaling_coefs(:, 5))'; % Elbow_Left = 6;
%   skXYZ( 7,:) = skXYZ( 6,:) + (vrrotvec2mat(skROT_aug( 7,:))*scaling_coefs(:, 6))'; % Wrist_Left = 7;
%   skXYZ( 8,:) = skXYZ( 7,:) + (vrrotvec2mat(skROT_aug( 8,:))*scaling_coefs(:, 7))'; % Hand_Left = 8;
%   skXYZ( 9,:) = skXYZ( 3,:) + (vrrotvec2mat(skROT_aug( 9,:))*scaling_coefs(:, 8))'; % Shoulder_Right = 9;
%   skXYZ(10,:) = skXYZ( 9,:) + (vrrotvec2mat(skROT_aug(10,:))*scaling_coefs(:, 9))'; % Elbow_Right = 10;
%   skXYZ(11,:) = skXYZ(10,:) + (vrrotvec2mat(skROT_aug(11,:))*scaling_coefs(:,10))'; % Wrist_Right = 11;
%   skXYZ(12,:) = skXYZ(11,:) + (vrrotvec2mat(skROT_aug(12,:))*scaling_coefs(:,11))'; % Hand_Right = 12;
%   skXYZ(13,:) = skXYZ( 1,:) + (vrrotvec2mat(skROT_aug(17,:))*scaling_coefs(:,16))'; % Hip_Left = 13;
%   skXYZ(14,:) = skXYZ(13,:) + (vrrotvec2mat(skROT_aug(18,:))*scaling_coefs(:,17))'; % Knee_Left = 14;
%   skXYZ(15,:) = skXYZ(14,:) + (vrrotvec2mat(skROT_aug(19,:))*scaling_coefs(:,18))'; % Ankle_Left = 15;
%   skXYZ(16,:) = skXYZ(15,:) + (vrrotvec2mat(skROT_aug(20,:))*scaling_coefs(:,19))'; % Foot_Left = 16; 
%   skXYZ(17,:) = skXYZ( 1,:) + (vrrotvec2mat(skROT_aug(13,:))*scaling_coefs(:,12))'; % Hip_Right = 17;
%   skXYZ(18,:) = skXYZ(17,:) + (vrrotvec2mat(skROT_aug(14,:))*scaling_coefs(:,13))'; % Knee_Right = 18;
%   skXYZ(19,:) = skXYZ(18,:) + (vrrotvec2mat(skROT_aug(15,:))*scaling_coefs(:,14))'; % Ankle_Right = 19;
%   skXYZ(20,:) = skXYZ(19,:) + (vrrotvec2mat(skROT_aug(16,:))*scaling_coefs(:,15))'; % Foot_Right = 20;