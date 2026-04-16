function [limbs] = getSkeletonConnectionMap_NoPaws()
% Thomas S. Murray
% 2013/08/31
%
% This version of the code removes hands and feet (i.e. no paws) 2014/12/10
%
% This function defines the mapping from the kinect skeletal joint order to
% the joint pairs that define skeltal limbs.
%
% Bradley 2010: Applications of Fresnel-Kirchhoff diffraction theory in the 
% analysis of human-motion Doppler sonar grams

% Kinect Joints:
% Hip_Center = 1;
% Spine = 2;
% Shoulder_Center = 3;
% Head = 4;
% Shoulder_Left = 5;
% Elbow_Left = 6;
% Wrist_Left = 7;
% Hand_Left = 8;
% Shoulder_Right = 9;
% Elbow_Right = 10;
% Wrist_Right = 11;
% Hand_Right = 12;
% Hip_Left = 13;
% Knee_Left = 14;
% Ankle_Left = 15;
% Foot_Left = 16; 
% Hip_Right = 17;
% Knee_Right = 18;
% Ankle_Right = 19;
% Foot_Right = 20;

% remove 8,12,16,20

% Hip_Center = 1;
% Spine = 2;
% Shoulder_Center = 3;
% Head = 4;
% Shoulder_Left = 5;
% Elbow_Left = 6;
% Wrist_Left = 7;
% Shoulder_Right = 8;
% Elbow_Right = 9;
% Wrist_Right = 10;
% Hip_Left = 11;
% Knee_Left = 12;
% Ankle_Left = 13;
% Hip_Right = 14;
% Knee_Right = 15;
% Ankle_Right = 16;

limbs = [[ 1  2];   % Spine: hip center to spine
         [ 2  3];   % spine to shoulder center
         [ 3  4];   % shoulder center to head
         [ 3  5];   % Left Hand: shoulder center to left shoulder
         [ 5  6];   % left shoulder to elbow
         [ 6  7];   % left elbow to wrist
         [ 3  8];   % Right Hand: shoulder center to right shoulder
         [ 8  9];   % right shoulder to elbow
         [ 9 10];   % right elbow to wrist
         [ 1 14];   % Right Leg: hip center to right hip
         [14 15];   % right hip to knee
         [15 16];   % right knee to ankle
         [ 1 11];   % Left Leg: hip center to left hip
         [11 12];   % left hip to knee
         [12 13]];  % left knee to ankle
                       
end