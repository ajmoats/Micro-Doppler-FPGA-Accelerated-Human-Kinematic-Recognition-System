function [limbs] = getSkeletonConnectionMap()
% Thomas S. Murray
% 2013/08/31
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

limbs = [[1 2];     % Spine: hip center to spine
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
                       
end