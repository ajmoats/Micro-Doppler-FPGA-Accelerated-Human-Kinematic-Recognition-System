function [segments] = getSegmentConnectionMap()
% Thomas S. Murray
% 2013/10/30
%
% This function defines the mapping to go from the kinect skeletal joint
% ordering to the 12 body segment model described in Bradley 2010.

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

% original segments:
segments = [[13 14];  % left thigh
            [17 18];  % right thigh
            [14 15];  % left lower leg
            [18 19];  % right lower leg
            [15 16];  % left foot
            [19 20];  % right foot
            [ 5  6];  % left upper arm
            [ 9 10];  % right upper arm
            [ 6  8];  % left forearm and hand
            [10 12];  % right forearm and hand
            [ 1  3];  % trunk
            [ 3  4]]; % head and neck

% % segments with more stable endpoints (wrists instead of hands and no feet)
% segments = [[13 14];  % left thigh
%             [17 18];  % right thigh
%             [14 15];  % left lower leg
%             [18 19];  % right lower leg
%             [ 5  6];  % left upper arm
%             [ 9 10];  % right upper arm
%             [ 6  7];  % left forearm and hand
%             [10 11];  % right forearm and hand
%             [ 1  3];  % trunk
%             [ 3  4]]; % head and neck
                       
end