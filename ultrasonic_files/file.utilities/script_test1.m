clc;
clear all;
close all;

% tic
% [catalog] = generateCatalog('..\test_usx2_kinect_20131105\');
% toc

load('..\test_usx2_kinect_20131105\catalog.mat');

% plot(catalog.dacqStamps,catalog.dacqStamps,'bo',...
%      catalog.kinectStamps,catalog.kinectStamps,'r.');
   
visualizeKinectAndUltrasound(catalog);
