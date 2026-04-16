% Thomas S. Murray
% 2015/01/19
%
% This script computes the velocity ranges corresponding to the frequency
% ranges used in the ultrasound processing.

clc;
clear all;
close all;

% speed of sound
c = 340;  % m/s

% us25 parameters
us25.fc     = 25E3;
us25.f_min  = us25.fc - 1E3;
us25.f_max  = us25.fc + 1E3;
us25.v_min  = c/2*(us25.f_min/us25.fc-1);
us25.v_max  = c/2*(us25.f_max/us25.fc-1);

% us33 parameters
us33.fc     = (33+1/3)*1E3;
us33.f_min  = us33.fc - 1E3;
us33.f_max  = us33.fc + 1E3;
us33.v_min  = c/2*(us33.f_min/us33.fc-1);
us33.v_max  = c/2*(us33.f_max/us33.fc-1);

% us40 parameters
us40.fc     = 40E3;
us40.f_min  = us40.fc - 1E3;
us40.f_max  = us40.fc + 1E3;
us40.v_min  = c/2*(us40.f_min/us40.fc-1);
us40.v_max  = c/2*(us40.f_max/us40.fc-1);