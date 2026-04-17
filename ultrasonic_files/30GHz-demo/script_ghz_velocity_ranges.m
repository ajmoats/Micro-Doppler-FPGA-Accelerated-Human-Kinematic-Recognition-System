% Thomas S. Murray
% 2015/01/19
%
% Adapted by Alexis Moats
% 2026/04/17
% This script computes the velocity ranges corresponding to the frequency
% ranges used in the GHz processing.

clc;
clear all;
close all;

% speed of sound
c = 340;  % m/s

% 30Ghz parameters
ghz.fc     = 25E3;
ghz.f_min  = ghz.fc - 1E3;
ghz.f_max  = ghz.fc + 1E3;
ghz.v_min  = c/2*(ghz.f_min/ghz.fc-1);
ghz.v_max  = c/2*(ghz.f_max/ghz.fc-1);
