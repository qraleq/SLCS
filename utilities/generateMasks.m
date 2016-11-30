clear all
close all
clc


%
directoryPath='D:\Projects\MATLAB Projects\Structured Light Compressive Sensing\data\1920x1080 Patterns\Measurement Masks\'
% projector resolution
projector.hResolution=1920;
projector.vResolution=1080;

% set number of masks to be generated, mask size and frequency of white
% pixels in a mask
noOfMasks=64;
mask_size=8;
plotBool=1;


generateMeasurementMasks(directoryPath, projector.hResolution, projector.vResolution, noOfMasks, plotBool);
