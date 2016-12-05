clear all
close all
clc


% directory to save measurement mask to
directoryPathMeasurement = 'D:\Projects\MATLAB Projects\Structured Light Compressive Sensing\data\1920x1080 Patterns\Measurement Masks\';
% directoryPathCalibration = 'D:\Projects\MATLAB Projects\Structured Light Compressive Sensing\data\1920x1080 Patterns\Calibration Masks\';

% projector resolution
projector.hResolution=1920;
projector.vResolution=1080;

% plot masks bool
plotBool=0;


generateMeasurementMasks(directoryPathMeasurement, projector.hResolution, projector.vResolution, plotBool);

% cd 'D:\Projects\MATLAB Projects\Structured Light Compressive Sensing\utilities'

% generateCalibrationMasks(directoryPathCalibration, projector.hResolution, projector.vResolution, plotBool);