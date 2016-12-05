close all
clear all
clc

addpath('utilities\')

directoryPath='D:\Measurements\Misc White\';

cropStruct.bool=1;
cropStruct.rect=[2930 1473 200 70];

images=readImagesFromDirectory(directoryPath, '.pgm', cropStruct);

