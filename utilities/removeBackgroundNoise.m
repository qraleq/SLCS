function measurements_without_background=removeBackgrounNoise(measurements, backgrounds_avg, plot)

% function removes estimated average background noise from input images

for i=1:length(measurements)    
    measurements_without_background{i}=measurements{i}-backgrounds_avg;
end