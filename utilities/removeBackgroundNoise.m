function measurements_without_background=removeBackgrounNoise(measurements, average_background_noise, plot)

% function removes estimated average background noise from input images

for i=1:length(measurements)    
    measurements_without_background{i}=measurements{i}-average_background_noise;
end