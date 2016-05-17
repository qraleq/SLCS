function measurements_without_background=remove_background_noise(measurements, backgrounds_avg, plot)

for i=1:length(measurements)
    
    measurements_without_background{i}=measurements{i}-backgrounds_avg;
    
end