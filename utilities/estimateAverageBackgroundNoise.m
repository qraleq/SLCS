function [average_background_noise]  = estimateAverageBackgroundNoise(backgrounds, plot_bool)

%% CALCULATE AVERAGE BACKGROUND VALUE IMAGE

% average background contains all noise that occured in measurement
% process, and it includes different noises from camera sensor itself,
% projector 'black' image noise and other types of noise

average_background_noise=zeros(size(backgrounds{1},1), size(backgrounds{1},2));

% calculate backgrounds average
for i=1:length(backgrounds)
    average_background_noise=average_background_noise+((backgrounds{i})/length(backgrounds));
end

% plot backgrounds average
if(plot_bool)
    figure, imagesc(average_background_noise), colormap gray, title('Backgrounds Average')
end