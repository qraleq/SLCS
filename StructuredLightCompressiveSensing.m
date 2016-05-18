clear all
close all
clc

tic

% data defines path to input images needed for SLCS reconstruction and
% utilities defines path to misc functions used in SLCS algorithm
addpath('data');
addpath('utilities')

%% GLOBAL VARIABLES DEFINITION & MISC
% no_of_phases is a variable showing us which of 4 phases of one measurement is
% being considered and manipulated with
% phase order is [1 2; 3 4] in Matlab notation

% no_of_calib_measurements - 4 different measurements
% are conducted to minimize error in gamma
% distortion estimation - we calculate their mean value
no_of_phases=4;
no_of_calib_measurements=4;

synth_mask_size=[8 8];
% global variable used to determine which of 4 Bayer CFA color is being
% used in further processing - you can choose between:
% R - only red channel
% G - both green channels from Bayer CFA
% G1 or G2 - one of green channels in Bayer CFA
% B - only blue channel
bayer_cfa_color='g';

% if plot_images is true, images loaded by bulk load functions are being ploted
plot_images.bool=0;

% generate transformation matrix psi - choice between dct and wav
% wavelet type - 'haar', 'db2', 'db4', 'sym4', 'sym8', ...

[psi, psi_inv]=generateMatrixPsi('dct',[]);

%% SET CROP ROI ON IMAGE
% determine region of interest(roi) on scene image and plot it

% loading whole(non-cropped) real measurement image to decide what is our ROI
crop_dummy.bool=0;
measurement_image_whole=imreadraw('D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\Measurements\1\mask_064.dng', crop_dummy, 'all');

% plot whole real measurement image
figure, imagesc(measurement_image_whole), colormap gray, title('Whole Measurement Image'), axis image
hold on

% defining crop_roi struct containing details about crop roi
crop_roi.bool=1;
crop_roi.roi_x_start=1960;
crop_roi.roi_y_start=1045;
crop_roi.block_size=399;

% draw rectangle on desired image roi
rectangle('Position', [crop_roi.roi_x_start, crop_roi.roi_y_start, crop_roi.block_size, crop_roi.block_size],'EdgeColor', 'r', 'LineWidth', 2);

%% LOAD BACKGROUND IMAGES

[backgrounds, meta_info_backgrounds] = imreadraw_from_directory('D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\Backgrounds\','.dng', crop_roi, bayer_cfa_color, plot_images.bool);

%% ESTIMATE AVERAGE BACKGROUND NOISE IMAGE

average_background_noise = estimateAverageBackgroundNoise(backgrounds, plot_images.bool);

clear backgrounds meta_info_backgrounds
%% LOAD CALIBRATION MEASUREMENTS
% calibration measurements are different percentage masks images 
% used to estimate gamma distortion in camera-projector system

% for phase_no=1:no_of_phases
%     for no_measurement=1:no_of_calib_measurements
%
%     [calib_measurements{phase_no}{no_measurement}, meta_info_calib_measurements{phase_no}{no_measurement}]=imreadraw_from_directory(['D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\',num2str(no_measurement),'. Different Percentage Mask\',num2str(phase_no),'\'],'.dng', crop_roi, bayer_color, plot_images.bool);
%
%     % removes background noise from calibration measurements
%     calib_measurements{phase_no}{no_measurement}=removeBackgroundNoise(calib_measurements{phase_no}{no_measurement}, backgrounds_avg);
%
%     % writes current progress into console - loading images may take some
%     % time to process
% 
%     phase_no, no_measurement
% 
%     end
% end
%
% 
% % save cablib_measurements variable because its loading time is too long
% % save('calib_measurements','calib_measurements')

load calib_measurements

%% LOAD REAL MEASUREMENTS
% real measurements are done by projecting 50% binary masks onto a real
% scene
% background noise is removed from real measurements just as in case of
% calibration measurements

% for phase_no=1:4
%     [measurements{phase_no}, meta_info_measurements{phase_no}]=imreadraw_from_directory(['D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\Measurements\',num2str(phase_no),'\'],'.dng', crop_roi, bayer_color, plot_images.bool);
%     measurements{phase_no}=removeBackgroundNoise(measurements{phase_no}, backgrounds_avg);
% end
%
% save measurements variable because its loading time is too long
% save('measurements', 'measurements')

load measurements

%% BLOB DETECTION AND PROCESSING
% using morphological operations and blob detection algorithm to detect
% bounding boxes of each and every measurement mask in image

for phase_no=1:no_of_phases
    % load calibration image with all pixels in 8x8 square switched on for
    % easier bounding rectangle detection and easier crop
    image{phase_no}=im2double(calib_measurements{phase_no}{1}{64}-average_background_noise);
    
    [bounding_box{phase_no}, blocks_no_x, blocks_no_y]=detectMaskBoundingRectangles(image{phase_no}, average_background_noise);
end

%% LOAD SYNTHETIC MASKS - REAL MEASUREMENTS MASKS AND CALIBRATION MASKS
% crop to only one synthetic submask - loaded images contain multiple masks
crop_masks.bool=1;
crop_masks.roi_x_start=0;
crop_masks.roi_y_start=0;
crop_masks.block_size=synth_mask_size(1);

% load synth measurement masks and produce measurement matrix phi and
% calculate number of ones(sum of ones in each mask should be 32)

[synth_masks, synth_mask_number_of_ones, phi]=loadSyntheticMasks('D:\Diplomski rad\1280x800 Patterns\Measurement Masks\1\','.png', crop_masks, 1);

% load calib masks and calculate number of ones in each mask
[synth_calib_masks, synth_calib_mask_number_of_ones]=loadSyntheticMasks('D:\Diplomski rad\1280x800 Patterns\Different Percentage Masks\1. Random Pattern\1\','.png', crop_masks, 0);


%% CROP REAL IMAGES BY PHASE ROI

% variable that holds sum of all 4 phases in our ROI - image reconstruction
summed_measurements_image=0;

for phase_no=1:no_of_phases
    for bbox_no=1:size(bounding_box{phase_no},1)
        
        % define active block for processing using ROIs detected by blob
        % detection algorithm - crop real measurements
        crop{phase_no}.roi_x_start=bounding_box{phase_no}(bbox_no,1);
        crop{phase_no}.roi_y_start=bounding_box{phase_no}(bbox_no,2);
        crop{phase_no}.block_size_x=bounding_box{phase_no}(bbox_no,3);
        crop{phase_no}.block_size_y=bounding_box{phase_no}(bbox_no,4);
                
        for p=1:64
            measurements_block{p}=imcrop(measurements{phase_no}{p}, [crop{phase_no}.roi_x_start crop{phase_no}.roi_y_start crop{phase_no}.block_size_x crop{phase_no}.block_size_y]);
            
            % estimate treshold_value for leftover noise after background
            % subtraction using wavelet transformation
            [a,d,v,h]=dwt2(measurements_block{p}, 'haar');
            treshold_value=median(abs(d(:)))/0.6745;
            
            measurements_block{p}=wthresh(measurements_block{p}, 'h', 4*treshold_value);
            
        end
        
        % measurement value calculation by summing measurement image pixel
        % values; this way we get input for compressive sensing
        % reconstruction
        summed_measurements_subimage{phase_no}=0;
        
        for mask_number=1:64
            measurement_value(mask_number)=0;
            
            measurement_value(mask_number)=sum(measurements_block{mask_number}(:));
            
            summed_measurements_subimage{phase_no}=summed_measurements_subimage{phase_no}+measurements{phase_no}{mask_number};
        end
        
        % plot current block being processed
        figure(207)
        imagesc(summed_measurements_subimage{phase_no}), colormap gray, title(['Measurement Images Sum ', num2str(phase_no)])
        hold on
        rectangle('Position', [crop{phase_no}.roi_x_start, crop{phase_no}.roi_y_start, crop{phase_no}.block_size_x, crop{phase_no}.block_size_y],'EdgeColor', 'b', 'LineWidth', 3);
        drawnow
        
        % image_whole is full scene reconstruction by summing all four phases
        summed_measurements_image=summed_measurements_image+summed_measurements_subimage{phase_no};
               

        % calibration measurements processing for gamma distortion
        % correction
        calib_measurement_avg_value=0;
        
        % for each measurement and for 1-64 ones in a calib mask
        for no_measurements=1:no_of_calib_measurements
            for p=1:64
                
                calib_measurement_sum{no_measurements}(p)=0;

                calib_measurements_crop{no_measurements}{p}=imcrop(calib_measurements{phase_no}{no_measurements}{p}, [crop{phase_no}.roi_x_start crop{phase_no}.roi_y_start crop{phase_no}.block_size_x crop{phase_no}.block_size_y]);
                
                % leftover noise tresholding
                [a,d,v,h]=dwt2(calib_measurements_crop{no_measurements}{p}, 'haar');
                treshold_value=median(abs(d(:)))/0.6745;
                
                calib_measurements_crop{no_measurements}{p}=wthresh(calib_measurements_crop{no_measurements}{p}, 'h', 4*treshold_value);
                
                calib_measurement_sum{no_measurements}(p)=sum(calib_measurements_crop{no_measurements}{p}(:));
                
            end
            
            % calculate average of all 4 measurements for single calib mask
            % with certain percentage of ones to improve gamma distortion
            % estimation
            calib_measurement_avg_value=calib_measurement_avg_value+calib_measurement_sum{no_measurements}/4;            
        end
        
        %% GAMMA CORRECTION
        % camera-projector system has some kind of gamma distortion function and in
        % this part we calculate degamma function and gamma correct the
        % measurements
                
        % downsample factor is used to reduce number of calibration
        % measurements used in regresion
        downsample_factor=1;
        
        gamma_function=polyfit(log(synth_calib_mask_number_of_ones(1:downsample_factor:end)),log(calib_measurement_avg_value(1:downsample_factor:end)), 1);
        
        gamma=gamma_function(1);
        A=exp(gamma_function(2));
        
        calib_measurement_avg_value_regresion=A.*(synth_calib_mask_number_of_ones.^gamma);
        
        % plot gamma correction function
        figure(109)
        plot(synth_calib_mask_number_of_ones(1:downsample_factor:end)', [calib_measurement_avg_value(1:downsample_factor:end)' calib_measurement_avg_value_regresion(1:downsample_factor:end)'])
        title('Gamma Correction Function - model and real')
        
        xlabel('Number of Ones In A Mask')
        ylabel('Intensity Sum')
        
        inv_gamma_function=polyfit(log(calib_measurement_avg_value(1:downsample_factor:end)),log(synth_calib_mask_number_of_ones(1:downsample_factor:end)), 1);
        
        lambda=inv_gamma_function(1);
        B=exp(inv_gamma_function(2));
              
        
        synth_calib_mask_number_of_ones_inv=B*(calib_measurement_avg_value.^lambda);
        
        % plot inverse gamma correction funcrion
        figure(110)
        
        title('Inverse Gamma Correction Function - model')
        plot(calib_measurement_avg_value(1:downsample_factor:end)', synth_calib_mask_number_of_ones_inv(1:downsample_factor:end)')
        
        xlabel('Intensity Sum')
        ylabel('Number of Ones In A Mask')
        
        % degamma measurement
        y=B*(measurement_value.^lambda);
         
        %% COMPRESSIVE SENSING
        
        % number of measurements used in image reconstruction
        no_of_measurements_for_reconstruction=32;
        
        phi_r=phi(1:no_of_measurements_for_reconstruction,:);
        
        % defining matrix theta y=theta*x
        theta = full(phi_r*psi_inv); % Phi_m * Psi^(-1)
        
        subimage_estimation = L1OptimizationCVX(y, psi, psi_inv, theta, no_of_measurements_for_reconstruction);
%         im_gray_est = L1OptimizationSeDuMi(y, theta, no_of_measurements_for_reconstruction);
        
        subimage_estimations{phase_no}{bbox_no} = (reshape(subimage_estimation, 8, 8));

    end
end
%%
% plot whole scene reconstruction
figure, imagesc(summed_measurements_image), colormap gray, title('Whole Image'), axis image

% reshape reconstructed subimages to reconstructed image
reconstructed_image=subimagesToImageReshape(subimage_estimations, synth_mask_size);

figure, imshow(reconstructed_image), colormap gray, title('Reconstruction'), axis image
%% MEASUREMENT VISUALIZATION

mv=(imresize(summed_measurements_image,[144 144]));

fun = @(block_struct) mean2(block_struct.data);

measurement_visualization = blockproc(mv,[8 8],fun);
measurement_visualization=(imresize(measurement_visualization,8,'nearest'));
figure, imagesc(measurement_visualization), colormap gray, axis image, title('Measurement Image')


toc
