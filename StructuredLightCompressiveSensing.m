clear all
close all
clc

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
    
    bounding_box{phase_no}=detectMaskBoundingRectangles(image{phase_no}, average_background_noise);
end

%% LOAD SYNTHETIC MASKS - REAL MEASUREMENTS MASKS AND CALIBRATION MASKS
% crop to only one synthetic submask - loaded images contain multiple masks
crop_masks.bool=1;
crop_masks.roi_x_start=0;
crop_masks.roi_y_start=0;
crop_masks.block_size=8;

% load synth measurement masks and produce measurement matrix phi and
% calculate number of ones(sum of ones in each mask should be 32)

[synth_masks, synth_mask_number_of_ones, phi]=loadSyntheticMasks('D:\Diplomski rad\1280x800 Patterns\Measurement Masks\1\','.png', crop_masks, 1);

% load calib masks and calculate number of ones in each mask
[synth_calib_masks, synth_calib_mask_number_of_ones]=loadSyntheticMasks('D:\Diplomski rad\1280x800 Patterns\Different Percentage Masks\1. Random Pattern\1\','.png', crop_masks, 0);


%% CROP REAL IMAGES BY PHASE ROI

% variable that holds sum of all 4 phases in our ROI - image reconstruction
summed_measurements_image=0;

parfor phase_no=1:no_of_phases
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
        
        % psi matrix generation - depending in which domain problem is sparse, use
        % corresponding transformation matrix
        
%         N=64;
%         
%         DCTm = dctmtx(N);
%         IDCTm = dctmtx(N)';
%         
%         psi=DCTm;
%         psi_inv=IDCTm;


        % number of measurements used in image reconstruction
        no_of_measurements_for_reconstruction=25;
        
        phi_r=phi(1:no_of_measurements_for_reconstruction,:);
        
        % defining matrix theta y=theta*x
        theta = full(phi_r*psi_inv); % Phi_m * Psi^(-1)
        
        % standard dual form: data conditioning for minimum L1 - SeDuMi algorithm
        % input data preparation for the problem
        
        
        %%
        
%         [M,N]=size(theta);
%         
%         b = [ spalloc(N,1,0); -sparse(ones(N,1)) ];
%         
%         At = [ -sparse(theta)   ,     spalloc(M,N,0)    ;...
%             sparse(theta)   ,     spalloc(M,N,0)    ;...
%             speye(N)        ,    -speye(N)          ;...
%             -speye(N)        ,    -speye(N)          ;...
%             spalloc(N,N,0)  ,    -speye(N)          ];
%         
%         % SEDUMI OPTIMIZATION
%         
%         for phase_no=1:no_of_phases
%         
%                 % Standard dual form: data conditioning for minimum L1
%         
%                 c = [ -sparse(y{phase_no}(:)); sparse(y{phase_no}(:)); spalloc(3*N,1,0) ];
%         
%                 % Optimization
%                 tic, [~,s]=sedumi(At, b, c); toc % SeDuMi
%         
%                 % Output data processing
%                 s=s(:);
%                 s=s(1:N);
%         
%                 yr{phase_no} = psi_inv * s;
%         
%                 yr{phase_no}= reshape(yr{phase_no},8,8);
%         end
%         
%         image_reconstruction=[yr{1} yr{2}; yr{3} yr{4}];
%         
%         figure, imshow(image_reconstruction), colormap gray, title('Reconstruction'), axis image
%         
        
        %% CVX SOLVER - alternative to SeDuMi
        
        image_est = [];
        
        % Reconstructing initial signal
        cvx_solver sedumi
        
        cvx_begin
        cvx_precision high
        
        variable s_est(64, 1);
        minimize(norm(s_est, 1));
        subject to
        theta * s_est == y(1:no_of_measurements_for_reconstruction)';
        
        cvx_end
        
        image_est = (psi_inv * s_est).';
        %     image_est = idct2(s_est);
        
        im_gray_est{phase_no}{bbox_no} = (reshape(image_est, 8, 8));
        im_gray_est_vector{phase_no}{bbox_no} = image_est;
        %         figure(101), imagesc(im_gray_est{phase_no}{bbox_no}), colormap gray
 
        %         image_gray{phase_no}=[im_gray_est{phase_no}{1}, im_gray_est{}]

    end
end

% plot whole scene reconstruction
figure, imagesc(summed_measurements_image), colormap gray, title('Whole Image'), axis image


%%

zero_matrix=zeros(8,8);
scene_reconstruction=zeros(144,144);

for bbox_no=1:81
    phase{1}{bbox_no}=[im_gray_est{1}{bbox_no}, zero_matrix; zero_matrix, zero_matrix];
    phase{2}{bbox_no}=[zero_matrix, im_gray_est{2}{bbox_no}; zero_matrix, zero_matrix];
    phase{3}{bbox_no}=[zero_matrix, zero_matrix; im_gray_est{3}{bbox_no}, zero_matrix];
    phase{4}{bbox_no}=[zero_matrix, zero_matrix; zero_matrix, im_gray_est{4}{bbox_no}];
    
end


phase_vector=(cell2mat(phase{1})+cell2mat(phase{2})+cell2mat(phase{3})+cell2mat(phase{4}))';
phase_vector=(phase_vector);

% phase_vector=phase_vector';

figure
imshow(phase_vector)

rSize=16;
cSize=16;

newMat=[];

for r=1:9
    subRow=[];
    for c=1:9
        subRow=[subRow phase_vector(1:rSize,:)];
        phase_vector(1:rSize,:)=[];
    end
    newMat=[newMat; subRow];
end

figure
imshow(newMat')

%%

 bzvz=(imresize(summed_measurements_image,[400 400]));


fun = @(block_struct) mean2(block_struct.data);

I2 = blockproc(bzvz,[8 8],fun);

figure
imagesc(I2)
colormap gray
