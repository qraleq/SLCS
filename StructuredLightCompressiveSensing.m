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

%% SET CROP ROI ON IMAGE
% determine region of interest(roi) on scene image and plot it

% loading whole(non-cropped) real measurement image to decide what is our ROI
crop_dummy.bool=0;
measurement_image_whole=imreadraw('D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\Measurements\1\mask_064.dng', crop_dummy, 'all');

% plot whole real measurement image
figure, imagesc(measurement_image_whole), colormap gray, title('Whole Measurement Image'), axis image
hold on

% defining crop_roi struct containing details about crop roi
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

% for no_phase=1:no_of_phases
%     for no_measurement=1:no_of_calib_measurements
%
%     [calib_measurements{no_phase}{no_measurement}, meta_info_calib_measurements{no_phase}{no_measurement}]=imreadraw_from_directory(['D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\',num2str(no_measurement),'. Different Percentage Mask\',num2str(no_phase),'\'],'.dng', crop_roi, bayer_color, plot_images.bool);
%
%     % removes background noise from calibration measurements
%     calib_measurements{no_phase}{no_measurement}=remove_background_noise(calib_measurements{no_phase}{no_measurement}, backgrounds_avg);
%
%     % writes current progress into console - loading images may take some
%     % time to process
% 
%     no_phase, no_measurement
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

% for no_phase=1:4
%     [measurements{no_phase}, meta_info_measurements{no_phase}]=imreadraw_from_directory(['D:\Diplomski rad\Shootings\Shooting - 5.5. - FER - dng\Measurements\',num2str(no_phase),'\'],'.dng', crop_roi, bayer_color, plot_images.bool);
%     measurements{no_phase}=remove_background_noise(measurements{no_phase}, backgrounds_avg);
% end
%
% save measurements variable because its loading time is too long
% save('measurements', 'measurements')

load measurements

%% BLOB DETECTION AND PROCESSING

% variable that holds sum of all 4 phases in our ROI - image reconstruction
image_whole=0;

% for each phase do:
for no_phase=1:no_of_phases
    % load calibration image with all pixels in 8x8 square switched on -
    % easier crop
    image{no_phase}=im2double(calib_measurements{no_phase}{1}{64}-average_background_noise);
    
    % use morphological operations on "white" calibration image so whole
    % ROI is covered by blob detection algorithm
    se_dilate=strel('square', 8);  
    image_dilate{no_phase}=imdilate(image{no_phase}, se_dilate);
    
%     se_erode=strel('square', 5);
%     image_dilate{no_phase}=imerode(image_dilate{no_phase}, se_erode);
    
    % convert image to bw by tresholding everything above 2*mean value of
    % background noise
    bw_image_dilate{no_phase}=image_dilate{no_phase}>2*mean2(average_background_noise);
    
    % detect blobs/regions algorithm
    blobDetectorStruct = regionprops(bw_image_dilate{no_phase}, 'BoundingBox', 'Extrema', 'Centroid', 'Area', 'Orientation');
    
    % this part of code does logical sorting of blocks detected by blob
    % detection algorithm
    extrema = cat(1, blobDetectorStruct.Extrema);

    left_most_bottom = extrema(7:8:end, :);
    left = left_most_bottom(:, 1);
    bottom = left_most_bottom(:, 2);
    % quantize the bottom coordinate
    bottom = 3 * round(bottom / 3);
    
    [sorted, sort_order] = sortrows([bottom left]);
    
    sorted_region_props = blobDetectorStruct(sort_order);
    
    bounding_box=cat(1, sorted_region_props.BoundingBox);
    area=cat(1, sorted_region_props.Area);
    centroid=cat(1, sorted_region_props.Centroid);
    
    % filter detected blobs by minimum area treshold
    average_area=mean(area);
    sorted_region_props_filtered=[bounding_box, area, centroid];
    
    % ignore non-significant variations in detected parameters
    sorted_region_props_filtered=round(sorted_region_props_filtered, 3, 'significant');
    
    indices = find(area<average_area);
    sorted_region_props_filtered(indices,:) = [];
    centroid(indices,:)=[];
    
    % sort region props by centroid values
    [sorted, sort_order] = sortrows([centroid]);
    sorted_region_props_filtered = sortrows(sorted_region_props_filtered,[6,-7, 2]);
    
    figure
    I = im2uint8(bw_image_dilate{no_phase});
    imshow(I, 'InitialMag', 'fit')
    hold on
    
    % put bounding rectangles aroun ROIs and put numeric label on each
    % block in centroid location
    for k = 1:size(sorted_region_props_filtered,1)
        bbox{no_phase}(k,:)=[sorted_region_props_filtered(k,1), sorted_region_props_filtered(k,2),sorted_region_props_filtered(k,3),sorted_region_props_filtered(k,4)];
        
        text(sorted_region_props_filtered(k,6), sorted_region_props_filtered(k,7), sprintf('%d', k), 'Color', 'b');
        rectangle('Position', [sorted_region_props_filtered(k,1), sorted_region_props_filtered(k,2),sorted_region_props_filtered(k,3),sorted_region_props_filtered(k,4)],'EdgeColor', 'r', 'LineWidth', 3);   
    end
    
    hold off
    
end

%% DRAW ALL 64 MEASUREMENTS FOR EACH PHASE IN ONE FIGURE

% for p=1:64
%     figure(200)
%     subplot(221), imagesc(measurements{1}{p}), colormap gray, title(['Phase 1 ', num2str(p)])
%     subplot(222), imagesc(measurements{2}{p}), colormap gray, title(['Phase 2 ', num2str(p)])
%     subplot(223), imagesc(measurements{3}{p}), colormap gray, title(['Phase 3 ', num2str(p)])
%     subplot(224), imagesc(measurements{4}{p}), colormap gray, title(['Phase 4 ', num2str(p)])
%     drawnow,
%     %     waitforbuttonpress
% end

%% LOAD SYNTHETIC MASKS - REAL MEASUREMENTS MASKS AND CALIBRATION MASKS
% crop to only one synthetic mask - loaded images contain multiple masks
crop_masks.bool=1;
crop_masks.roi_x_start=0;
crop_masks.roi_y_start=0;
crop_masks.block_size=8;

% load synth measurement masks and produce measurement matrix phi and
% calculate number of ones(sum of ones in each mask should be 32)

[synth_masks, synth_mask_number_of_ones, phi]=load_synthetic_masks('D:\Diplomski rad\1280x800 Patterns\Measurement Masks\1\','.png', crop_masks, 0);

% load calib masks and calculate number of ones in each mask
% (that is only used to check if everything is ok)
[synth_calib_masks, synth_calib_mask_number_of_ones]=load_synthetic_masks('D:\Diplomski rad\1280x800 Patterns\Different Percentage Masks\1. Random Pattern\1\','.png', crop_masks, 0);


%% CROP REAL IMAGES BY PHASE ROI

for no_phase=1:no_of_phases
    for bbox_no=1:size(bbox{no_phase},1)
        % define active block for processing - using ROIs detected by blob
        % detection algorithm - crop real measurements
        crop{no_phase}.roi_x_start=bbox{no_phase}(bbox_no,1);
        crop{no_phase}.roi_y_start=bbox{no_phase}(bbox_no,2);
        crop{no_phase}.block_size_x=bbox{no_phase}(bbox_no,3);
        crop{no_phase}.block_size_y=bbox{no_phase}(bbox_no,4);
                
        for p=1:64
            measurements_crop{no_phase}{p}=imcrop(measurements{no_phase}{p}, [crop{no_phase}.roi_x_start crop{no_phase}.roi_y_start crop{no_phase}.block_size_x crop{no_phase}.block_size_y]);
            
            % estimate treshold_value for leftover noise after background
            % subtraction using wavelet transformation
            [a,d,v,h]=dwt2(measurements_crop{no_phase}{p}, 'haar');
            treshold_value=median(abs(d(:)))/0.6745;
            
            measurements_crop{no_phase}{p}=wthresh(measurements_crop{no_phase}{p}, 'h', 4*treshold_value);
            
        end
        
        %% MEASUREMENT SUM CALCULATION - CALCULATING INPUT VALUES FOR CS RECONSTRUCTION
        
        measurements_sum_image{no_phase}=0;
        measurements_sum_image_crop{no_phase}=0;
        
        for mask_number=1:64
            measurements_sum{no_phase}(mask_number)=0;
            measurements_sum{no_phase}(mask_number)=sum(measurements_crop{no_phase}{mask_number}(:));
            
            measurements_sum_image{no_phase}=measurements_sum_image{no_phase}+measurements{no_phase}{mask_number};
            measurements_sum_image_crop{no_phase}=measurements_sum_image_crop{no_phase}+measurements_crop{no_phase}{mask_number};
        end
        
        % plot current block being processed
        figure(207)
        imagesc(measurements_sum_image{no_phase}), colormap gray, title(['Measurement Images Sum ', num2str(no_phase)])
        hold on
        rectangle('Position', [crop{no_phase}.roi_x_start, crop{no_phase}.roi_y_start, crop{no_phase}.block_size_x, crop{no_phase}.block_size_y],'EdgeColor', 'b', 'LineWidth', 3);
        drawnow
        
        % image_whole is full scene reconstruction by summing all four phases
        image_whole=image_whole+measurements_sum_image{no_phase};
               

        %% CROP CALIBRATION MEASUREMENTS AND CALCULATE CALIBRATION MEASUREMENTS SUM
        
        calib_measurement_avg{no_phase}=0;
        % for each measurement and for 1-64 ones in a calib mask
        for no_measurements=1:no_of_calib_measurements
            for p=1:64
                calib_measurement_sum_by_percentage{no_phase}{no_measurements}(p)=0;

                calib_measurements_crop{no_phase}{no_measurements}{p}=imcrop(calib_measurements{no_phase}{no_measurements}{p}, [crop{no_phase}.roi_x_start crop{no_phase}.roi_y_start crop{no_phase}.block_size_x crop{no_phase}.block_size_y]);
                
                % leftover noise tresholding
                [a,d,v,h]=dwt2(calib_measurements_crop{no_phase}{no_measurements}{p}, 'haar');
                treshold_value=median(abs(d(:)))/0.6745;
                
                calib_measurements_crop{no_phase}{no_measurements}{p}=wthresh(calib_measurements_crop{no_phase}{no_measurements}{p}, 'h', 4*treshold_value);
                
                calib_measurement_sum_by_percentage{no_phase}{no_measurements}(p)=sum(calib_measurements_crop{no_phase}{no_measurements}{p}(:));
                
            end
            % calculate average of all 4 measurements for single calib mask
            % with certain percentage of ones
            calib_measurement_avg{no_phase}=calib_measurement_avg{no_phase}+calib_measurement_sum_by_percentage{no_phase}{no_measurements}/4;
            
        end
        
        %% GAMMA CORRECTION
        % camera-projector system has some kind of gamma distortion function and in
        % this part we calculate degamma function and gamma correct the
        % measurements
                
        % downsample factor is used to reduce number of calibration
        % measurements used in regresion
        downsample_factor=1;
        
        gamma_function{no_phase}=polyfit(log(synth_calib_mask_number_of_ones(1:downsample_factor:end)),log(calib_measurement_avg{1}(1:downsample_factor:end)), 1);
        
        gamma{no_phase}=gamma_function{no_phase}(1);
        A{no_phase}=exp(gamma_function{no_phase}(2));
        
        calib_measurements_sum_averaged_regresion{no_phase}=A{no_phase}.*(synth_calib_mask_number_of_ones.^gamma{no_phase});
        
        %         figure
        %         plot(synth_calib_mask_number_of_ones(1:downsample_factor:end)', [calib_measurement_avg{no_phase}(1:downsample_factor:end)' calib_measurements_sum_averaged_regresion{no_phase}(1:downsample_factor:end)'])
        %         title('Gamma Correction Function - model and real')
        %
        %         xlabel('Number of Ones In A Mask')
        %         ylabel('Intensity Sum')
        
        inv_gamma_function{no_phase}=polyfit(log(calib_measurement_avg{no_phase}(1:downsample_factor:end)),log(synth_calib_mask_number_of_ones(1:downsample_factor:end)), 1);
        
        lambda{no_phase}=inv_gamma_function{no_phase}(1);
        B{no_phase}=exp(inv_gamma_function{no_phase}(2));
        
        synth_calib_mask_number_of_ones_inv{no_phase}=B{no_phase}*(calib_measurement_avg{no_phase}.^lambda{no_phase});
        
        %         figure
        %
        %         title('Inverse Gamma Correction Function - model')
        %         plot(calib_measurement_avg{no_phase}(1:downsample_factor:end)', synth_calib_mask_number_of_ones_inv{no_phase}(1:downsample_factor:end)')
        %
        %         xlabel('Intensity Sum')
        %         ylabel('Number of Ones In A Mask')
        
        % degamma measurement
        y{no_phase}=B{no_phase}*(measurements_sum{no_phase}.^lambda{no_phase});
        
        
        %% TRANSFORMATION MATRICES PSI GENERATION
        % in the CS problem, linear bases are usually defined by matrices Phi and Psi
        % Producing corresponding matrices of 2D linear transforms, typically given by MATLAB functions
        
        wavelet = 'haar'; % 'haar', 'db2', 'db4', 'sym4', 'sym8', ...
        
        im=zeros(8,8);
        
        [rows, cols] = size(im);
        % n = ceil(log2(min(rows,cols))); % maximum number of wavelet decomposition levels
        n = wmaxlev(size(im), wavelet); % maximum number of wavelet decomposition levels
        [C,S] = wavedec2(im, n, wavelet); % conversion to 2D, wavelet decomposition
        
        % DWT 2D matrix Psi
        % The matrix columns are unit impulse responses for each pixel
        i = 1;
        delta = zeros(rows*cols,1); % Store to 1D vector for simplicity
        delta(i) = 1; % Unit impulse at the first pixel position
        C = wavedec2(reshape(delta,[rows,cols]), n, wavelet); % conversion to 2D, wavelet decomposition
        DWTm = sparse(length(C), rows*cols); % space alocation for the result (sparse matrix)
        DWTm(:,i) = C.'; % first image column
        
        for i=2:rows*cols,
            delta(i-1)=0;
            delta(i) = 1; % Unit impulse at each pixel position
            C = wavedec2(reshape(delta,[rows,cols]), n, wavelet); % conversion to 2D, wavelet decomposition
            DWTm(:,i) = C.'; % all image columns
        end
        
        % Check the matrix construction
        % Y = (DWT * reshape(im, rows*cols,1)).'; % Aplication on an image: conversion to 1D, matrix multiplication
        % C = wavedec2(im, n, wavelet); % Direct implementation using MATLAB function
        % max(abs(Y-C))  % must be zero
        
        
        % IDWT 2D matrix (Psi^(-1))
        % The matrix columns are unit impulse responses for each spectrum coefficient
        i=1;
        clen = size(DWTm,1);
        delta = zeros(clen,1);
        delta(i) = 1; % Unit impulse at the first wavelet spectrum coefficient position
        xr = waverec2(delta, S, wavelet); % wavelet reconstruction (inverse transform)
        IDWTm = sparse(rows*cols, clen); % space alocation for the result (sparse matrix)
        IDWTm(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
        
        for i=2:clen,
            delta(i-1)=0;
            delta(i) = 1; % Unit impulse at each wavelet spectrum coefficient position
            xr = waverec2(delta, S, wavelet); % wavelet reconstruction (inverse transform)
            IDWTm(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
        end
        
        % Check the perfect reconstruction
        % full(max(max(abs(IDWT * DWT - speye(rows*cols)))))  % must be zero
        
        %% DCT MATRIX GENERATION
        % The matrix columns are unit impulse responses for each pixel
        
        i = 1;
        delta = zeros(rows*cols,1); % Store to 1D vector for simplicity
        delta(i) = 1; % Unit impulse at the first pixel position
        C = dct2(reshape(delta,[rows, cols])); % conversion to 2D, DCT
        DCTm = sparse(rows*cols, rows*cols); % space alocation for the result (sparse matrix)
        DCTm(:,i) = C(:); % first image column
        
        for i=2:rows*cols
            delta(i-1)=0;
            delta(i) = 1; % Unit impulse at each pixel position
            C = dct2(reshape(delta,[rows, cols])); % conversion to 2D, DCT
            DCTm(:,i) = C(:); % all image columns
        end
        
        % Check the matrix construction
        % x = randn(8,8);
        % Y = (DCT * reshape(x, 8*8,1)).'; % Aplication on an image: conversion to 1D, matrix multiplication
        % C = dct2(x); % Direct implementation using MATLAB function
        % max(abs(Y-C(:).'))  % must be zero
        
        %% IDCT 2D matrix 8 x 8
        % The matrix columns are unit impulse responses for each spectrum coefficient
        
        i=1;
        clen = size(DCTm,1);
        delta = zeros(clen,1);
        delta(i) = 1; % Unit impulse at the first wavelet spectrum coefficient position
        xr = idct2(reshape(delta,[rows,cols])); % idct (inverse transform)
        IDCTm = sparse(rows*cols, clen); % space alocation for the result (sparse matrix)
        IDCTm(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
        
        for i=2:clen,
            delta(i-1)=0;
            delta(i) = 1; % Unit impulse at each DCT spectrum coefficient position
            xr = idct2(reshape(delta,[rows,cols])); % reconstruction (inverse transform)
            IDCTm(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
        end
        
        %% COMPRESSIVE SENSING
        
        % psi matrix generation - depending in which domain problem is sparse, use
        % corresponding transformation matrix
        
        N=64;
        
        DCTm = dctmtx(N);
        IDCTm = dctmtx(N)';
        
%         psi=DCTm;
%         psi_inv=IDCTm;
%         
        psi=DWTm;
        psi_inv=IDWTm;

        % number of measurements used in image reconstruction
        no_of_measurements_for_reconstruction=50;
        
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
%         for no_phase=1:no_of_phases
%         
%                 % Standard dual form: data conditioning for minimum L1
%         
%                 c = [ -sparse(y{no_phase}(:)); sparse(y{no_phase}(:)); spalloc(3*N,1,0) ];
%         
%                 % Optimization
%                 tic, [~,s]=sedumi(At, b, c); toc % SeDuMi
%         
%                 % Output data processing
%                 s=s(:);
%                 s=s(1:N);
%         
%                 yr{no_phase} = psi_inv * s;
%         
%                 yr{no_phase}= reshape(yr{no_phase},8,8);
%         end
%         
%         image_reconstruction=[yr{1} yr{2}; yr{3} yr{4}];
%         
%         figure, imshow(image_reconstruction), colormap gray, title('Reconstruction'), axis image
%         
        
        %% CVX SOLVER - alternative to SeDuMi
        
        clear s_est
        clear image_est
        image_est = [];
        
        % Reconstructing initial signal
        cvx_solver sedumi
        
        cvx_begin
        cvx_precision high
        
        variable s_est(64, 1);
        minimize(norm(s_est, 1));
        subject to
        theta * s_est == y{no_phase}(1:no_of_measurements_for_reconstruction)';
        
        cvx_end
        
        image_est = (psi_inv * s_est).';
        %     image_est = idct2(s_est);
        
        im_gray_est{no_phase}{bbox_no} = (reshape(image_est, 8, 8));
        im_gray_est_vector{no_phase}{bbox_no} = image_est;
        %         figure(101), imagesc(im_gray_est{no_phase}{bbox_no}), colormap gray
 
        %         image_gray{no_phase}=[im_gray_est{no_phase}{1}, im_gray_est{}]

    end
end

% plot whole scene reconstruction
figure, imagesc(image_whole), colormap gray, title('Whole Image'), axis image


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
phase_vector=flipud(phase_vector);

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
imshow(fliplr((newMat')))

%%

 bzvz=(imresize(image_whole,[144 144]));


fun = @(block_struct) mean2(block_struct.data);

I2 = blockproc(bzvz,[8 8],fun);

figure
imagesc(I2)
colormap gray
axis image