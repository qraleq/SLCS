%% D. Sersic, M. Vucic, October 2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compressive sensing (CS)
%
% Assumption of sparsity
% There is a linear base Psi in which observed phenomenon is sparse:
%    only K non-zero spectrum samples out of total N are non-zero, K << N
% Examples: Psi == DWT (discrete wavelet transform), DCT (discrete cosine transform),...
%           Many natural phenomena have sparse DWT, DCT, ... spectra
%
% Assumption of incoherence
% Observation of the phenomenon is conducted in linear base Phi that is incoherent with Psi
%
% CS hypothesis
% The observed phenomenon can be estimated in a statistically reliable way from
% only M >= K  << N observations by solving:
%
% min_L1 s
% subject to y_m = Phi_m * Psi^(-1) * s
%
% Phi_m contains only M rows of observation matrix Phi, where M << N
% (e.g only M observations of y are available)
% s - sparse representation of the observed phenomenon in linear base Psi:
%     observed phenomenon x = Psi^(-1) * s
% y - observations (measurements) in linear base Phi:
%     y = Phi * x

%% 2D example
clear all
close all
clc

image_est = [];

%% In the CS problem, linear bases are usually defined by matrices Phi and Psi
% Producing corresponding matrices of 2D linear transforms, typically given by MATLAB functions

wavelet = 'haar'; % 'haar', 'db2', 'db4', 'sym4', 'sym8', ...
r=8;
c=8;
dummy_wav=zeros(8,8);

block_size = 8;

% n = ceil(log2(min(rows,cols))); % maximum number of wavelet decomposition levels
n = wmaxlev([8 8], wavelet); % maximum number of wavelet decomposition levels
[C,S] = wavedec2(dummy_wav, n, wavelet); % conversion to 2D, wavelet decomposition

%% DWT 2D matrix Psi
% The matrix columns are unit impulse responses for each pixel
i = 1;
delta = zeros(r*c,1); % Store to 1D vector for simplicity
delta(i) = 1; % Unit impulse at the first pixel position
C = wavedec2(reshape(delta,[r,c]), n, wavelet); % conversion to 2D, wavelet decomposition
DWTm = sparse(length(C), r*c); % space alocation for the result (sparse matrix)
DWTm(:,i) = C.'; % first image column

for i=2:r*c,
    delta(i-1)=0;
    delta(i) = 1; % Unit impulse at each pixel position
    C = wavedec2(reshape(delta,[r,c]), n, wavelet); % conversion to 2D, wavelet decomposition
    DWTm(:,i) = C.'; % all image columns
end

% Check the matrix construction
% Y = (DWT * reshape(im, rows*cols,1)).'; % Aplication on an image: conversion to 1D, matrix multiplication
% C = wavedec2(im, n, wavelet); % Direct implementation using MATLAB function
% max(abs(Y-C))  % must be zero


%% IDWT 2D matrix (Psi^(-1))
% The matrix columns are unit impulse responses for each spectrum coefficient
i=1;
clen = size(DWTm,1);
delta = zeros(clen,1);
delta(i) = 1; % Unit impulse at the first wavelet spectrum coefficient position
xr = waverec2(delta, S, wavelet); % wavelet reconstruction (inverse transform)
IDWTm = sparse(r*c, clen); % space alocation for the result (sparse matrix)
IDWTm(:,i) = reshape(xr, r*c, 1).'; % conversion to 1D

for i=2:clen,
    delta(i-1)=0;
    delta(i) = 1; % Unit impulse at each wavelet spectrum coefficient position
    xr = waverec2(delta, S, wavelet); % wavelet reconstruction (inverse transform)
    IDWTm(:,i) = reshape(xr, r*c, 1).'; % conversion to 1D
end

% Check the perfect reconstruction
% full(max(max(abs(IDWT * DWT - speye(rows*cols)))))  % must be zero

%% DCT MATRIX GENERATION
% The matrix columns are unit impulse responses for each pixel

i = 1;
delta = zeros(block_size*block_size,1); % Store to 1D vector for simplicity
delta(i) = 1; % Unit impulse at the first pixel position
C = dct2(reshape(delta,[block_size, block_size])); % conversion to 2D, DCT
DCTm = sparse(block_size*block_size, block_size*block_size); % space alocation for the result (sparse matrix)
DCTm(:,i) = C(:); % first image column

for i=2:block_size*block_size
    delta(i-1)=0;
    delta(i) = 1; % Unit impulse at each pixel position
    C = dct2(reshape(delta,[block_size, block_size])); % conversion to 2D, DCT
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
xr = idct2(reshape(delta,[block_size,block_size])); % idct (inverse transform)
IDCTm = sparse(block_size*block_size, clen); % space alocation for the result (sparse matrix)
IDCTm(:,i) = reshape(xr, block_size*block_size, 1).'; % conversion to 1D

for i=2:clen,
    delta(i-1)=0;
    delta(i) = 1; % Unit impulse at each DCT spectrum coefficient position
    xr = idct2(reshape(delta,[block_size,block_size])); % reconstruction (inverse transform)
    IDCTm(:,i) = reshape(xr, block_size*block_size, 1).'; % conversion to 1D
end
%%
N2=block_size^2;

%         DCTm = dctmtx(N2);
%         IDCTm = dctmtx(N2)';


image = im2double(rgb2gray(imread('D:\Diplomski rad\peppers.png')));


figure, imagesc(image), title('Real image'), colormap gray, axis image

% image=imresize(image,[round(size(image,1)/8)*8,round(size(image,2)/8)*8]);
image=imresize(image,[32 32]);
% image=randn(16,16);

figure, imagesc(image), title('Real image - resized'), colormap gray, axis image


[rows, cols]=size(image);

for k=1:8:rows-8+1
    for l=1:8:cols-8+1
        
        im=image(k:k+7, l:l+7);
        
        [r, c] = size(im);
        
        psi=DWTm;
        psi_inv=IDWTm;
        
%         psi=DCTm;
%         psi_inv=IDCTm;
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Random unitary matrix (Phi)
        %         phi = randn(r*c);
        
        %         phi = rand(r*c);
        
        % binarize random matrix (Phi)
        
        %
        %         [U,D,V] = svd(phi);
        %         phi = U*eye(size(D))*V';
        %         %         phi=(phi)>0;
        
        percentage = 100;
        numOfMeasurements = ceil(percentage/100 * 8*8);
        
        phi = (rand(1000, 8*8)-1/2) > 0;
        phi = phi(sum(phi,2) == size(phi,2)/2,:);
        phi = phi(1:numOfMeasurements,:);
        
        % Phi(floor(numOfMeasurements/2)+1:end,:) = ~Phi(1:floor(numOfMeasurements/2),:);
        phi = phi/numOfMeasurements;
        
        %         figure, imagesc(R), colormap gray, title('Measurement Matrix - Phi'), , axis image
        
        % Check the orthonormality
        % max(max(abs(R' * R - eye(size(R)))))  %  must be zero
        
        %% double(
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check coherence
        sqrt(size(phi,2)) * max(max(abs(phi'*psi_inv)));  % From 1 - incoherent to sqrt(size(phi,2)) - coherent
        
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create ideally K sparse data
        %
        s_est = wavedec2(im, n, wavelet).'; % wavelet decomposition (transform)
        ss = sort(abs(s_est));
        % desired sparsity percentage
        p = 0.99; % desired K/N
        thr = ss(ceil((1-p) * length(ss)));
        ss = wthresh(s_est, 's', thr); % Seting N-K values of the wavelet coefficients to zero
        
        imw = waverec2(ss, S, wavelet); % wavelet reconstruction (inverse transform)
        %         figure, imagesc(imw), title('Ideally K sparse image'), colormap gray, axis image
        
        
        
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Simulated observation
        %
        
        % Observation matrix  x  input data converted to 1D
        % y = R * reshape(im, rows*cols, 1);  % Real data
        y = phi * reshape(imw, r*c, 1); % Ideally K sparse data
        
        % percentage of used measurements
        p = 0.99; % desired M/N,   K <= M   << N
        ind = rand(r*c, 1) > (1-p); % only M observations out of total N
        
        y_m = y(ind);
        phi_r = phi(ind,:); % reduced observation matrix (Phi_m)
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CS reconstruction
        % (L1 optimization problem)
        
        % min_L1 s
        % subject to y_m = Phi_m * Psi^(-1) * s
        
        % Different optimization packages
        method = 'SeDuMi'; % Options: 'L1 magic (matrix free)',  'L1 magic', 'SeDuMi'
        
        theta = phi_r*psi_inv;
        
        switch method
            
            case 'SeDuMi'
                %                 psi_inv = R_m*IDWT; % Phi_m * Psi^(-1)
                %                 psi_inv = R_m*IDCTm; % Phi_m * Psi^(-1)
                
                % Standard dual form: data conditioning for minimum L1
                
                [N1,N2]=size(theta);
                b = [ spalloc(N2,1,0); -sparse(ones(N2,1)) ];
                
                At = [ -sparse(theta) , spalloc(N1,N2,0) ;...
                    sparse(theta) , spalloc(N1,N2,0) ;...
                    speye(N2)         , -speye(N2)      ;...
                    -speye(N2)         , -speye(N2)      ;...
                    spalloc(N2,N2,0)   , -speye(N2)      ];
                
                c = [ -sparse(y_m(:)); sparse(y_m(:)); spalloc(3*N2,1,0) ];
                
                % Optimization
                tic, [~,s_est]=sedumi(At,b,c); toc % SeDuMi
                
                % Output data processing
                s_est=s_est(:);
                s_est=s_est(1:N2);
                
                %                 signal_est = psi_inv * s_est;
                signal_est = (psi_inv * s_est).';
                
                %                 figure, imagesc(yr), title('Reconstructed image - DCT'), colormap gray, axis image
                
                signal_est = waverec2(s_est, S, wavelet); % wavelet reconstruction (inverse transform)

                image_est(k:k+7, l:l+7) = reshape(signal_est, [block_size, block_size]);
                
                
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
                
            case 'cvx'
                
                cvx_begin quiet
                variable s_est(N2, 1);
                minimize( norm(s_est, 1) );
                subject to
                theta * s_est == y_m;
                cvx_end
                
                signal_est = (psi_inv * s_est).';
                
                signal_est = waverec2(s_est, S, wavelet); % wavelet reconstruction (inverse transform)
                
                image_est(k:k+7, l:l+7)= reshape(signal_est,[8 8]);
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
        end
        
    end
end


%%
figure, imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image

%% Reconstruction of the image from its sparse representation
% im_r = waverec2(s, S, wavelet); % wavelet reconstruction (inverse transform)
% figure, imagesc(im_r), title('Reconstructed image - DWT'), colormap gray, axis image
