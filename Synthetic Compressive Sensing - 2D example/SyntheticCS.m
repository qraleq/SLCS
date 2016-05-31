%% D. Sersic, M. Vucic, October 2015
%% Update: I. Ralasic, May 2016
% Compressive sensing (CS) - 2D example
%
% Assumption of sparsity
% There is a linear base Psi in which observed phenomenon is sparse:
% only K non-zero spectrum samples out of total N are non-zero, K << N
%
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

clear
close all
clc

addpath('utilities')

image_est = [];
psi_type='dwt';
wavelet_type='haar';

% Different optimization packages
method = 'SeDuMi'; % Options: 'cvx', 'SeDuMi'

% Choose block size
block_size = 8;

%% In the CS problem, linear bases are usually defined by matrices Phi and Psi

[psi, psi_inv, C, S]=generateTransformationMatrix(psi_type,[], block_size);
phi=generateMeasurementMatrix([],block_size);

% Check coherence
sqrt(size(phi,2)) * max(max(abs(phi'*psi_inv)))  % From 1 - incoherent to sqrt(size(phi,2)) - coherent

image = im2double(rgb2gray(imread('D:\Diplomski rad\peppers.png')));

figure, imagesc(image), title('Real image'), colormap gray, axis image

% image=imresize(image,[round(size(image,1)/8)*8,round(size(image,2)/8)*8]);
image=imresize(image,[64 64]);

figure, imagesc(image), title('Real image - resized'), colormap gray, axis image

[rows, cols]=size(image);

tic

for k=1:block_size:rows-block_size+1
    for l=1:block_size:cols-block_size+1
        
        im=image(k:k+block_size-1, l:l+block_size-1);
        
        sparsified_image=sparsifyImage(im,[],0.99);
        
        % Simulated observation
        
        % Observation matrix  x  input data converted to 1D
        % y = R * reshape(im, rows*cols, 1);  % Real data
        y = phi * reshape(sparsified_image, block_size*block_size, 1); % Ideally K sparse data
        
        % percentage of used measurements
        p = 0.9; % desired M/N,   K <= M   << N
        ind = rand(block_size*block_size, 1) > (1-p); % only M observations out of total N
        
        y_m = y(ind);
        phi_r = phi(ind,:); % reduced observation matrix (Phi_m)
        
        % CS reconstruction - L1 optimization problem
        
        % min_L1 subject to y_m = Phi_m * Psi^(-1) * s
        
        theta = phi_r*psi_inv;
        [M,N]=size(theta);
        
        switch method
            case 'SeDuMi'
                
                % Standard dual form: data conditioning for minimum L1
                
                
                b = [ spalloc(N,1,0); -sparse(ones(N,1)) ];
                
                At = [ -sparse(theta) , spalloc(M,N,0) ;...
                    sparse(theta) , spalloc(M,N,0) ;...
                    speye(N)         , -speye(N)      ;...
                    -speye(N)         , -speye(N)      ;...
                    spalloc(N,N,0)   , -speye(N)      ];
                
                c = [ -sparse(y_m(:)); sparse(y_m(:)); spalloc(3*N,1,0) ];
                
                % Optimization
                pars.fid=0; % suppress output
                K.l = max(size(At));
                
                tic, [~,s_est]=sedumi(At,b,c,K,pars); toc % SeDuMi
                
                % Output data processing
                s_est=s_est(:);
                s_est=s_est(1:N);
                
                if(strcmp(psi_type,'dct'))
                    signal_est = (psi_inv * s_est).';
                    
                elseif(strcmp(psi_type,'dwt'))
                    signal_est = waverec2(s_est, S, wavelet_type); % wavelet reconstruction (inverse transform)
                end
                
                
                image_est(k:k+block_size-1, l:l+block_size-1) = reshape(signal_est, [block_size block_size]);
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
                
            case 'cvx'
                
                cvx_solver mosek
                cvx_begin quiet
                variable s_est(N, 1);
                minimize( norm(s_est, 1) );
                subject to
                theta * s_est == y_m;
                cvx_end
                
                if(strcmp(psi_type,'dct'))
                    signal_est = (psi_inv * s_est).';
                    
                elseif(strcmp(psi_type,'dwt'))
                    signal_est = waverec2(s_est, S, wavelet_type); % wavelet reconstruction (inverse transform)
                end
                
                image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
        end
        
    end
end

toc

figure, imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction - final'), colormap gray, axis image
