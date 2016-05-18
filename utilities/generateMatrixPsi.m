function [psi, psi_inv]=generateMatrixPsi(psi_type, wavelet_type)

%% TRANSFORMATION MATRICES PSI GENERATION
% in the CS problem, linear bases are usually defined by matrices Phi and Psi
% Producing corresponding matrices of 2D linear transforms, typically given by MATLAB functions
im=zeros(8,8);

[rows, cols] = size(im);

if(strcmp(psi_type,'wav'))
    

    % n = ceil(log2(min(rows,cols))); % maximum number of wavelet decomposition levels
    n = wmaxlev(size(im), wavelet_type); % maximum number of wavelet decomposition levels
    [C,S] = wavedec2(im, n, wavelet_type); % conversion to 2D, wavelet decomposition
    
    % DWT 2D matrix Psi
    % The matrix columns are unit impulse responses for each pixel
    i = 1;
    delta = zeros(rows*cols,1); % Store to 1D vector for simplicity
    delta(i) = 1; % Unit impulse at the first pixel position
    C = wavedec2(reshape(delta,[rows,cols]), n, wavelet_type); % conversion to 2D, wavelet decomposition
    psi = sparse(length(C), rows*cols); % space alocation for the result (sparse matrix)
    psi(:,i) = C.'; % first image column
    
    for i=2:rows*cols,
        delta(i-1)=0;
        delta(i) = 1; % Unit impulse at each pixel position
        C = wavedec2(reshape(delta,[rows,cols]), n, wavelet_type); % conversion to 2D, wavelet decomposition
        psi(:,i) = C.'; % all image columns
    end
    
    % Check the matrix construction
    % Y = (DWT * reshape(im, rows*cols,1)).'; % Aplication on an image: conversion to 1D, matrix multiplication
    % C = wavedec2(im, n, wavelet); % Direct implementation using MATLAB function
    % max(abs(Y-C))  % must be zero
    
    
    % IDWT 2D matrix (Psi^(-1))
    % The matrix columns are unit impulse responses for each spectrum coefficient
    i=1;
    clen = size(psi,1);
    delta = zeros(clen,1);
    delta(i) = 1; % Unit impulse at the first wavelet spectrum coefficient position
    xr = waverec2(delta, S, wavelet_type); % wavelet reconstruction (inverse transform)
    psi_inv = sparse(rows*cols, clen); % space alocation for the result (sparse matrix)
    psi_inv(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
    
    for i=2:clen,
        delta(i-1)=0;
        delta(i) = 1; % Unit impulse at each wavelet spectrum coefficient position
        xr = waverec2(delta, S, wavelet_type); % wavelet reconstruction (inverse transform)
        psi_inv(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
    end
    
    % Check the perfect reconstruction
    % full(max(max(abs(IDWT * DWT - speye(rows*cols)))))  % must be zero
    
elseif(strcmp(psi_type,'dct'))
    
    %% DCT MATRIX GENERATION
    
    i = 1;
    delta = zeros(rows*cols,1); % Store to 1D vector for simplicity
    delta(i) = 1; % Unit impulse at the first pixel position
    C = dct2(reshape(delta,[rows, cols])); % conversion to 2D, DCT
    psi = sparse(rows*cols, rows*cols); % space alocation for the result (sparse matrix)
    psi(:,i) = C(:); % first image column
    
    for i=2:rows*cols
        delta(i-1)=0;
        delta(i) = 1; % Unit impulse at each pixel position
        C = dct2(reshape(delta,[rows, cols])); % conversion to 2D, DCT
        psi(:,i) = C(:); % all image columns
    end
    
    % Check the matrix construction
    % x = randn(8,8);
    % Y = (DCT * reshape(x, 8*8,1)).'; % Aplication on an image: conversion to 1D, matrix multiplication
    % C = dct2(x); % Direct implementation using MATLAB function
    % max(abs(Y-C(:).'))  % must be zero
    
    %% IDCT 2D matrix 8 x 8
    % The matrix columns are unit impulse responses for each spectrum coefficient
    
    i=1;
    clen = size(psi,1);
    delta = zeros(clen,1);
    delta(i) = 1; % Unit impulse at the first wavelet spectrum coefficient position
    xr = idct2(reshape(delta,[rows,cols])); % idct (inverse transform)
    psi_inv = sparse(rows*cols, clen); % space alocation for the result (sparse matrix)
    psi_inv(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
    
    for i=2:clen,
        delta(i-1)=0;
        delta(i) = 1; % Unit impulse at each DCT spectrum coefficient position
        xr = idct2(reshape(delta,[rows,cols])); % reconstruction (inverse transform)
        psi_inv(:,i) = reshape(xr, rows*cols, 1).'; % conversion to 1D
    end
    
end

%% ALTERNATIVE DCT MATRIX PSI GENERATION
% psi matrix generation - depending in which domain problem is sparse, use
% corresponding transformation matrix

% N=64;
% 
% DCTm = dctmtx(N);
% IDCTm = dctmtx(N)';
% 
% psi=DCTm;
% psi_inv=IDCTm;