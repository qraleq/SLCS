function [psi, psi_inv, C, S]=generateTransformationMatrix(psi_type, wavelet_type, block_size)

% Producing corresponding matrices of 2D linear transforms, typically given by MATLAB functions

if(isempty(wavelet_type))
    wavelet_type = 'haar'; % 'haar', 'db2', 'db4', 'sym4', 'sym8', ...
end

dummy_zeros=zeros(block_size,block_size);

% n = ceil(log2(min(rows,cols))); % maximum number of wavelet decomposition levels
n = wmaxlev(size(dummy_zeros), wavelet_type); % maximum number of wavelet decomposition levels
[C,S] = wavedec2(dummy_zeros, n, wavelet_type); % conversion to 2D, wavelet decomposition

%% DWT 2D matrix Psi
% The matrix columns are unit impulse responses for each pixel
i = 1;
delta = zeros(block_size*block_size,1); % Store to 1D vector for simplicity
delta(i) = 1; % Unit impulse at the first pixel position
C = wavedec2(reshape(delta,[block_size,block_size]), n, wavelet_type); % conversion to 2D, wavelet decomposition
DWTm = sparse(length(C), block_size*block_size); % space alocation for the result (sparse matrix)
DWTm(:,i) = C.'; % first image column

for i=2:block_size*block_size,
    delta(i-1)=0;
    delta(i) = 1; % Unit impulse at each pixel position
    C = wavedec2(reshape(delta,[block_size,block_size]), n, wavelet_type); % conversion to 2D, wavelet decomposition
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
xr = waverec2(delta, S, wavelet_type); % wavelet reconstruction (inverse transform)
IDWTm = sparse(block_size*block_size, clen); % space alocation for the result (sparse matrix)
IDWTm(:,i) = reshape(xr, block_size*block_size, 1).'; % conversion to 1D

for i=2:clen,
    delta(i-1)=0;
    delta(i) = 1; % Unit impulse at each wavelet spectrum coefficient position
    xr = waverec2(delta, S, wavelet_type); % wavelet reconstruction (inverse transform)
    IDWTm(:,i) = reshape(xr, block_size*block_size, 1).'; % conversion to 1D
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

%         DCTm = dctmtx(N2);
%         IDCTm = dctmtx(N2)';

if(strcmp(psi_type,'dct'))
    psi=DCTm;
    psi_inv=IDCTm;
elseif(strcmp(psi_type,'dwt'))
    psi=DWTm;
    psi_inv=IDWTm;
end
