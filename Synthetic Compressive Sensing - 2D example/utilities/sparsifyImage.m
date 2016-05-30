function [sparsified_image]=sparsifyImage(image, wavelet_type, percentage)

if(isempty(wavelet_type))
    wavelet_type = 'haar'; % 'haar', 'db2', 'db4', 'sym4', 'sym8', ...
end

% Create ideally K sparse data
n = wmaxlev(size(image), wavelet_type); % maximum number of wavelet decomposition levels
im_wav= wavedec2(image, n, wavelet_type).'; % wavelet decomposition (transform)

[~,S] = wavedec2(image, n, wavelet_type); % conversion to 2D, wavelet decomposition

ss = sort(abs(im_wav));
% desired sparsity percentage
percentage = 0.99; % desired K/N
thr = ss(ceil((1-percentage) * length(ss)));
ss = wthresh(im_wav, 's', thr); % Seting N-K values of the wavelet coefficients to zero

sparsified_image = waverec2(ss, S, wavelet_type); % wavelet reconstruction (inverse transform)
%         figure, imagesc(sparsified_image), title('Ideally K sparse image'), colormap gray, axis image
