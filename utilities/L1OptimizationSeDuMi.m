function [image_est]=L1OptimizationSeDuMi(y, psi, psi_inv, theta, no_of_measurements_for_reconstruction)

% standard dual form: data conditioning for minimum L1 - SeDuMi algorithm
% input data preparation for the problem
[M,N]=size(theta);

b = [ spalloc(N,1,0); -sparse(ones(N,1)) ];

At = [ -sparse(theta)   ,     spalloc(M,N,0)    ;...
    sparse(theta)   ,     spalloc(M,N,0)    ;...
    speye(N)        ,    -speye(N)          ;...
    -speye(N)        ,    -speye(N)          ;...
    spalloc(N,N,0)  ,    -speye(N)          ];

% SEDUMI OPTIMIZATION


% Standard dual form: data conditioning for minimum L1

c = [ -sparse(y(1:M)'); sparse(y(1:M)'); spalloc(3*N,1,0) ];

% Optimization
tic, [~,s_est]=sedumi(At, b, c); toc % SeDuMi

% Output data processing
s_est=s_est(:);
s_est=s_est(1:N);

image_est = (psi_inv * s_est).';






