function [image_est]=L1OptimizationCVX(y, psi, psi_inv, theta, no_of_measurements_for_reconstruction,S)

%% CVX SOLVER - alternative to SeDuMi

image_est = [];

% Reconstructing initial signal
cvx_solver sedumi

cvx_begin quiet
    cvx_precision high

    variable s_est(64, 1);
    minimize(norm(s_est, 1));
    subject to
    theta * s_est == y(1:no_of_measurements_for_reconstruction)';
cvx_end

image_est = (psi_inv * s_est).';

% signal_est = waverec2(s_est, S, 'haar');
%     image_est = idct2(s_est);

