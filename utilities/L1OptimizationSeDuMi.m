function []=L1OptimizationSeDuMi(y)

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

for phase_no=1:no_of_phases
    
    % Standard dual form: data conditioning for minimum L1
    
    c = [ -sparse(y{phase_no}(:)); sparse(y{phase_no}(:)); spalloc(3*N,1,0) ];
    
    % Optimization
    tic, [~,s]=sedumi(At, b, c); toc % SeDuMi
    
    % Output data processing
    s=s(:);
    s=s(1:N);
    
    yr{phase_no} = psi_inv * s;
    
    yr{phase_no}= reshape(yr{phase_no},8,8);
end

image_reconstruction=[yr{1} yr{2}; yr{3} yr{4}];

figure, imshow(image_reconstruction), colormap gray, title('Reconstruction'), axis image


