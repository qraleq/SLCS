function [phi]=generateMeasurementMatrix(phi_type, block_size)

% Random unitary matrix (Phi)

%         phi = randn(block_size*block_size);

%         phi = rand(r*c);

% binarize random matrix (Phi)

%
%         [U,D,V] = svd(phi);
%         phi = U*eye(size(D))*V';
%         %         phi=(phi)>0;

if(isempty(phi_type))
%     percentage = 100;
%     numOfMeasurements = ceil(percentage/100 * block_size*block_size);
    
    phi = (rand(block_size^2, block_size^2)-1/2) > 0;
%     phi = phi(sum(phi,2) == size(phi,2)/2,:);
    
    % Phi(floor(numOfMeasurements/2)+1:end,:) = ~Phi(1:floor(numOfMeasurements/2),:);
%     phi = phi/numOfMeasurements;
    
    %         figure, imagesc(R), colormap gray, title('Measurement Matrix - Phi'), , axis image
    
end

