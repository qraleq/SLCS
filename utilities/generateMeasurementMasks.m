function [] = generateMeasurementMasks(directoryPath, hResolution, vResolution, plotBool)
%generateMeasurementMasks - function generates noOfMasks measurement masks and also mask tiles in
%hResolution*vResolution
%   Detailed explanation goes here

% measurements are performed in 4 phases by default
% mask_size is 8x8 by default
% 64 different measurement masks are used by default

cd(directoryPath)

% subdirectory for mask tiles
mkdir('Mask Tiles')
cd 'Mask Tiles'


noOfPhases=4;
mask_size=8;
noOfMasks=64;

for maskNo=1:noOfMasks
    
    % generate binary mask with exactly 50% of ones
%     randomMask{maskNo} = mod( reshape(randperm(mask_size*mask_size), mask_size, mask_size), 2 );
    
    randomMask{maskNo} = false(1,64) ;
    randomMask{maskNo}(1:32) = true ;
    randomMask{maskNo} = randomMask{maskNo}(randperm(numel(randomMask{maskNo})));
    randomMask{maskNo}=double(reshape(randomMask{maskNo},[8 8]));
    
    
    zerosMask=zeros(8,8);
    
    if(plotBool==1)
        % plot one mask
        figure(101)
        imagesc(randomMask{maskNo});
        drawnow
        colormap gray
        title('Single Mask Preview')
    end
    
    
    % generate mask with frequencyOfOnes*100% of white pixels in a mask
    %     randomMask{maskNo}=binornd(ones(mask_size,mask_size), frequencyOfOnes);
    
    
    % generate all white pixels mask
    %     randomMask{maskNo}=ones(mask_size,mask_size);
    
    
    for phaseNo=1:noOfPhases
        
        switch phaseNo
            case 1
                maskTile{phaseNo}=[randomMask{maskNo} zerosMask
                    zerosMask zerosMask];
            case 2
                maskTile{phaseNo}=[zerosMask randomMask{maskNo}
                    zerosMask zerosMask];
            case 3
                maskTile{phaseNo}=[zerosMask zerosMask
                    randomMask{maskNo} zerosMask];
            case 4
                maskTile{phaseNo}=[zerosMask zerosMask
                    zerosMask randomMask{maskNo}];
        end
        
        
        wholeMask{maskNo}{phaseNo}=repmat(maskTile{phaseNo}, [round(vResolution/(mask_size*2)), round(hResolution/(mask_size*2))]);
        wholeMask{maskNo}{phaseNo}=wholeMask{maskNo}{phaseNo}(1:vResolution,1:hResolution);
        
        wholeMask{maskNo}{phaseNo}(vResolution-7:vResolution,1:hResolution)=0;
        
        %         wholeMask{maskNo}{phaseNo}(1:16,1:hResolution)=0;
        %         wholeMask{maskNo}{phaseNo}(1:vResolution,1:16)=0;
        %         wholeMask{maskNo}{phaseNo}(1:vResolution,hResolution-16:hResolution)=0;
        
        
        % draw all masks
        if(plotBool==1)
            figure(102)
            colormap gray
            imagesc(wholeMask{maskNo}{phaseNo})
            title(['Whole Mask - maskNo: ', num2str(maskNo), '- phaseNo: ', num2str(phaseNo)])
            drawnow
        end
        
        cd(directoryPath)
        
        fileNameString1=sprintf('measurement_mask_%d_%02d.png',phaseNo, maskNo);
        imwrite(wholeMask{maskNo}{phaseNo}, fileNameString1);
        
        
        cd 'Mask Tiles'
        fileNameString2=sprintf('maskTile_%02d.png',maskNo);
        imwrite(randomMask{maskNo}, fileNameString2);
        
        cd ..
    end
end

end

