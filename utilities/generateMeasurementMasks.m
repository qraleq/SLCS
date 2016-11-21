clear all
close all
clc

% projector resolution
hResolution=1920;
vResolution=1080;

% set number of masks to be generated, mask size and frequency of white
% pixels in a mask
noOfMasks=64;
mask_size=8;
frequencyOfOnes=2/4;
noOfPhases=4;


for maskNo=1:noOfMasks
    
    % generate mask with 50% frequency of ones in a mask
    randomMask = mod( reshape(randperm(mask_size*mask_size), mask_size, mask_size), 2 );
    zerosMask=zeros(8,8);
    
    % plot one mask
    imagesc(randomMask);
    colormap gray
    title('Single Mask Preview')
    
    % generate mask with frequencyOfOnes*100% of white pixels in a mask
    %     mask_temp=binornd(ones(mask_size,mask_size), frequencyOfOnes);
    
    
    % generate all white pixels mask
    %     mask_temp=ones(mask_size,mask_size);
    
    
    for phaseNo=1:noOfPhases
        
        switch phaseNo
            case 1
                maskTile{phaseNo}=[randomMask zerosMask
                    zerosMask zerosMask];
            case 2
                maskTile{phaseNo}=[zerosMask randomMask
                    zerosMask zerosMask];
            case 3
                maskTile{phaseNo}=[zerosMask zerosMask
                    randomMask zerosMask];
            case 4
                maskTile{phaseNo}=[zerosMask zerosMask
                    zerosMask randomMask];
        end
        
        
        wholeMask{maskNo}{phaseNo}=repmat(maskTile{phaseNo}, [round(vResolution/(mask_size*2)), round(hResolution/(mask_size*2))]);
        wholeMask{maskNo}{phaseNo}=wholeMask{maskNo}{phaseNo}(1:vResolution,1:hResolution);
        
        
        %         % draw all masks
        %         figure(2)
        %         imagesc(wholeMask{maskNo}{phaseNo})
        %         colormap gray
        %         title(['Whole Mask - maskNo: ', num2str(maskNo), '- phaseNo: ', num2str(phaseNo)])
        %         drawnow
        
        
        %     cd 'D:\Diplomski rad\1280x800 Patterns\MeasurementMasks'
        
        %     string1=sprintf('maskTile%d.png',i);
        %     imwrite(mask_temp, string1);
        
        %     string2=sprintf('mask%d.png',i);
        %     imwrite(mask1{i}, string2);
        
        
        %     string1=sprintf('maskTileR%d.png',i);
        %     imwrite(mask_temp2, string1);
        
        %     string2=sprintf('maskR%d.png',i);
        %     imwrite(mask2{i}, string2);
        
        %     cd ..
    end
end