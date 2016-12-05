clear all
close all
clc

%% jednaka vjerojatnost pojavljivanja

index = zeros(8*33, 32);

for k=1:4*33
    tmp = randperm(64);
    
    
    index(k,:) = tmp(1:32);
    index(end-k+1,:) = tmp(33:end);
end

figure(1)
histogram(index, 64);

%%

sumaB=0;
sumaW=0;

for k=1:8*33
    testB=zeros(1,64);
    testW=zeros(1,64);
    
    if(mod(k,33)==0)
%         testB(index(k,1:mod(k,33)))=1;
        testW(index(k,1:end))=1;
    else
        testB(index(k,1:mod(k,33)))=1;
        testW(index(k,mod(k,33)+1:end))=1;
    end
    
    
    
    crtajB{k}=testB;
    crtajW{k}=testW;
    
    %     figure(1)
    %     subplot(121)
    %     imagesc(testB)
    %     subplot(122)
    %     imagesc(testW)
    %     waitforbuttonpress;
    
    sumaB=sumaB+crtajB{k};
    sumaW=sumaW+crtajW{k};
    
    %     figure(3)
    %     subplot(121)
    %     imagesc(reshape(sumaB,[8,8]))
    %     subplot(122)
    %     imagesc(reshape(sumaW,[8,8]))
end

% figure(4)
% imagesc(reshape(sumaB+sumaW,[8,8]))


%%

noOfPhases=4;
mask_size=8;
noOfMasks=8*33;
plotBool=0;
directoryPath='D:\Projects\MATLAB Projects\Structured Light Compressive Sensing\data\1920x1080 Patterns\';

hResolution=1920;
vResolution=1080;

% for maskNo=1:noOfMasks
    
    % generate binary mask with exactly 50% of ones
    %     randomMask = mod( reshape(randperm(mask_size*mask_size), mask_size, mask_size), 2 );
    
    randomMask=ones(8,8);
%     randomMask = reshape(crtajB{maskNo},[8 8]);
    zerosMask=zeros(8,8);
    
    if(plotBool==1)
        % plot one mask
        figure(101)
        imagesc(randomMask);
        drawnow
        colormap gray
        title('Single Mask Preview')
    end
    
    
    % generate mask with frequencyOfOnes*100% of white pixels in a mask
    %     randomMask=binornd(ones(mask_size,mask_size), frequencyOfOnes);
    
    
    % generate all white pixels mask
    %     randomMask=ones(mask_size,mask_size);
    
    
    for phaseNo=1:noOfPhases
        
        switch phaseNo
            case 1
                maskTile=[randomMask zerosMask
                    zerosMask zerosMask];
            case 2
                maskTile=[zerosMask randomMask
                    zerosMask zerosMask];
            case 3
                maskTile=[zerosMask zerosMask
                    randomMask zerosMask];
            case 4
                maskTile=[zerosMask zerosMask
                    zerosMask randomMask];
        end
        
        
        wholeMask=repmat(maskTile, [round(vResolution/(mask_size*2)), round(hResolution/(mask_size*2))]);
        wholeMask=wholeMask(1:vResolution,1:hResolution);
        
        wholeMask(vResolution-7:vResolution,1:hResolution)=0;
        
        %         wholeMask(1:16,1:hResolution)=0;
        %         wholeMask(1:vResolution,1:16)=0;
        %         wholeMask(1:vResolution,hResolution-16:hResolution)=0;
        
        
        % draw all masks
        if(plotBool==1)
            figure(102)
            colormap gray
            imagesc(wholeMask)
            title(['Whole Mask - maskNo: ', num2str(maskNo), '- phaseNo: ', num2str(phaseNo)])
            drawnow
        end
        
        cd(directoryPath)
        
%         fileNameString1=sprintf('calib_black_%02d_%d_%02d.png',ceil(maskNo/33), phaseNo, mod(maskNo,33));
        
        fileNameString1=sprintf('all_bright_%d.png', phaseNo);

        imwrite(wholeMask, fileNameString1);
        
        
        cd ..
        
    end
% end
