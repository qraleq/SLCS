function [] = generateCalibrationMasks(directoryPath, hResolution, vResolution, plotBool)
%generateCalibrationMasks - function generates noOfMasks calibration masks
%in wanted resolution
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

for noOfCalibMeasurements=1:4
    for noOfOnes=1:64
        
        % generate binary mask with exactly 50% of ones
        %     randomMask{noOfOnes} = mod( reshape(randperm(mask_size*mask_size), mask_size, mask_size), 2 );
        
        calibMask{noOfOnes} = false(1,64) ;
        calibMask{noOfOnes}(1:noOfOnes) = true ;
        calibMask{noOfOnes} = calibMask{noOfOnes}(randperm(numel(calibMask{noOfOnes})));
        calibMask{noOfOnes}=double(reshape(calibMask{noOfOnes},[8 8]));
        
        
        zerosMask=zeros(8,8);
        
        if(plotBool==1)
            % plot one mask
            figure(101)
            imagesc(calibMask{noOfOnes});
            drawnow
            colormap gray
            title('Single Mask Preview')
        end
        
        
        % generate mask with frequencyOfOnes*100% of white pixels in a mask
        
        
        % generate all white pixels mask
        %     mask_temp=ones(mask_size,mask_size);
        
        
        for phaseNo=1:noOfPhases
            
            switch phaseNo
                case 1
                    maskTile{phaseNo}=[calibMask{noOfOnes} zerosMask
                        zerosMask zerosMask];
                    maskTile_complement{phaseNo}=[~calibMask{noOfOnes} zerosMask
                        zerosMask zerosMask];
                case 2
                    maskTile{phaseNo}=[zerosMask calibMask{noOfOnes}
                        zerosMask zerosMask];
                    maskTile_complement{phaseNo}=[zerosMask ~calibMask{noOfOnes}
                        zerosMask zerosMask];
                case 3
                    maskTile{phaseNo}=[zerosMask zerosMask
                        calibMask{noOfOnes} zerosMask];
                    maskTile_complement{phaseNo}=[zerosMask zerosMask
                        ~calibMask{noOfOnes} zerosMask];
                case 4
                    maskTile{phaseNo}=[zerosMask zerosMask
                        zerosMask calibMask{noOfOnes}];
                    maskTile_complement{phaseNo}=[zerosMask zerosMask
                        zerosMask ~calibMask{noOfOnes}];
            end
            
            
            wholeMask{noOfOnes}{phaseNo}=repmat(maskTile{phaseNo}, [round(vResolution/(mask_size*2)), round(hResolution/(mask_size*2))]);
            wholeMask{noOfOnes}{phaseNo}=wholeMask{noOfOnes}{phaseNo}(1:vResolution,1:hResolution);
            
            
            
            wholeMask_complement{noOfOnes}{phaseNo}=repmat(maskTile_complement{phaseNo}, [round(vResolution/(mask_size*2)), round(hResolution/(mask_size*2))]);
            wholeMask_complement{noOfOnes}{phaseNo}=wholeMask_complement{noOfOnes}{phaseNo}(1:vResolution,1:hResolution);
            
            wholeMask{noOfOnes}{phaseNo}(vResolution-7:vResolution,1:hResolution)=0;
            wholeMask_complement{noOfOnes}{phaseNo}(vResolution-7:vResolution,1:hResolution)=0;
            
            %         wholeMask{noOfOnes}{phaseNo}(1:16,1:hResolution)=0;
            %         wholeMask{noOfOnes}{phaseNo}(1:vResolution,1:16)=0;
            %         wholeMask{noOfOnes}{phaseNo}(1:vResolution,hResolution-16:hResolution)=0;
            
            
            % draw all masks
            if(plotBool==1)
                figure(102)
                colormap gray
                subplot(121)
                imagesc(wholeMask{noOfOnes}{phaseNo})
                subplot(122)
                imagesc(wholeMask_complement{noOfOnes}{phaseNo})
                title(['Whole Mask - noOfOnes: ', num2str(noOfOnes), '- phaseNo: ', num2str(phaseNo)])
                drawnow
            end
            
            cd(directoryPath)
            
            fileNameString1=sprintf('calib_mask_%d_%d_%02d.png',noOfCalibMeasurements,phaseNo, noOfOnes);
            imwrite(wholeMask{noOfOnes}{phaseNo}, fileNameString1);
            
            fileNameString1=sprintf('calib_mask_complement_%d_%d_%02d.png',noOfCalibMeasurements,phaseNo, noOfOnes);
            imwrite(wholeMask_complement{noOfOnes}{phaseNo}, fileNameString1);
            
            cd 'Mask Tiles'
            fileNameString2=sprintf('maskTile_%02d.png',noOfOnes);
            imwrite(calibMask{noOfOnes}, fileNameString2);
            
            cd ..
        end
    end
    
end
end

