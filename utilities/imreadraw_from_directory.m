function [raw_images, meta_info]=imreadraw_from_directory(directory, extension, crop, bayer_color, plot)

D=dir([directory,'*',extension]);
directory_scene=directory;

numberOfFiles = length(D(not([D.isdir])));

for i=1:numberOfFiles
    
    filename = [directory_scene, num2str(i,'mask_%0.03d'),extension]; % Put file name here
    
    warning off MATLAB:tifflib:TIFFReadDirectory:libraryWarning
    warning off
    
    t{i} = Tiff(filename,'r');
    offsets{i} = getTag(t{i},'SubIFD');
    setSubDirectory(t{i},offsets{i}(1));
    raw_images{i} = double(read(t{i})); % Create variable 'raw', the Bayer CFA data
    close(t{i});
    meta_info{i} = imfinfo(filename);
    
    % choose which color you want to extract from Bayer CFA
    if(bayer_color=='r')
        %         raw_images{i}(1:2:end,1:2:end)=0; %r
        raw_images{i}(2:2:end,1:2:end)=0; %g
        raw_images{i}(1:2:end,2:2:end)=0; %g
        raw_images{i}(2:2:end,2:2:end)=0; %b
    elseif(bayer_color=='g1')
        raw_images{i}(1:2:end,1:2:end)=0; %r
        %         raw_images{i}(2:2:end,1:2:end)=0; %g
        raw_images{i}(1:2:end,2:2:end)=0; %g
        raw_images{i}(2:2:end,2:2:end)=0; %b
    elseif(bayer_color=='g2')
        raw_images{i}(1:2:end,1:2:end)=0; %r
        raw_images{i}(2:2:end,1:2:end)=0; %g
        %         raw_images{i}(1:2:end,2:2:end)=0; %g
        raw_images{i}(2:2:end,2:2:end)=0; %b
    elseif(bayer_color=='g')
        raw_images{i}(1:2:end,1:2:end)=0; %r
        %         raw_images{i}(2:2:end,1:2:end)=0; %g
        %         raw_images{i}(1:2:end,2:2:end)=0; %g
        raw_images{i}(2:2:end,2:2:end)=0; %b
    elseif(bayer_color=='b')
        raw_images{i}(1:2:end,1:2:end)=0; %r
        raw_images{i}(2:2:end,1:2:end)=0; %g
        raw_images{i}(1:2:end,2:2:end)=0; %g
        %         raw_images{i}(2:2:end,2:2:end)=0; %b
    else
        %         raw_images{i}(1:2:end,1:2:end)=0; %r
        %         raw_images{i}(2:2:end,1:2:end)=0; %g
        %         raw_images{i}(1:2:end,2:2:end)=0; %g
        %         raw_images{i}(2:2:end,2:2:end)=0; %b
    end
    
    
    
    %     % Crop to only valid pixels
    %     x_origin{i} = meta_info{i}.SubIFDs{1}.ActiveArea(2)+1; % +1 due to MATLAB indexing
    %     width{i} = meta_info{i}.SubIFDs{1}.DefaultCropSize(1);
    %     y_origin{i} = meta_info{i}.SubIFDs{1}.ActiveArea(1)+1;
    %     height{i} = meta_info{i}.SubIFDs{1}.DefaultCropSize(2);
    %     raw_images_valid{i} = double(raw(y_origin:y_origin+height-1,x_origin:x_origin+width-1));
    
    if(crop.bool)
        raw_images{i}=imcrop(raw_images{i},[crop.roi_x_start crop.roi_y_start crop.block_size crop.block_size]);
    end
    
%     if(background_subtract.bool)
%         raw_images{i}=raw_images{i}-background_subtract.avg;
%     end
    
    if(plot)
        figure(199)
        imagesc(raw_images{i})
        colormap gray
        drawnow
    end
    
    warning on
end


