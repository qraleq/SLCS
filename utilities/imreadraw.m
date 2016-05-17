function [raw_image, t, meta_info]=imreadraw(filename, crop, bayer_color)

warning off MATLAB:tifflib:TIFFReadDirectory:libraryWarning
warning off
t = Tiff(filename,'r');
offsets = getTag(t,'SubIFD');
setSubDirectory(t,offsets(1));
raw_image = double(read(t)); % Create variable 'raw', the Bayer CFA data
close(t);
meta_info = imfinfo(filename);

    % choose which color you want to extract from Bayer CFA
    if(strcmp(bayer_color,'r'))
        %         raw_image(1:2:end,1:2:end)=0; %r
        raw_image(2:2:end,1:2:end)=0; %g
        raw_image(1:2:end,2:2:end)=0; %g
        raw_image(2:2:end,2:2:end)=0; %b
    elseif(strcmp(bayer_color,'g1'))
        raw_image(1:2:end,1:2:end)=0; %r
        %         raw_image(2:2:end,1:2:end)=0; %g
        raw_image(1:2:end,2:2:end)=0; %g
        raw_image(2:2:end,2:2:end)=0; %b
    elseif(strcmp(bayer_color,'g2'))
        raw_image(1:2:end,1:2:end)=0; %r
        raw_image(2:2:end,1:2:end)=0; %g
        %         raw_image(1:2:end,2:2:end)=0; %g
        raw_image(2:2:end,2:2:end)=0; %b
    elseif(strcmp(bayer_color,'g'))
        raw_image(1:2:end,1:2:end)=0; %r
        %         raw_image(2:2:end,1:2:end)=0; %g
        %         raw_image(1:2:end,2:2:end)=0; %g
        raw_image(2:2:end,2:2:end)=0; %b
    elseif(strcmp(bayer_color,'b'))
        raw_image(1:2:end,1:2:end)=0; %r
        raw_image(2:2:end,1:2:end)=0; %g
        raw_image(1:2:end,2:2:end)=0; %g
        %         raw_image(2:2:end,2:2:end)=0; %b
    elseif(strcmp(bayer_color,'all'))
        %         raw_image(1:2:end,1:2:end)=0; %r
        %         raw_image(2:2:end,1:2:end)=0; %g
        %         raw_image(1:2:end,2:2:end)=0; %g
        %         raw_image(2:2:end,2:2:end)=0; %b
    end

if(crop.bool)
    raw_image=imcrop(raw_image,[crop.roi_x_start crop.roi_y_start crop.block_size crop.block_size]);
else
end
warning on


