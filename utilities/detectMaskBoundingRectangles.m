function [bbox, blocks_no_x, blocks_no_y] = detectMaskBoundingRectangles(image, average_background_noise, projectorRes, plot_images)

% use morphological operations on "white" calibration image so our roi is
% extended and so that we crop whole measurement mask

se_dilate=strel('square', 2);
image_dilate=imdilate(image, se_dilate);

% convert image to bw by tresholding everything above 4*mean value of
% background noise
bw_image_dilate=image_dilate>3*mean2(average_background_noise);


% detect blobs/regions algorithm
blobDetectorStruct = regionprops(bw_image_dilate, 'BoundingBox', 'Centroid', 'Area');

% extract data from blobDetectorStruct
bounding_box=cat(1, blobDetectorStruct.BoundingBox);
area=cat(1, blobDetectorStruct.Area);
centroid=cat(1, blobDetectorStruct.Centroid);

% filter detected blobs by minimum area treshold
average_blob_size=mean(area);
sorted_region_props_filtered=[bounding_box, area, centroid];

indices = find(area<0.5*average_blob_size);
sorted_region_props_filtered(indices,:) = [];
centroid(indices,:)=[];

blocks_no_x=projectorRes.x/16;
blocks_no_y=projectorRes.y/16;

% classify detected blob centroids into bins
[N1, ~, bin1] = histcounts(centroid(:,1), blocks_no_x);
[N2, ~, bin2] = histcounts(centroid(:,2), blocks_no_y);

sorted_region_props_filtered=[sorted_region_props_filtered bin1 bin2];
% sort region props by centroid values
sorted_region_props_filtered = sortrows(sorted_region_props_filtered,[8, 9]);

if(plot_images.bool)
    figure
    I = im2uint8(bw_image_dilate);
    imshow(I, 'InitialMag', 'fit')
    hold on
end

% put bounding rectangles aroun ROIs and put numeric label on each
% block in centroid location
for k = 1:size(sorted_region_props_filtered,1)
    bbox(k,:)=sorted_region_props_filtered(k,:);
    
    if(plot_images.bool)
        text(sorted_region_props_filtered(k,6), sorted_region_props_filtered(k,7), sprintf('%d', k), 'Color', 'b');
        rectangle('Position', [sorted_region_props_filtered(k,1), sorted_region_props_filtered(k,2),sorted_region_props_filtered(k,3),sorted_region_props_filtered(k,4)],'EdgeColor', 'r', 'LineWidth', 3);
    end
end