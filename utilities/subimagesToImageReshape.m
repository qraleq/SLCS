function [reshaped]=subimagesToImageReshape(image_estimation, synth_mask_size)

zero_matrix=zeros(8,8);

for bbox_no=1:81
    phase{1}{bbox_no}=[image_estimation{1}{bbox_no}, zero_matrix; zero_matrix, zero_matrix];
    phase{2}{bbox_no}=[zero_matrix, image_estimation{2}{bbox_no}; zero_matrix, zero_matrix];
    phase{3}{bbox_no}=[zero_matrix, zero_matrix; image_estimation{3}{bbox_no}, zero_matrix];
    phase{4}{bbox_no}=[zero_matrix, zero_matrix; zero_matrix, image_estimation{4}{bbox_no}];   
end

blocksVector=(cell2mat(phase{1})+cell2mat(phase{2})+cell2mat(phase{3})+cell2mat(phase{4}))';
% figure
% imshow(blocksVector)

% subimage contains 4 8x8 phases which makes it 16x16 subimage
rSize=16;
cSize=16;

reshaped=[];

for r=1:9
    subRow=[];
    for c=1:9
        subRow=[subRow blocksVector(1:rSize,:)];
        blocksVector(1:rSize,:)=[];
    end
    reshaped=[reshaped; subRow];
end

reshaped=reshaped';