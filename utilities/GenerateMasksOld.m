clear all
close all
clc

hResolution=1280;
vResolution=800;

noOfMasks=64;
mask_size=8;
frequencyOfOnes=32;

% for frequencyOfOnes=0:64
for i=1:noOfMasks

% generating exactly 50% binary mask
%     mask_temp = mod( reshape(randperm(mask_size*mask_size), mask_size, mask_size), 2 );

% generating aproximately 50% binary mask
%     mask_temp=binornd(ones(mask_size,mask_size), frequencyOfOnes);

% generating full ones matrix
% mask_temp=ones(mask_size,mask_size);

% generating matrix with controlable number of ones
mask_temp = false(1,64) ;
mask_temp(1:frequencyOfOnes) = true ;
mask_temp = mask_temp(randperm(numel(mask_temp)));
mask_temp=double(reshape(mask_temp,[8 8]));

zero=zeros(8,8);

% 
% mask_temp1=[mask_temp1 mask_temp1
%     mask_temp1 mask_temp1];

mask_temp1=[mask_temp zero
    zero zero];

mask_temp2=[zero mask_temp
    zero zero];

mask_temp3=[zero zero
    mask_temp zero];

mask_temp4=[zero zero
    zero mask_temp];

mask1=repmat(mask_temp1, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);
mask2=repmat(mask_temp2, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);
mask3=repmat(mask_temp3, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);
mask4=repmat(mask_temp4, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);


% mask1=repmat(mask_temp1, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);
% mask2=repmat(mask_temp2, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);
% mask3=repmat(mask_temp3, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);
% mask4=repmat(mask_temp4, [vResolution/mask_size*0.5, hResolution/mask_size*0.5]);

%         mask1{i}=repmat(mask_temp1, [vResolution/mask_size, hResolution/mask_size]);



% figure(1)
% %     figure
% colormap gray
% subplot(121)
% imagesc(mask_temp1)
% title('Mask Tile')
% subplot(122)
% imagesc(mask1{i})
% title('Whole Mask')
% drawnow

cd 'D:\Diplomski rad\1280x800 Patterns\Measurement Masks\'

% string1=sprintf('white1.png');
% imwrite(mask1, string1);
%
% string1=sprintf('white2.png');
% imwrite(mask2, string1);
%
% string1=sprintf('white3.png');
% imwrite(mask3, string1);
%
% string1=sprintf('white4.png');
% imwrite(mask4, string1);



% string1=sprintf('mask1_%d.png',i);
% imwrite(mask1{i}, string1);
%
% string1=sprintf('mask2_%d.png',i);
% imwrite(mask2{i}, string1);
%
% string1=sprintf('mask3_%d.png',i);
% imwrite(mask3{i}, string1);
%
% string1=sprintf('mask4_%d.png',i);
% imwrite(mask4{i}, string1);


%% print images with percentage description
% 
% string1=sprintf('mask_tile_%f.png',frequencyOfOnes);
% imwrite(mask_temp1, string1);
% 
% string1=sprintf('mask_tile_%f_1.png',frequencyOfOnes);
% imwrite(zeros, string1);

string1=sprintf('mask1_%0.02d.png',i);
imwrite(mask1, string1);

string1=sprintf('mask2_%0.02d.png',i);
imwrite(mask2, string1);

string1=sprintf('mask3_%0.02d.png',i);
imwrite(mask3, string1);

string1=sprintf('mask4_%0.02d.png',i);
imwrite(mask4, string1);


%%

%     string2=sprintf('mask%d.png',i);
%     imwrite(mask1{i}, string2);
%

%     string1=sprintf('maskTileR%d.png',i);
%     imwrite(mask_temp2, string1);

%     string2=sprintf('maskR%d.png',i);
%     imwrite(mask2{i}, string2);

cd ..

end