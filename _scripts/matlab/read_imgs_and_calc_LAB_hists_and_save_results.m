[main_dir,photo_dir] = getpaths('kamyab');
% path = 'E:\thesis_phd\Image_enhancement_MrNazemi\Dataset\fivek_dataset\raw_photos\HQa1to700\photos';
raw_photo_folders = dir(photo_dir);
img_count = 1;
fprintf('[INFO]reading image from folders ... \n');
for i = 3 : length(raw_photo_folders)-1
    raw_photos_images = dir([photo_dir,'\', raw_photo_folders(i).name , '\' , 'photos']);
    
    for j = 3 : length(raw_photos_images)
        raw_image = imread([photo_dir,'\',raw_photo_folders(i).name , '\' , 'photos' , '\' , raw_photos_images(j).name]);
%         figure;imshow(raw_image);
        fprintf('\t [INFO] reading image %d of %d from folder %s/ photos /%s \n',j-2 ,length(raw_photos_images)-1,photo_dir,raw_photo_folders(i).name);
       
        if length(size(raw_image)) == 2 % gray scale image
            raw_image = cat(3, raw_image, raw_image, raw_image);
        end
        lab_image = rgb2lab(raw_image);
        hist_l = imhist(lab_image(:,:,1));
        hist_a = imhist(lab_image(:,:,2));
        hist_b = imhist(lab_image(:,:,3));
        
        hist_l_normalized = hist_l / sum(hist_l(:));
        hist_a_normalized = hist_a / sum(hist_a(:));
        hist_b_normalized = hist_b / sum(hist_b(:));
        
        [label_categorical ,label_str] = find_label( main_dir , raw_photos_images(j).name);
        images{img_count} = raw_image;%reshape(raw_image(1:170,1:256,3) , 170,256,3,1);
        image_hists_lab(: , img_count) = [hist_l_normalized', hist_a_normalized', hist_b_normalized'];
        label_one_shot(:,img_count) = label_categorical';
        label_string{img_count} = label_str;
        
        img_count = img_count + 1;
    end
end
save images images -v7.3
save image_hists_lab image_hists_lab -v7.3
save label_one_shot label_one_shot -v7.3
save label_string label_string -v7.3
