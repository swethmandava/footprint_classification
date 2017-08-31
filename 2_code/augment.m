% Augment the clean train images as described in the readme
files = dir('2\train\*.png');
for i = 1:size(files,1)
    filename = files(i).name
    I = imread(['2\train\', filename]);
    file = strsplit(filename, '.');
    file = file{1};
    
    %Add gaussian noise
    im = imnoise(I, 'Gaussian', 0, 0.2);
    imwrite(im, ['2\train_aug\', file, '_.png']);
    
    %Rotate image
    for j = 1:7
        ang = j*45;
        i = imrotateq(im, ang);
        f = strcat('2\train_aug\', file, '_', num2str(ang),'.png');
        imwrite(i, f);
    end
    
    %Flip image to obtain opposite foot
    im = flip(im, 2);
    imwrite(im, ['2\train_aug\', file, '_f.png']);
    
    %Rotate flipped image
    for j = 1:7
        ang = j*45;
        i = imrotateq(im, ang);
        f = strcat('2\train_aug\', file, '_f_', num2str(ang),'.png');
        imwrite(i, f);
    end
end