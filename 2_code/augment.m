files = dir('2\train\*.png');
for i = 1:size(files,1)
    filename = files(i).name
    I = imread(['2\train\', filename]);
    file = strsplit(filename, '.');
    file = file{1};
    if(str2num(file) < 547)
        continue
    end
    im = imnoise(I, 'Gaussian', 0, 0.2);
    imwrite(im, ['2\train_aug\', file, '_.png']);
    
    for j = 1:7
        ang = j*45;
        i = imrotateq(im, ang);
        f = strcat('2\train_aug\', file, '_', num2str(ang),'.png');
        imwrite(i, f);
    end
    im = flip(im, 2);
    imwrite(im, ['2\train_aug\', file, '_f.png']);
    for j = 1:7
        ang = j*45;
        i = imrotateq(im, ang);
        f = strcat('2\train_aug\', file, '_f_', num2str(ang),'.png');
        imwrite(i, f);
    end
end