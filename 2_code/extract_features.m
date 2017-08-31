X = [];
train_files = dir('2\train_aug\*.png');
num_files = size(train_files,1);
for i = 1:num_files
    filename = train_files(i).name
    label = strsplit(filename, '.');
    label = strsplit(label{1}, '_');
    label = str2num(label{1});
    
    I = imread(['2\train_aug\', filename]);    
    points = detectSURFFeatures(I);
    points = points.selectStrongest(60);
    [features, points] = extractFeatures(I, points);
    features = reshape(features, [1,60,64]);
    X = [X; features];
end