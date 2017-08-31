X_train = [];
train_files = dir('2\train_aug\*.png');
num_files = size(train_files,1);
for i = 1:num_files
    filename = train_files(i).name
    label = strsplit(filename, '.');
    label = strsplit(label{1}, '_');
    label = str2num(label{1});
    
    I = imread(['2\train_aug\', filename]); 
%     Select 60 strongest SURF features of the given image
    points = detectSURFFeatures(I);
    points = points.selectStrongest(60);
    [features, points] = extractFeatures(I, points);
    features = reshape(features, [1,60,64]);
    X_train = [X_train; features];
end

Y = 1:1000;
Y = repmat(Y, 960,1);
Y = Y(:);

% Fit a KNN to the data
mdl = fitcknn(X_train, Y, 'NumNeighbors', 1)

Y_test = zeros(10000,1);
test_files = dir('2/test/');

% Use the model to predict class
for i = 1:test_files
    file = test_files{i}
    I = imread(['2/test/', file]);
    points = detectSURFFeatures(I);
    points = points.selectStrongest(60);
    [features, points] = extractFeatures(I, points);
    label = predict(mdl, features);
    file = strsplit(file, '.');
    index = str2num(file);
    Y_test(index) = label;
end