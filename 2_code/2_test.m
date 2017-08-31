load('TRAIN_knn.mat')
X_train = [];
for i = 1:size(X,1)
    X_train = [X_train; X{i}];
end
Y = 1:1000;
Y = repmat(Y, 960,1);
Y = Y(:);

mdl = fitcknn(X_train, Y, 'NumNeighbors', 1)

Y_test = zeros(10000,1);
test_files = dir('2/test/');
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