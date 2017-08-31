% X = cell(16000,1);
% X_grouped = zeros(16000,3840);
% Y = zeros(16000,1);
% train_files = dir('2\train_aug\*.png');
% num_files = size(train_files,1);
% for i = 1:num_files
%     filename = train_files(i).name
%     label = strsplit(filename, '.');
%     label = strsplit(label{1}, '_');
%     label = str2num(label{1});
%     
%     I = imread(['2\train_aug\', filename]);    
%     points = detectSURFFeatures(I);
%     points = points.selectStrongest(60);
%     [features, points] = extractFeatures(I, points);
%     X{i} =  features;
%     features_T = features';
%     X_grouped(i,:) = features_T(:)';
%     Y(i) = label;
% end
% [idx, C] = kmeans(X_grouped, 200);
% 
% save('TRAIN_knn.mat', 'X', 'X_grouped')
load('TRAIN_knn.mat', 'X', 'X_grouped')
test_files = dir('2\valid\*.png');
num_files = size(test_files,1);
Y_test = zeros(num_files,1);
X_test = cell(num_files,1);
X_test_grouped = zeros(num_files, 3840);
for i = 1:num_files
    filename = test_files(i).name
    index = strsplit(filename, '.');
    index = str2num(index{1});
    I = imread(['2\valid\', filename]);
    points = detectSURFFeatures(I);
    [features, points] = extractFeatures(I, points);    
    features_T = features';
    best = 0; ind_best = 0;
    for j = 1:size(X,1)
        indexpairs = matchFeatures(X{j}, features);
        if(size(indexpairs,1) > best)
            best = size(indexpairs,1);
            ind_best = j;
        end
        if(best == size(features,1))
            continue
        end
    end
    if(ind_best == 0)
        Y_test(index) = 1;
    else
        Y_test(index) = floor((ind_best - 1)/16);
    end
    X_test{index} = features;
    X_test_grouped(index, :) = features_T(:)';
end
save('TEST_knn.mat', 'X_test', 'X_test_grouped')
csvwrite('2_oto.csv', Y_test);