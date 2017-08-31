% load('train_svm.mat', 'X_train_svm', 'Y_train_svm');
% load('test_svm.mat', 'X_test_svm', 'Y_test_svm');
% load('test.mat')
% X_test =  X; Y_test = X;
% load('train.mat')
% 
% [idx, C] = kmeans(X_train_svm, 200);
% Y_labels = zeros(10000,1);
for i = 1:size(X_test_svm,1)
    error = abs(C - repmat(X_test_svm(i,:), 200, 1));
    [~, grp] = min(sum(error.^2,2));
    classes = find(idx == grp);
    best = 0; ind_best = 0;
    train_t = X_test((i-1)*60 + 1:i*60, :);
    for j = 1:size(classes,1)
        index = classes(j);
        train_f = reshape(X(j, : , :), 60,64);
        indexpairs = matchFeatures(train_f, train_t);
        if(size(indexpairs,1) > best)
            best = size(indexpairs,1);
            ind_best = j;
        end
    end
    Y_labels(Y_test_svm(i)) = Y(ind_best);
end