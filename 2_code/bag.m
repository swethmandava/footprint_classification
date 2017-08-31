trainingSet = imageDatastore('2\train_aug\', 'IncludeSubfolders', 1);
imds = imageSet('2\train_aug\');
A = 1:1000;
A = repmat(A, 16,1);
A = A(:)';
A = strread(num2str(A),'%s');
trainingSet.Labels = categorical(A);
bagf = bagOfFeatures(imds);
img = readimage(imds, 1);
featureVector = encode(bagf, img);
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
categoryClassifier = trainImageCategoryClassifier(trainingSet, bagf);

test_files = dir('2\valid\*.png');
num_files = size(test_files,1);
Y_test = zeros(num_files,1);
for i = 1:num_files
    filename = test_files(i).name
    index = strsplit(filename, '.');
    index = str2num(index{1});
    img = imread(['2\valid\', filename]);
    [label, ~] = predict(categoryClassifier, img);
    Y_test(index) = categoryClassifier.Labels(label);
end