files = dir('3/*.jpg');
clusters = 0;
cluster_descriptors = cell(2,1);
cluster_names = cell(2,1);
for i = 1:size(files,1)
    filename = strcat('3/', files(i).name);
    I = imread(filename);
    I = imgaussfilt(I);
%     I = mat2gray(entropyfilt(I));
%     M = im2double(I);
%     J = dct2(M);
%     J(abs(J) < 0.8) = 0;
%     I = idct2(J);
%     T = adaptthresh(I);
%     I = imbinarize(I,T);
%     imshowpair(I,M,'montage')
%     imwrite(I, ['3_pro/', files(i).name]);
    points = detectSURFFeatures(I);
    points = points.selectStrongest(60);
    features1 = extractFeatures(I, points);

     features = dct2(I, [20,20]);
     features1 = features(:);
     features1 = features1/max(features1);
    found = 0;
    
    matches = zeros(clusters,1);
    for cluster = 1:clusters
        
        [indexPairs, matchmetric] = matchFeatures(features1,cluster_descriptors{cluster}, 'Unique', true);
%         match_heur = sum(4-matchmetric) / (4*size(features1,1))
        match_heur = 1.0 * size(indexPairs,1)/ size(features1,1);
%         match_heur = sum((features1-cluster_descriptors{cluster}).^2)
        if(match_heur > 0.05)
            found = 1;
            matches(cluster) = match_heur;
        end
    end
    if(found == 1)
        [~, cluster] = max(matches);
        max(matches)
        indexPairs = matchFeatures(features1,cluster_descriptors{cluster}, 'Unique', true);
        cluster_descriptors{cluster} = [cluster_descriptors{cluster}; features1(indexPairs(:,1), :)];
        cluster_names{cluster} = [cluster_names{cluster}; filename];
    else
        clusters = clusters + 1;
        cluster_descriptors{clusters} = features1;
        cluster_names{clusters} = {filename};
    end
end