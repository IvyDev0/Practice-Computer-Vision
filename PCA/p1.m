clear; clc
% 对 15x7个训练样本做 PCA 提取特征
for k = 1:15
    for i = 1:7
        img = imread(['p1_data/00', num2str(k), '/0', num2str(i), '.jpg']);
        array = im2double(rgb2gray(img));
        [m, n] = size(array);
        Sample( (k-1)*7+i,: ) = reshape(array, 1, m*n);
    end
    Avarage(k,:) = mean(Sample((k-1)*7+1:k*7,:),1);  % 得到每个组训练样本的平均脸
end 
[base, SCORE, latent] = pca(Sample); 


% 去方差占比前90%，进一步降维
weight = latent(:) / sum(latent);
sumw(1) = weight(1);
for i = 2:size(latent)
    sumw(i) = sumw(i-1)+weight(i);
    if sumw(i) > 0.9
        dimension = i;
        break
    end
end
base = base(:,1:dimension);
PAvarage = Avarage * base;  % 将15个组的平均脸降维

% 对 15x4个测试样本
correct = 0;
for k = 1:15
    for i = 8:11
        if i >= 10
            img = imread(['p1_data/00', num2str(k), '/', num2str(i), '.jpg']);
        else
            img = imread(['p1_data/00', num2str(k), '/0', num2str(i), '.jpg']);
        end
        array = im2double(rgb2gray(img));
        [m, n] = size(array);
        Ptest = reshape(array, 1, m*n) * base;
        % 计算15个组的特征脸和该降维后的测试特征脸的欧式距离
        % 将PAvarage的每一行减去Ptest
        Dis = PAvarage - repmat(Ptest, size(PAvarage,1) ,1);
        Dis = sum(Dis.^2,2);
        if find(Dis==min(Dis)) == k 
            correct = correct+1;
        end 
    end
end     
acc = correct / (15*4)
    
    

