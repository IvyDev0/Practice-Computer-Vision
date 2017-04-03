clear; clc
% 15组训练集
for k = 1:15
    for i = 1:11
        if i >= 10
            img = imread(['p1_data/00', num2str(k), '/', num2str(i), '.jpg']);
        else
            img = imread(['p1_data/00', num2str(k), '/0', num2str(i), '.jpg']);
        end
        array = im2double(rgb2gray(img));
        [m, n] = size(array);
        % Data 每一行为一个图像的灰度值向量
        Data(i,1:m*n,k) = reshape(array, 1, m*n);
    end
    Sample(:,1:m*n,k) = Data(1:7,1:m*n,k);
    Test(:,1:m*n,k) = Data(8:11,1:m*n,k);
    
    % 用训练样本做PCA分析
    % 得到正交变换的基，和特征脸
    base(1:m*n, :, k) = pca(Sample(:, :, k)); 
    D(:,:,k) = Sample(:,:,k) * base(:,:,k);    
end

% Test
for k = 1:15
    correct = 0;
    for i = 8:11        
        test = Test(i-7,:,k) * base(:,:,k); % 得到测试样本的特征脸
        for j = 1:15 
            % 计算测试与每组训练样本之间的特征脸的距离
            for m = 1:7
                Dis(m,:,j) = D(m,:,j)-test;
            end
            distance(j) = norm(Dis(:,:,j),inf);
        end
        % 看是和哪一组特征脸的距离最小
        if find(distance==min(distance)) == k  
            correct = correct+1;
        end        
    end
    ratio(k) = correct / 4; % 该组的识别率
end
acc = sum(ratio) / 15 % 15组的平均识别率