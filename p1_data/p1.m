clear; clc
% 15????
for k = 1:15
    for i = 1:11
        if i >= 10
            img = imread(['00', num2str(k), '/', num2str(i), '.jpg']);
        else
            img = imread(['00', num2str(k), '/0', num2str(i), '.jpg']);
        end
        array = im2double(rgb2gray(img));
        [m, n] = size(array);
        % Data ??????????????
        Data(i,1:m*n,k) = reshape(array, 1, m*n);
    end
    Sample(:,1:m*n,k) = Data(1:7,1:m*n,k);
    Test(:,1:m*n,k) = Data(8:11,1:m*n,k);
    
    % ??????PCA??
    % ?????????????
    base(1:m*n, :, k) = pca(Sample(:, :, k)); 
    D(:,:,k) = Sample(:,:,k) * base(:,:,k);    
end

% Test
for k = 1:15
    correct = 0;
    for i = 8:11        
        test = Test(i-7,:,k) * base(:,:,k); % ??????????
        for j = 1:15 
            % ????????????????????
            for m = 1:7
                Dis(m,:,j) = D(m,:,j)-test;
            end
            distance(j) = norm(Dis(:,:,j),inf);
        end
        % ??????????????
        if find(distance==min(distance)) == k  
            correct = correct+1;
        end        
    end
    ratio(k) = correct / 4; % ??????
end
acc = sum(ratio) / 15 % 15???????