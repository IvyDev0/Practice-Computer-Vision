% sift_mosaic.m
% im1,im2是图像文件名
function mosaic = sift_mosaic(im1, im2)

% single
im1 = im2single(im1) ;
im2 = im2single(im2) ;

% grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1) ; else im1g = im1 ; end
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end

%% SIFT 匹配
[f1,d1] = vl_sift(im1g) ; % f1 为关键点，d1 为相应的特征描述子。
[f2,d2] = vl_sift(im2g) ;

[matches, scores] = vl_ubcmatch(d1,d2) ;
numMatches = size(matches,2) ; 
% from sift_mosaic.m
%% RANSAC with homography model
% 先准备好这些匹配点的矩阵
X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ; % 前两行是匹配点的坐标，第三行全初始化为 1
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

clear H score ok ;
for t = 1:100 % 这里重复 100 次足够达到很好效果 
  % 计算单映射
  subset = vl_colsubset(1:numMatches, 4) ; % 从匹配点中随机选择 4 个
  A = [] ;
  for i = subset
    A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ; 
  end
  [U,S,V] = svd(A) ;
  H{t} = reshape(V(:,9),3,3) ;

  % 统计映射之后的配对点的个数
  X2_ = H{t} * X1 ;
  du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
  dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
  ok{t} = (du.*du + dv.*dv) < 6*6 ;
  score(t) = sum(ok{t}) ;
end
% 取准确配对最多的一个 H
[score, best] = max(score) ;
H = H{best} ;
ok = ok{best} ;

% from sift_mosaic.m
% 拼接

box2 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
box2_ = inv(H) * box2 ;
box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
ur = min([1 box2_(1,:)]):max([size(im1,2) box2_(1,:)]) ;
vr = min([1 box2_(2,:)]):max([size(im1,1) box2_(2,:)]) ;

[u,v] = meshgrid(ur,vr) ;
im1_ = vl_imwbackward(im2double(im1),u,v) ; % 第一幅图这部分取自原图的数据

z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
im2_ = vl_imwbackward(im2double(im2),u_,v_) ; % 第二幅图的部分是变换后的原图

mass = ~isnan(im1_) + ~isnan(im2_) ; % 两图相加
im1_(isnan(im1_)) = 0 ;
im2_(isnan(im2_)) = 0 ;
mosaic = (im1_ + im2_) ./ mass ; % 加权
![trees](trees.jpg)
figure(2) ; clf ;
imagesc(mosaic) ; axis image off ;
title('Mosaic') ;

if nargout == 0, clear mosaic ; end
end