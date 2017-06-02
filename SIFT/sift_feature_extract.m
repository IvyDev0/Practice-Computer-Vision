function [feature px py] = sift_feature_extract(imgfile)
	row = 256;
	colum = 256;
	img = imread(imgfile);
	img = imresize(img,[row,colum]);
	img = im2double(rgb2gray(img));

	% 这里 octave 个数取 3；每个 octave 的高斯图像层数取 4，DoG 层数就是 3。
	sigma0 = sqrt(2);
	octave = 3; % 6*sigma*k^(octave*level)<=min(m,n)/(2^(octave-2))
	level = 3;
	D = cell(1,octave);
	 
	% 初始化高斯差分尺度空间 DoG 大小
	for i=1:octave
		D(i) = mat2cell(zeros(row*2^(2-i)+2,colum*2^(2-i)+2,level),row*2^(2-i)+2,colum*2^(2-i)+2,level);
	end
	% 第一个 octave 里的第一幅图由原始图像插值得到：
	temp_img = kron(img,ones(2));
	temp_img = padarray(temp_img,[1,1],'replicate');


	%% 图像金字塔
	for i=1:octave
	    temp_D = D{i}; 
	    for j=1:level
	    	scale = sigma0 * sqrt(2)^(2^(i-1)+(j-1)); % 第 i 个 octave 的 第 j 层图的 scale 值
	        p = level*(i-1);
	        figure(1);
	        subplot(octave,level,p+j); 

	        f = fspecial('gaussian',[1,floor(6*scale)],scale); % 尺度为 scale 的高斯函数
	        L1 = temp_img;
	        % 图像 temp_img 的尺度空间 L2，即一个变化尺度的高斯核与该图像的卷积
          	L2 = conv2(temp_img,f,'same'); 
          	L2 = conv2(L2,f','same');
          	temp_D(:,:,j) = L2-L1; % 得到差分空间

	        if(j==level)
            	temp_img = L2(2:end-1,2:end-1); % 将开始下一个octave，这里缩小图片尺度
          	end

	        imshow(uint8(255 * mat2gray(temp_D(:,:,j)))); % 画出 DoG 图像
	    end
	    D{i} = temp_D;
	    temp_img = temp_img(1:2:end,1:2:end); % 二分重采样
	    temp_img = padarray(temp_img,[1,1],'both','replicate');
	end


	%% 寻找关键点
	% 每一个采样点要和它的26个相邻点比较，如果该点是最大或最小值，
	% 就认为该点是图像在该尺度下的一个特征点
	interval = level-1;
	number = 0;
	for i = 2:octave+1
	    number=number+(2^(i-octave)*colum)*(2*row)*interval;
	end
	extrema = zeros(1,4*number);
	flag = 1;
	for i = 1:octave
	    [m,n,~] = size(D{i});
	    m = m-2; % 一张图边上的一圈是无法进行极值比较的，故大小减 2
	    n = n-2;
	    volume = m*n/(4^(i-1)); 
	    for k = 2:interval  % 每一个 octave 中，首末两层是无法进行极值比较的，取中间的图像     
	        for j = 1:volume
	            x = ceil(j/n);
	            y = mod(j-1,m)+1;
	            sub = D{i}(x:x+2,y:y+2,k-1:k+1);
	            large = max(max(max(sub)));
	            little = min(min(min(sub)));
	            if(large==D{i}(x+1,y+1,k))
	                temp=[i,k,j,1];
	                extrema(flag:(flag+3))=temp;
	                flag=flag+4;
	            end
	            if(little==D{i}(x+1,y+1,k))
	                temp=[i,k,j,-1];
	                extrema(flag:(flag+3))=temp;
	                flag=flag+4;
	            end
	        end
	    end
	end
	idx = extrema==0;
	extrema(idx)=[];
	[m,n]=size(img);
	x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
	y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
	ry=y./2.^(octave-1-extrema(1:4:end));
	rx=x./2.^(octave-1-extrema(1:4:end));
	figure(2)
	subplot(1,3,1);
	imshow(img)
	hold on
	plot(ry,rx,'r+');

	%% 除去不好的特征点
	% 去除低对比度的关键点和不稳定的边缘响应点
	threshold=0.1;
	r=10;
	extr_volume=length(extrema)/4;
	[m,n]=size(img);
	secondorder_x=conv2([-1,1;-1,1],[-1,1;-1,1]);
	secondorder_y=conv2([-1,-1;1,1],[-1,-1;1,1]);
	for i=1:octave
	    for j=1:level
	        test=D{i}(:,:,j);
	        temp=-1./conv2(test,secondorder_y,'same').*conv2(test,[-1,-1;1,1],'same');
	        D{i}(:,:,j)=temp.*conv2(test',[-1,-1;1,1],'same')*0.5+test;
	    end
	end
	for i=1:extr_volume
	    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
	    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
	    rx=x+1;
	    ry=y+1;
	    rz=extrema(4*(i-1)+2);
	    z=D{extrema(4*(i-1)+1)}(rx,ry,rz);
	    if(abs(z)<threshold)
	        extrema(4*(i-1)+4)=0;
	    end
	end
	idx=find(extrema==0);
	idx=[idx,idx-1,idx-2,idx-3];
	extrema(idx)=[];
	extr_volume=length(extrema)/4;
	x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
	y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
	ry=y./2.^(octave-1-extrema(1:4:end));
	rx=x./2.^(octave-1-extrema(1:4:end));
	figure(2)
	subplot(1,3,2);
	imshow(img)
	hold on
	plot(ry,rx,'g+');

	% 去除边缘响应
	for i=1:extr_volume
	    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
	    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
	    rx=x+1;
	    ry=y+1;
	    rz=extrema(4*(i-1)+2);
	    Dxx=D{extrema(4*(i-1)+1)}(rx-1,ry,rz)+D{extrema(4*(i-1)+1)}(rx+1,ry,rz)-2*D{extrema(4*(i-1)+1)}(rx,ry,rz);
	    Dyy=D{extrema(4*(i-1)+1)}(rx,ry-1,rz)+D{extrema(4*(i-1)+1)}(rx,ry+1,rz)-2*D{extrema(4*(i-1)+1)}(rx,ry,rz);
	    Dxy=D{extrema(4*(i-1)+1)}(rx-1,ry-1,rz)+D{extrema(4*(i-1)+1)}(rx+1,ry+1,rz)-D{extrema(4*(i-1)+1)}(rx-1,ry+1,rz)-D{extrema(4*(i-1)+1)}(rx+1,ry-1,rz);
        deter=Dxx*Dyy-Dxy*Dxy;
        R=(Dxx+Dyy)/deter;
        R_threshold=(r+1)^2/r;
        if(deter<0||R>R_threshold)
            extrema(4*(i-1)+4)=0;
        end
	end
	idx=find(extrema==0);
	idx=[idx,idx-1,idx-2,idx-3];
	extrema(idx)=[];
	extr_volume=length(extrema)/4;
	x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
	y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
	ry=y./2.^(octave-1-extrema(1:4:end));
	rx=x./2.^(octave-1-extrema(1:4:end));
	figure(2)
	subplot(1,3,3);
	imshow(img)
	hold on
	plot(ry,rx,'b+');
	px = rx; py = ry;

	%% 特征点方向分配
	kpori=zeros(1,36*extr_volume);
	minor=zeros(1,36*extr_volume);
	f=1;
	flag=1;
	for i=1:extr_volume
	    %
	    scale=sigma0*sqrt(2)^(1/level)^((extrema(4*(i-1)+1)-1)*level+(extrema(4*(i-1)+2)));
	    width=2*round(3*1.5*scale);
	    count=1;
	    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
	    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
	    %make sure the point in the searchable area
	    if(x>(width/2)&&y>(width/2)&&x<(m/2^(extrema(4*(i-1)+1)-2)-width/2-2)&&y<(n/2^(extrema(4*(i-1)+1)-2)-width/2-2))
	        rx=x+1;
	        ry=y+1;
	        rz=extrema(4*(i-1)+2);
	        reg_volume=width*width;%3? thereom
	        % make weight matrix
	        weight=fspecial('gaussian',width,1.5*scale);
	        %calculate region pixels' magnitude and region orientation
	        reg_mag=zeros(1,count);
	        reg_theta=zeros(1,count);
	    for l=(rx-width/2):(rx+width/2-1)
	        for k=(ry-width/2):(ry+width/2-1)
	            reg_mag(count)=sqrt((D{extrema(4*(i-1)+1)}(l+1,k,rz)-D{extrema(4*(i-1)+1)}(l-1,k,rz))^2+(D{extrema(4*(i-1)+1)}(l,k+1,rz)-D{extrema(4*(i-1)+1)}(l,k-1,rz))^2);
	            reg_theta(count)=atan2((D{extrema(4*(i-1)+1)}(l,k+1,rz)-D{extrema(4*(i-1)+1)}(l,k-1,rz)),(D{extrema(4*(i-1)+1)}(l+1,k,rz)-D{extrema(4*(i-1)+1)}(l-1,k,rz)))*(180/pi);
	            count=count+1;
	        end
	    end
	    %make histogram 
	    mag_counts=zeros(1,36);
	    for x=0:10:359
	        mag_count=0;
	       for j=1:reg_volume
	           c1=-180+x;
	           c2=-171+x;
	           if(c1<0||c2<0)
	           if(abs(reg_theta(j))<abs(c1)&&abs(reg_theta(j))>=abs(c2))
	               mag_count=mag_count+reg_mag(j)*weight(ceil(j/width),mod(j-1,width)+1);
	           end
	           else
	               if(abs(reg_theta(j)>abs(c1)&&abs(reg_theta(j)<=abs(c2))))
	                   mag_count=mag_count+reg_mag(j)*weight(ceil(j/width),mod(j-1,width)+1);
	               end
	           end
	       end
	          mag_counts(x/10+1)=mag_count;
	    end
	    % find the max histogram bar and the ones higher than 80% max
	    [maxvm,~]=max(mag_counts);
	     kori=find(mag_counts>=(0.8*maxvm));
	     kori=(kori*10+(kori-1)*10)./2-180;
	     kpori(f:(f+length(kori)-1))=kori;
	     f=f+length(kori);
	     temp_extrema=[extrema(4*(i-1)+1),extrema(4*(i-1)+2),extrema(4*(i-1)+3),extrema(4*(i-1)+4)];
	     temp_extrema=padarray(temp_extrema,[0,length(temp_extrema)*(length(kori)-1)],'post','circular');
	     long=length(temp_extrema);
	     minor(flag:flag+long-1)=temp_extrema;
	     flag=flag+long;
	    end
	end
	idx= minor==0;
	minor(idx)=[];
	extrema=minor;
	% delete unsearchable points and add minor orientation points
	idx= kpori==0;
	kpori(idx)=[];
	extr_volume=length(extrema)/4;

	%% 描述符的生成
	d=4; % David G. Lowe 将区域分成 4*4.
	pixel=4;
	feature=zeros(d*d*8,extr_volume);
	for i=1:extr_volume
	    descriptor=zeros(1,d*d*8);% feature dimension is 128=4*4*8;
	    width=d*pixel;
	    % x,y centeral point and prepare for location rotation
	    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
	    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
	    z=extrema(4*(i-1)+2);
	        if((m/2^(extrema(4*(i-1)+1)-2)-pixel*d*sqrt(2)/2)>x&&x>(pixel*d/2*sqrt(2))&&(n/2^(extrema(4*(i-1)+1)-2)-pixel*d/2*sqrt(2))>y&&y>(pixel*d/2*sqrt(2)))
	        sub_x=(x-d*pixel/2+1):(x+d*pixel/2);
	        sub_y=(y-d*pixel/2+1):(y+d*pixel/2);
	        sub=zeros(2,length(sub_x)*length(sub_y));
	        j=1;
	        for p=1:length(sub_x)
	            for q=1:length(sub_y)
	                sub(:,j)=[sub_x(p)-x;sub_y(q)-y];
	                j=j+1;
	            end
	        end
	        distort=[cos(pi*kpori(i)/180),-sin(pi*kpori(i)/180);sin(pi*kpori(i)/180),cos(pi*kpori(i)/180)];
	   		% accordinate after distort
	        sub_dis=distort*sub;
	        fix_sub=ceil(sub_dis);
	        fix_sub=[fix_sub(1,:)+x;fix_sub(2,:)+y];
	        patch=zeros(1,width*width);
	        for p=1:length(fix_sub)
	        patch(p)=D{extrema(4*(i-1)+1)}(fix_sub(1,p),fix_sub(2,p),z);
	        end
	        temp_D=(reshape(patch,[width,width]))';
	        %create weight matrix.
	        mag_sub=temp_D;        
	        temp_D=padarray(temp_D,[1,1],'replicate','both');
	        weight=fspecial('gaussian',width,width/1.5);
	        mag_sub=weight.*mag_sub;
	        theta_sub=atan((temp_D(2:end-1,3:1:end)-temp_D(2:end-1,1:1:end-2))./(temp_D(3:1:end,2:1:end-1)-temp_D(1:1:end-2,2:1:end-1)))*(180/pi);
	        % create orientation histogram
	        for area=1:d*d
	        cover=pixel*pixel;
	        ori=zeros(1,cover);
	        magcounts=zeros(1,8);
	        for angle=0:45:359
	          magcount=0;
	          for p=1:cover;
	              x=(floor((p-1)/pixel)+1)+pixel*floor((area-1)/d);
	              y=mod(p-1,pixel)+1+pixel*(mod(area-1,d));
	              c1=-180+angle;
	              c2=-180+45+angle;
	              if(c1<0||c2<0)
	                  if (abs(theta_sub(x,y))<abs(c1)&&abs(theta_sub(x,y))>=abs(c2))
	                      
	                      ori(p)=(c1+c2)/2;
	                      magcount=magcount+mag_sub(x,y);
	                  end
	              else
	                  if(abs(theta_sub(x,y))>abs(c1)&&abs(theta_sub(x,y))<=abs(c2))
	                      ori(p)=(c1+c2)/2;
	                      magcount=magcount+mag_sub(x,y);
	                  end
	              end              
	          end
	          magcounts(angle/45+1)=magcount;
	        end
	        descriptor((area-1)*8+1:area*8)=magcounts;
	        end
	        descriptor=normr(descriptor);
	        % cap 0.2
	        for j=1:numel(descriptor)
	            if(abs(descriptor(j))>0.2)
	            descriptor(j)=0.2;        
	            end
	        end
	        descriptor=normr(descriptor);
	        else
	            continue;
	        end
	        feature(:,i)=descriptor';
	end
	index=find(sum(feature));
	feature=feature(:,index);