clc
clear all

//Image acquisition & Pre-processing

[filename pathname]=uigetfile('*.jpg','Select An Image');
a=imread(fullfile(pathname,filename)); 
//a=imread('C:\Users\WELCOME\Desktop\project\bt2.jpg');
A=rgb2gray(a);
k=imnoise(A,'salt & pepper',0.001);
figure(1),
//subplot(1,2,1);
imshow(k)
title('Original Image');
figure(2),
a1=immedian(k,3);
//subplot(1,2,2);
//a1=imadjust(b);
imshow(a1);
title('Pre-Processed Image');


//K-means Clustering

function centroids = randCentroids(X, K)
    centroids = zeros(K,1); 
    idx = grand(1,"prm",(1:size(X,1)));
    centroids = X(idx(1:K), :);
endfunction


function indices = minDistCentroids(X, centroids)
  K = size(centroids, 1);
  indices = zeros(size(X,1), 1);
  m = size(X,1);

  for i=1:m
    k = 1;
    min_eucdist = sum((X(i,:) - centroids(1,:)) .^ 2);
    for j=2:K
        eucdist = sum((X(i,:) - centroids(j,:)) .^ 2);
        if(eucdist < min_eucdist)
          min_eucdist = eucdist;
          k = j;
        end
    end
    indices(i) = k;
  end
  endfunction


function centroids = computeNewCentroids(X, indices, K)

  centroids = zeros(K, 1);
  
  for i=1:K
    xi = X(indices==i,:);
    ck = size(xi,1);
    centroids(i, :) = (1/ck) * sum(xi);
  end
  endfunction

[l,b1]=size(a1);
imd=matrix(a1,l*b1,1);
imd=im2double(imd);
K = 4;
centroids = randCentroids(imd, K);

for i=1:18
  indices = minDistCentroids(imd, centroids);
  centroids = computeNewCentroids(imd, indices, K);
end

imdIDX=matrix(indices,l,b1);
//figure(2),
//imshow(uint8(imdIDX));
//title('K-means clustered Image')
figure(3),
subplot(2,2,1)
imshow(imdIDX==1);
title('cluster 1');
subplot(2,2,2)
imshow(imdIDX==2);
title('cluster 2');
subplot(2,2,3)
imshow(imdIDX==3);
title('cluster 3');
subplot(2,2,4)
imshow(imdIDX==4);
title('cluster 4');


//Morphological Operation

n=input('Position of tumor image: ');
L=(imdIDX==n);
 SE=imcreatese('ellipse',7,7);
    er=imerode(L,SE);  
    di=imdilate(er,SE);
    figure(4),
    imshow(di);
    title('Image after opening operation');
    
    
//Tumor area calculation

di1=sparse(di);
wp=nnz(di1);
area_of_tumor=sqrt(wp)*0.264;
z="Area of tumor:  " + string(area_of_tumor) + " mm2";
messagebox(z,"Area of tumor","info");
