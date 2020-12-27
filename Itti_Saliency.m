function [iCM, oCM, gCM, sm] = Itti_Saliency(img,varargin)   %varargin表示用在一个函数中，输入参数不确定的情况

% Load image
image1=img;
scale=5;

% Extract channels
iIm=ExtractChannels(image1);

%Create intensity pyramid
iPyr=GaussianPyramid(iIm,scale);
        
%Create orientation pyramid
oPyr=OrientationPyramid(iPyr,scale);

% Create feature maps
iFM=IntensityFeatureMap(iPyr);     %intensity
oFM=OrientationFeatureMap(oPyr);     %orientation
gfFM=tiduFeatureMap(iPyr,scale);     %gradient

% Create conspicuity maps
[iCM,oCM,gCM]=ConspicuityMap(iFM,oFM,gfFM,iPyr);
        
sm=SaliencyMap(iCM,oCM,gCM);
end

% ------------------------------------------------------------------------
% ExtractChannels
% ------------------------------------------------------------------------

function iIm=ExtractChannels(image)

[xx,yy,bb]=size(image);
tem=zeros(xx,yy);
for ic=1:bb
    tem = tem+image(:,:,ic);
end
iIm=tem./bb;
end


% ------------------------------------------------------------------------
% GaussianPyramid
% ------------------------------------------------------------------------

function pyramid = GaussianPyramid(image,scale)

pyramid{1} = image;

for level=2:scale
    im = gausmooth(pyramid{level-1});
    s=ceil(size(pyramid{level-1})*2);
	pyramid{level} = imresize(im,s);
end
end

% ------------------------------------------------------------------------
% GaussianSmooth
% ------------------------------------------------------------------------
function im = gausmooth(im)

[m,n] = size(im);
GaussianDieOff = .0001;  
pw = 1:30; 
ssq = 2;
width = find(exp(-(pw.*pw)/(2*ssq))>GaussianDieOff,1,'last');
if isempty(width)
    width = 1;  % the user entered a really small sigma
end
t = (-width:width);
gau = exp(-(t.*t)/(2*ssq))/sum(exp(-(t.*t)/(2*ssq)));

im = imfilter(im, gau,'conv','replicate');   % run the filter accross rows
im = imfilter(im, gau','conv','replicate'); % and then accross columns
end

% ------------------------------------------------------------------------
% OrientationPyramid
% ------------------------------------------------------------------------

function oPyr = OrientationPyramid(iPyr,scale)

gabor{1}=gabor_fn(1,0.5,0,2,0);
gabor{2}=gabor_fn(1,0.5,0,2,pi/4);
gabor{3}=gabor_fn(1,0.5,0,2,pi/2);
gabor{4}=gabor_fn(1,0.5,0,2,pi*3/4);

for l=1:scale
    for o=1:4
        oPyr{l,o}=imfilter(iPyr{l},gabor{o},'symmetric');
    end
end
end

% ------------------------------------------------------------------------
% GaborFilterBank (with complex Gabors)
% ------------------------------------------------------------------------


function gb=gabor_fn(bw,gamma,psi,lambda,theta)
 
sigma = lambda/pi*sqrt(log(2)/2)*(2^bw+1)/(2^bw-1);
sigma_x = sigma;
sigma_y = sigma/gamma;

sz=fix(8*max(sigma_y,sigma_x));
if mod(sz,2)==0, sz=sz+1;end
 
[x,y]=meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));

% Rotation 
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);
 
gb=exp(-0.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);

end


% ------------------------------------------------------------------------
% IntensityFeatureMap
% ------------------------------------------------------------------------

function iFM = IntensityFeatureMap(iPyr)

for c=1:3
    for delta = 1:2
        s = c + delta;
        iFM{c,s} = abs(Subtract(iPyr{c}, iPyr{s}));
    end
end
end

% ------------------------------------------------------------------------
% OrientationFeatureMap
% ------------------------------------------------------------------------

function oFM = OrientationFeatureMap(oPyr)

for c=1:3
    for delta = 1:2
        s = c + delta;
        for o=1:4
            oFM{c,s,o}=abs(Subtract(oPyr{c,o},oPyr{s,o}));
        end
    end
end
end

% ------------------------------------------------------------------------
% GradientFeatureMap
% ------------------------------------------------------------------------

function gfFM=tiduFeatureMap(iPyr,scale)
for i=1:scale
    [Fx,Fy]=gradient(iPyr{i});
    fFM{i}=sqrt(Fx.^2+Fy.^2);     %Create gradient pyramid
end
for c=1:3
    for delta = 1:2
        s = c + delta;
        gfFM{c,s} = abs(Subtract(fFM{c}, fFM{s}));
    end
end
end

% ------------------------------------------------------------------------
% ConspicuityMap
% ------------------------------------------------------------------------

function [iCM,oCM,gCM] = ConspicuityMap(varargin)

iFM = varargin{1};
oFM = varargin{2};
gfFM = varargin{3};
ip=varargin{4};

dim=size(ip{1});
iCM=zeros(dim);
for c=1:3
    for delta = 1:2
        s = c + delta;
        weight=1;
        iCM = Add(iCM,weight*Normalize(iFM{c,s}),ip);
    end
end

oCM=zeros(dim);
for c=1:3
    for delta = 1:2
        s = c + delta;
        weight=1;
        for o=1:4
            oCM=Add(oCM,weight*Normalize(oFM{c,s,o}),ip);
        end
    end
end

gCM=zeros(dim);
for c=1:3
    for delta = 1:2
        s = c + delta;
        weight=1;
        gCM = Add(gCM,weight*Normalize(gfFM{c,s}),ip);
    end
end
end

% ------------------------------------------------------------------------
% Normalize
% ------------------------------------------------------------------------

function normalized = Normalize(map)

minValue = min(min(map));
map = map-minValue;
maxValue = max(max(map));
if maxValue>0
    map = map/maxValue;
end

% Position of local maxima
lmax = LocalMaxima(map);

% Position of global maximum
gmax = (map==1.0);

% Local maxima excluding global maximum
lmax = lmax .* (gmax==0);

% Average of local maxima excluding global maximum
nmaxima=sum(sum(lmax));
if nmaxima>0
    m = sum(sum(map.*lmax))/nmaxima;
else
    m = 0;
end
normalized = map*(1.0-m)^2;
end

% ------------------------------------------------------------------------
% LocalMaxima
% ------------------------------------------------------------------------

function maxima = LocalMaxima(A)
nRows=size(A,1);
nCols=size(A,2);
% compare with bottom, top, left, right
maxima =           (A > [A(2:nRows, :);   zeros(1, nCols)]);
maxima = maxima .* (A > [zeros(1, nCols); A(1:nRows-1, :)]);
maxima = maxima .* (A > [zeros(nRows, 1), A(:, 1:nCols-1)]);
maxima = maxima .* (A > [A(:, 2:nCols),   zeros(nRows, 1)]);
end

% ------------------------------------------------------------------------
% Cross scale subtract
% ------------------------------------------------------------------------

function result = Subtract(im1, im2)

im2 = imresize(im2, size(im1), 'bilinear');
result = im1 - im2;
end

% ------------------------------------------------------------------------
% Cross scale add
% ------------------------------------------------------------------------

function result = Add(im1, im2, im3)
im1 = imresize(im1, size(im3{1}), 'bilinear');
im2 = imresize(im2, size(im3{1}), 'bilinear');
result = im1 + im2;
end

% ------------------------------------------------------------------------
% SaliencyMap
% ------------------------------------------------------------------------

function sm=SaliencyMap(varargin)

iCM = varargin{1};
oCM = varargin{2};
if length(varargin) == 2
    sm=(Normalize(iCM)+Normalize(oCM))/2;
else
    cCM = varargin{3};
    sm=(Normalize(iCM)+Normalize(cCM)+Normalize(oCM))/3;
end
end
