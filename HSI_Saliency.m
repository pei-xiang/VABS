function [rgb, iCM, oCM, gCM, Sm,  ...
          Sm_HSI_IOC, Sm_HSI_IOG, Sm_HSI_IOE]...
          = HSI_Saliency(scene,varargin)



clear global
data = normalise(scene, '', 1);
% ------------------------------------------------------------------------
% Extract rgb,intensity and all bands
% ------------------------------------------------------------------------

rgb=data(:,:,1:3);

% ------------------------------------------------------------------------
% HVAM
% ------------------------------------------------------------------------

[iCM, oCM, gCM, Sm] = Itti_Saliency(rgb);

% ------------------------------------------------------------------------
% saliency maps generated from different combinations of conspicuity maps
% ------------------------------------------------------------------------
Sm_HSI_IOC = SaliencyMap(iCM, oCM);
Sm_HSI_IOG = SaliencyMap(iCM, gCM);
Sm_HSI_IOE = SaliencyMap(oCM, gCM);


% ------------------------------------------------------------------------
% Normalize to uint8 0-255
% ------------------------------------------------------------------------
% iCM=uint8(Normalize(iCM)*255);
% oCM=uint8(Normalize(oCM)*255);
% Sm=uint8(Normalize(Sm)*255);
% Sm_HSI_IOC=uint8(Normalize(Sm_HSI_IOC)*255);
% Sm_HSI_IOG=uint8(Normalize(Sm_HSI_IOG)*255);
% Sm_HSI_IOE=uint8(Normalize(Sm_HSI_IOE)*255);



% ------------------------------------------------------------------------
% Normalize
% ------------------------------------------------------------------------

function normalized = Normalize(map)

% Normalize map to range [0..1]
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


% ------------------------------------------------------------------------
% SaliencyMap
% ------------------------------------------------------------------------

function sm=SaliencyMap(varargin)

iCM = varargin{1};
oCM = varargin{2};

if length(varargin) == 2
    sm=(Normalize(iCM)+Normalize(oCM))/2;
elseif length(varargin) == 3
    cCM = varargin{3};
    sm = (Normalize(iCM)+Normalize(cCM)+Normalize(oCM))/3;
elseif length(varargin) == 4
    sm = (Normalize(iCM)+Normalize(oCM)+Normalize(varargin{3})+Normalize(varargin{4}))/4;    
end




