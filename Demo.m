% This is an implementation of the VABS algorithm for hyperspectral anomaly detection.
% Author: Pei Xiang
%         Xidian University
%
% Please refer to the following paper if you use this code:
%
% @article{VABS, 
% author = {Pei Xiang, Jiangluqi Song, Hanlin Qin, Wei Tan, Huan Li and Huixin Zhou},
% title = {Visual Attention and Background Subtraction with Adaptive Weight for Hyperspectral Anomaly Detection},
% jounal = {Submitted to JSTARS},
% year = {2020}
%
% Version 1.0
% Copyright 2020, Xidian University


clear;
clc;
close all;

load Sandiego.mat;
load PlaneGT.mat;
map=PlaneGT;
A=double(data);
[lenth,with,bands]=size(A);
MM=lenth*with;
mask=map;

%% OCF
X = permute(A, [3, 1, 2]);
X = X(:, :);
k=30;
S_FG = get_graph(X);
F_NC_FG = get_F_NC(S_FG);
[para_NC_IE.bnds_rnk, ~] = Entrop(X);
S_FG = get_graph(X);
para_NC_IE.F = F_NC_FG; 
para_NC_IE.is_maximize = 1;
para_NC_IE.X = X; 
para_NC_IE.operator_name = 'sum';
band_set = ocf(para_NC_IE, k);

%% HVAM
final_band=A(:,:,band_set);
[rgb, iCM, oCM, gCM, Sm,Sm_HSI_IOC,Sm_HSI_IOG,Sm_HSI_IOE] = HSI_Saliency(final_band,1);
sm=mat2gray(Sm);
figure;
imshow(sm);
%% BS
FilterType = 0;
Iteration = 2000;
[result,energy]=CF(sm, FilterType, Iteration);
figure;
imshow(result);
mresult=sm-result;
figure;
imshow(mresult);

ms=mean2(mresult);

fresult=zeros(lenth,with);
for i=1:lenth
    for j=1:with
        fresult(i,j)=mresult(i,j)^2;
    end
end
fresult1=mat2gray(fresult);
figure;
imshow(fresult1);
figure;
imagesc(fresult1);
axis image;
%% AW
sp=reshape(A,lenth*with,bands);
sp1=sp';
mn1=mean(sp1,2);
dis=zeros(lenth,with);
for is=1:lenth
    for js=1:with
        xx=A(is,js,:);
        xx=xx(:);
        dis(is,js)=acos(dot(xx,mn1)/(norm(xx)*norm(mn1)));
    end
end
figure;
imshow(dis);
figure;
imagesc(dis);
axis image;

ffresult=fresult1.*dis;
ffresult=mat2gray(ffresult);
figure;
imshow(ffresult);
figure;
imagesc(ffresult);
axis image;

%% ROC
disp('Running ROC...');
mask = reshape(mask, 1, MM);
r1=reshape(ffresult,1,MM);
anomaly_map = logical(double(mask)>=1);
normal_map = logical(double(mask)==0);
r_max = max(r1(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r1 > tau);
  anomaly_map_rx =reshape(anomaly_map_rx , 1, MM);
  PF8(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD8(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area = sum((PF8(1:end-1)-PF8(2:end)).*(PD8(2:end)+PD8(1:end-1))/2);

figure,
plot(PF8, PD8, 'b', 'LineWidth', 3);
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('VABC')
axis([0 1 0 1])