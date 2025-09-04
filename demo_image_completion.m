clear all

addpath(genpath('f_graphTT_vi'))
addpath(genpath('f_graphTT_opt'))
addpath('f_perfevaluate')
addpath('TestImages')
addpath('rely')

%% Load the Image
img_name = 'TestImages/airplane.mat';
load(img_name);
X = img.*255; % the original 'airplane' image

missing_rate = 90;
Mask = binornd(1,1-missing_rate/100,size(X));

IfNoiseOn = false;
Noise_type = 'salt & pepper'; % 'Gaussian'; % 

if IfNoiseOn
    if strcmp(Noise_type, 'Gaussian') % add Gaussian noise of mean 0 and variance 0.01 on the original image
        noise_var = 0.01;
        outlier_rate = 0;
        X_noise = imnoise(img,'gaussian',0,noise_var).*255;
    elseif strcmp(Noise_type, 'salt & pepper') % add 10\% salt and pepper noise
        outlier_rate = 0.1;
        X_noise = imnoise(img, 'salt & pepper', outlier_rate).* 255;
    end
    Y = X_noise .* Mask;
else
    Y = X .* Mask;
    outlier_rate = 0;
end

%% Graph Laplacian
% 1. For completion on an RGB image, the graph Laplacian is adopted to
% regularize only the 1st&2nd mode of the input tensor, i.e., the rows and
% columns of the image.
% 2. For other data format like video, please set the Lap accordingly.
% 3. For more details on the choice of the Laplacian, please see the
% reference paper

imgsize = size(Y);
Lap = cell(1,ndims(Y));
for ell = 1:ndims(Y)
    alpha = 1;
    Lap{ell} = zeros(imgsize(ell),imgsize(ell));
    if ell <= 2 % prior information on the 1st&2nd mode
        for i = 1:imgsize(ell)
            for j = 1:imgsize(ell)
                Lap{ell}(i,j) = exp(-(i-j)^2*alpha);
            end
        end
        D = diag(sum(Lap{ell}));
        Lap{ell} = D-Lap{ell}+1e-6*eye(imgsize(ell));
    else
        Lap{ell} = eye(imgsize(ell)); % set as an identity matrix if there is no prior information
    end
end

%% Perform graphTT-opt or graphTT-VI
MethodName = 'graphTTvi'; % 'graphTTopt' for the optimization based methods
% MethodName = 'graphTTopt';

IsOutlier = (outlier_rate ~= 0);
if strcmp(MethodName,'graphTTvi')
    [A_completed] = VITTC_gh(Y,Y,Mask,Lap,...
                                'maxiter',50,...
                                'isOutlier',IsOutlier,...
                                'show_info',true);
elseif strcmp(MethodName,'graphTTopt')
    if IfNoiseOn
        if outlier_rate ~= 0 % for salt-and-pepper noise
            beta_0 = 2;
            betaE = 0.01;
        else % use the following for Gaussian noise
            beta_0 = 100;
            betaE = 0;
        end
    else
        beta_0 = 2;
        betaE = 0;
    end
    rank_init = [1,64,3,1];
    [A_completed] = ttc_graph(Y,Y,Mask,Lap,beta_0,rank_init,...
                                'betaE',betaE,...
                                'show_info',true);
end

%% Evaluate the performance
rse = rse_score(A_completed,X);
psnr = psnr_score(A_completed,X);
ssim = ssim_index(rgb2gray(uint8(A_completed)),rgb2gray(uint8(X)));

%% save the result
figure;
subplot(1,3,1); imshow(X./255);
subplot(1,3,2); imshow(Y./255);
subplot(1,3,3); imshow(A_completed./255);

fprintf('PSNR: %.2f, SSIM: %.3f\n', psnr, ssim);
% if IfNoiseOn
%     save_name = sprintf('airplane_mr%d_noisy_%s.mat',missing_rate,MethodName);
% else
%     save_name = sprintf('airplane_mr%d_clean_%s.mat',missing_rate,MethodName);
% end
% save(save_name,'A_completed','rse','psnr','ssim')
