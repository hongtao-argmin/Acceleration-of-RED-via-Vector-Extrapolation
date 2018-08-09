% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% Demonstration of the image restoration experiments conducted in
% Y. Romano, M. Elad, and P. Milanfar, "The Little Engine that Could: 
% Regularization by Denoising (RED)", submitted to SIAM Journal on Imaging
% Sciences, 2016. https://arxiv.org/abs/1611.02862
%
% This example reads a ground-truth image, degrades the image by 
% first blurring or downscaling it, followed by an addition of random white
% Gaussian noise. Then it calls to RED in order to restore the image. 
% This example compares the input and output PSNR, shows and saves the 
% results. The suggested image-adaptive Laplacian-regularization functional 
% is minimized using the Fixed-Point, ADMM, and Steepest Descent methods. 
% Please refer to the paper for more details.
%
% The following are the degradation models that this example handles:
% 'UniformBlur'  - 9X9 uniform psf with noise-level equal to sqrt(2)
% 'GaussianBlur' - 25X25 Gaussian psf with std 1.6 and noise-level
%                  equal to sqrt(2)
% 'Downscale'    - 7X7 Gaussian psf with std 1.6 and noise-level equal to 5
%
% The denoising engine is TNRD: Yunjin Chen, and Thomas Pock, "Trainable 
% Nonlinear Reaction Diffusion: A Flexible Framework for Fast and Effective
% Image Restoration", IEEE TPAMI 2016. The code is available in
% http://www.icg.tugraz.at/Members/Chenyunjin/about-yunjin-chen
% Note: Enable parallel pool to reduce runtime.
%
% The degradation process is similar to the one suggested in NCSR paper:
% Weisheng Dong, Lei Zhang, Guangming Shi, and Xin Li "Nonlocally 
% Centralized Sparse Representation for Image Restoration", IEEE-TIP, 2013.
% The code is available in http://www4.comp.polyu.edu.hk/~cslzhang/NCSR.htm
%

% clc;
clear;
% close all;

% configure the path
% denoising functions
addpath(genpath('./tnrd_denoising/'));
% SD, FP, and ADMM methods
addpath(genpath('./minimizers/'));
% contains the default params
addpath(genpath('./parameters/'));
% contains basic functions
addpath(genpath('./helper_functions/'));
% test images for the debluring and super resolution problems, 
% taken from NCSR software package
addpath(genpath('./test_images/'));
addpath(genpath('./VE/'));
% set light_mode = true to run the code in a sub optimal but faster mode
% set light_mode = false to obtain the results reported in the RED paper
light_mode = false;

if light_mode
    fprintf('Running in light mode. ');
    fprintf('Turn off to obatain the results reported in RED paper.\n');
else
    fprintf('Light mode option is off. ');
    fprintf('Reproducing the result in RED paper.\n');
end

%% read the original image
% deblur
% file_name = 'butterfly.tif';
% file_name = 'boats.tif';
% file_name = 'cameraman.tif';
% file_name = 'house.tif';
% file_name = 'parrots.tif';
% file_name = 'lena256.tif';
% file_name = 'barbara.tif';
file_name = 'starfish.tif';
% file_name = 'peppers.tif';
% file_name = 'leaves.tif';
% file_name = 'text.tif';



% superresolution
% file_name = 'butterfly.tif';
% file_name = 'flower.tif';
% file_name = 'girl.tif';
% file_name = 'parthenon.tif';
% file_name = 'parrots.tif';
% file_name = 'raccoon.tif';
% file_name = 'bike.tif';
% file_name = 'hat.tif';
% file_name = 'plants.tif';

%=========================================================================
Name = strsplit(file_name,'.');
Name = Name{1};
fprintf('Reading %s image...', file_name);
orig_im = imread(['./test_images/' file_name]);
orig_im = double(orig_im);

fprintf(' Done.\n');


%% define the degradation model

% choose the secenrio: 'UniformBlur', 'GaussianBlur', or 'Downscale'

% degradation_model = 'UniformBlur';
degradation_model = 'GaussianBlur';
% degradation_model = 'Downscale';

fprintf('Test case: %s degradation model.\n', degradation_model);

switch degradation_model
    case 'UniformBlur'
        % noise level
        input_sigma = sqrt(2);
        % filter size
        psf_sz = 9;
        % create uniform filter
        psf = fspecial('average', psf_sz);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function-handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
        
    case 'GaussianBlur'
        % noise level
        input_sigma = sqrt(2);
        % filter size
        psf_sz = 25;
        % std of the Gaussian filter
        gaussian_std = 1.6;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
        
    case 'Downscale'
        % noise level
        input_sigma = 5;
        % filter size
        psf_sz = 7;
        % std of the Gaussian filter
        gaussian_std = 1.6;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % scaling factor
        scale = 3;
        if mod(size(orig_im,1),scale) ~=0 || mod(size(orig_im,2),scale) ~=0
            if size(orig_im,3) == 1
                orig_im = orig_im(1:end-mod(size(orig_im,1),scale),1:end-mod(size(orig_im,2),scale)); 
            else
                orig_im = orig_im(1:end-mod(size(orig_im,1),scale),1:end-mod(size(orig_im,2),scale),:);
            end
        end
        
        if mod(size(orig_im,1),scale) ~=0
            error('The image need to be cropped to satisfy the scale factor\n')
        end
        % compute the size of the low-res image
        lr_im_sz = [ceil(size(orig_im,1)/scale),...
                    ceil(size(orig_im,2)/scale)];        
        % create the degradation operator
        H = CreateBlurAndDecimationOperator(scale,lr_im_sz,psf);
        % downscale
        ForwardFunc = @(in_im) reshape(H*in_im(:),lr_im_sz);        
        % upscale
        BackwardFunc = @(in_im) reshape(H'*in_im(:),scale*lr_im_sz);
        % special initialization (e.g. the output of other method)
        % use bicubic upscaler
        InitEstFunc = @(in_im) imresize(in_im,scale,'bicubic');
    otherwise
        error('Degradation model is not defined');
end


%% degrade the original image

switch degradation_model
    case {'UniformBlur', 'GaussianBlur'}
        fprintf('Blurring...');
        % blur each channel using the ForwardFunc
        input_im = zeros( size(orig_im) );
        
        for ch_id = 1:size(orig_im,3)
            input_im(:,:,ch_id) = ForwardFunc(orig_im(:,:,ch_id));
        end
        % use 'seed' = 0 to be consistent with the experiments in NCSR
        randn('seed', 0);

    case 'Downscale'
        
        fprintf('Downscaling...');
        % blur the image, similar to the degradation process of NCSR
        input_im = Blur(orig_im, psf);
        % decimate
        input_im = input_im(1:scale:end,1:scale:end,:);
        % use 'state' = 0 to be consistent with the experiments in NCSR
        randn('state', 0);

    otherwise
        error('Degradation model is not defined');
end

% add noise
fprintf(' Adding noise...');
input_im = input_im + input_sigma*randn(size(input_im));

% convert to YCbCr color space if needed
input_luma_im = PrepareImage(input_im);
orig_luma_im = PrepareImage(orig_im);

if strcmp(degradation_model,'Downscale')
    % upscale using bicubic
    input_im = imresize(input_im,scale,'bicubic');
    input_im = input_im(1:size(orig_im,1), 1:size(orig_im,2), :); 
end
fprintf(' Done.\n');
psnr_input = ComputePSNR(orig_im, input_im);

%% set parameters in MPE
param_VE.KK = 5;       % 8
param_VE.Iter_pre = 0; % 8
param_VE.isVE = true;
param_VE.method = 'mpe';
param_VE.isRoubst = false;
param_VE.rho = 1;
fprintf('VE parameters setting Done.\n');
%% minimize the Laplacian regularization functional via Fixed Point
fprintf('Restoring using RED: Fixed-Point method\n');
switch degradation_model
    case 'UniformBlur'
        params_fp = GetUniformDeblurFPParams(light_mode, psf, use_fft);
    case 'GaussianBlur'
        params_fp = GetGaussianDeblurFPParams(light_mode, psf, use_fft);
    case 'Downscale'
        assert(exist('use_fft','var') == 0);
        params_fp = GetSuperResFPParams(light_mode);
    otherwise
        error('Degradation model is not defined');
end


%%
% [im_out, psnr_out] = RunFP_orig(input_luma_im, ForwardFunc, BackwardFunc,...
%     InitEstFunc, input_sigma, params_fp, orig_luma_im);
%%
param_VE.isVE = false;
[est_fp_im,psnr_out_set_fp,fun_val_set_fp,CPU_time_set_fp] = RunVEFP(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             input_sigma,...
                             params_fp,...
                             orig_luma_im,param_VE);
 
psnr_fp = psnr_out_set_fp(end);                         
out_fp_im = MergeChannels(input_im,est_fp_im);
fprintf('Done.\n');

%% call MPE
fprintf('Restoring using RED-VE: Fixed-Point method\n');
param_VE.isVE = true;
[est_fp_VE_im,psnr_out_set_fp_VE,fun_val_set_fp_VE,CPU_time_set_fp_VE] = RunVEFP(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             input_sigma,...
                             params_fp,...
                             orig_luma_im,param_VE);
psnr_fp_VE = psnr_out_set_fp_VE(end);                         
out_fp_VE_im = MergeChannels(input_im,est_fp_VE_im);
fprintf('VE is Done.\n');

%% print the final objective value
fprintf('The final objective of FP is %f \n',fun_val_set_fp(end))
fprintf('The final objective of FP-MPE is %f \n',fun_val_set_fp_VE(end))
%% plot figure
figure
loglog((1:length(fun_val_set_fp)),fun_val_set_fp,'k--',...
     (1:length(fun_val_set_fp_VE)),fun_val_set_fp_VE,'r-','linewidth',1.5)
xlabel('Iteration');%z =
% set(z,'interpreter','latex')
ylabel('Cost')
legend('FP','FP-MPE','location','best')
title([Name '-' degradation_model])
grid on
%Set_Images_SR/
% print(['/home/tao/Dropbox/Acceleration-RED-MPE/manuscript/fig/' 'Obj_' degradation_model '_deblur_' Name],'-dpdf','-bestfit')

figure
semilogx((1:length(psnr_out_set_fp)),psnr_out_set_fp,'k--',...
    (1:length(psnr_out_set_fp_VE)),psnr_out_set_fp_VE,'r-','linewidth',1.5)
xlabel('Iteration');%z =
% set(z,'interpreter','latex')
ylabel('PSNR')
legend('FP','FP-MPE','location','best')
title([Name '-' degradation_model])
grid on
%Set_Images_SR/
% print(['/home/tao/Dropbox/Acceleration-RED-MPE/manuscript/fig/' 'PSNR_' degradation_model '_deblur_' Name],'-dpdf','-bestfit')

figure
semilogx(CPU_time_set_fp,fun_val_set_fp,'k--',...
    CPU_time_set_fp_VE,fun_val_set_fp_VE,'r-','linewidth',1.5)
z = xlabel('${Seconds} $');
set(z,'interpreter','latex')
ylabel('Cost')
legend('FP','FP-MPE','location','best')
title([Name '-' degradation_model])
grid on
axes('position',[.4 .4 .35 .35])
plot(CPU_time_set_fp(1:4),fun_val_set_fp(1:4),'k--',...
    CPU_time_set_fp_VE(1:4),fun_val_set_fp_VE(1:4),'r-','linewidth',1.5)
axis tight
grid on
%Set_Images_SR/
% print(['/home/tao/Dropbox/Acceleration-RED-MPE/manuscript/fig/' 'Obj_' degradation_model '_deblur_' Name '_CPU'],'-dpdf','-bestfit')
%%

save([Name '-' degradation_model '.mat'],'fun_val_set_fp','fun_val_set_fp_VE','psnr_out_set_fp','psnr_out_set_fp_VE','CPU_time_set_fp','CPU_time_set_fp_VE')

%% display final results

fprintf('Image name %s \n', file_name);
fprintf('Input PSNR = %f \n', psnr_input);
fprintf('RED: Fixed-Point PSNR = %f \n', psnr_fp);
fprintf('RED: Fixed-Point-VE PSNR = %f \n', psnr_fp_VE);

%% write images

if ~exist('./results/','dir')
    mkdir('./results/');
end

fprintf('Writing the images to ./results...');

imwrite(uint8(input_im),['./results/input_' file_name]);
imwrite(uint8(out_fp_im),['./results/est_fp_' file_name]);
imwrite(uint8(out_fp_VE_im),['./results/est_fpVE_' file_name]);

fprintf(' Done.\n');

