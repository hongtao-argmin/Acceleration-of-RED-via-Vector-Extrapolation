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

% Objective:
%   Minimize E(x) = 1/(2sigma^2)||Hx-y||_2^2 + 0.5*lambda*x'*(x-denoise(x))
%   via the fixed-point method.
%   Please refer to Section 4.2 in the paper for more details:
%   "Deploying the Denoising Engine for Solving Inverse Problems --
%   Fixed-Point Strategy".
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   input_sigma - noise level
%   params.lambda - regularization parameter
%   params.outer_iters - number of total iterations
%   params.inner_iters - number of inner FP iterations
%   params.use_fft - solve the linear system Az = b using FFT rather than
%                    running gradient descent for params.inner_iters. This
%                    feature is suppoterd only for deblurring
%   params.psf     - the Point Spread Function (used only when
%                    use_fft == true).
%   params.effective_sigma - the input noise level to the denoiser
%   orig_im - the original image, used for PSNR evaluation ONLY
%
% Outputs:
%   im_out - the reconstructed image
%   psnr_out - PSNR measured between x_est and orig_im


function [im_out, psnr_out_set,fun_val_set,CPU_time_set,resid_norm_set] = RunFP(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params, orig_im,param_MPE,param_MPE_inner,isPlot)
window_size = 1;
iswindow = true;
QQ_temp = [];

if param_MPE.KK<=0
    error('window is too large')
end
% print infp every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
% PRINT_MOD = floor(params.outer_iters/5);
resid_norm_set = [];
if nargin <=9
    isPlot = false; % whether to plot the figure
end

if ~QUIET
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end

% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
inner_iters = params.inner_iters;
effective_sigma = params.effective_sigma;

% initialization
x_est = InitEstFunc(y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if param_MPE.isMPE
    QQ_MPE = zeros(length(x_est(:)),param_MPE.KK+1);
    x_MPE_0 = zeros(length(x_est(:)),1);
    MPE_count = 1;
end

if param_MPE_inner.isMPE && ~isfield(params,'use_fft')
    QQ_MPE_inner = zeros(length(x_est(:)),param_MPE_inner.KK+1);
    x_MPE_inner_0 = zeros(length(x_est(:)),1);
    MPE_inner_count = 1;
end

fun_val_set = [];
psnr_out_set = [];
CPU_time_set = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ht_y = BackwardFunc(y)/(input_sigma^2);

% compute the fft of the psf (useful for deblurring)
if isfield(params,'use_fft') && params.use_fft == true
    [h, w, ~] = size(y);
    fft_psf = zeros(h, w);
    t = floor(size(params.psf, 1)/2);
    fft_psf(h/2+1-t:h/2+1+t, w/2+1-t:w/2+1+t) = params.psf;
    fft_psf = fft2( fftshift(fft_psf) );
    
    fft_y = fft2(y);
    fft_Ht_y = conj(fft_psf).*fft_y / (input_sigma^2);
    fft_HtH = abs(fft_psf).^2 / (input_sigma^2);
end

if param_MPE.isMPE && param_MPE.Iter_pre == 0
    x_MPE_0 = x_est(:);
end
x_pre = x_est;
Count_pre = 0;
Start_Time = tic;
% outer iterations
for k = 1:1:outer_iters
    
    % apply the denoising engine
    f_x_est = Denoiser(x_est, effective_sigma);%*0.999^(k-1)/1.005^(k-1)
    
    % solve Az = b by a variant of the SD method, where
    % A = 1/(sigma^2)*H'*H + lambda*I, and
    % b = 1/(sigma^2)*H'*y + lambda*denoiser(x_est)
    if isfield(params,'use_fft') && params.use_fft == true
        b = fft_Ht_y + lambda*fft2(f_x_est);
        A = fft_HtH + lambda;
        x_est = real(ifft2( b./A ));
        x_est = max( min(x_est, 255), 0);
        Count_pre = Count_pre+1;
        if param_MPE.isMPE && Count_pre>param_MPE.Iter_pre
            if MPE_count<= param_MPE.KK+1
                QQ_MPE(:,MPE_count) = x_est(:);
                MPE_count = MPE_count+1;
            end
            if MPE_count>param_MPE.KK+1
                if iswindow && isempty(QQ_temp)
                    x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK,QQ_MPE,param_MPE.method);
                    QQ_temp = QQ_MPE(:,end-window_size+1:end);
                    param_MPE.KK = param_MPE.KK-window_size;
                    QQ_MPE = [];
                else
                    if iswindow
                        x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK+window_size,[QQ_temp QQ_MPE],param_MPE.method);
                        QQ_temp = QQ_MPE(:,end-window_size+1:end);
                    else
                        x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK,QQ_MPE,param_MPE.method);
                    end
                end
                MPE_count = 1;
                x_est = reshape(x_MPE_0,size(x_est));
                x_est = max( min(x_est, 255), 0);
                if param_MPE.Iter_pre==0
                    x_MPE_0 = x_est(:);
                end
                Count_pre = 0;
            end
        elseif  param_MPE.isMPE && Count_pre == param_MPE.Iter_pre
            x_MPE_0 = x_est(:);
        end
        %         fprintf('The difference between images in %d-iteration is %.5f:\n',k,norm(x_est(:)-x_pre(:))/numel(x_est))%
        %         x_pre = x_est;
    else
        Count_pre_inner = 0;
        
        if param_MPE_inner.isMPE && param_MPE_inner.Iter_pre == 0
            x_MPE_inner_0 = x_est(:);
        end
        
        if isPlot
            resid_norm = [];
            figure(1)
        end
        for j = 1:1:inner_iters % gradient method
            b = Ht_y + lambda*f_x_est;
            ht_h_x_est = BackwardFunc(ForwardFunc(x_est))/(input_sigma^2);
            res = b - ht_h_x_est - lambda*x_est;
            if isPlot
                resid_norm = [resid_norm;norm(res(:))];
                semilogy(resid_norm);
                drawnow;
            end
            A_res = BackwardFunc(ForwardFunc(res))/(input_sigma^2) + lambda*res;
            mu_opt = mean(res(:).*A_res(:))/mean(A_res(:).*A_res(:));
            %             mu_opt = 80;
            x_est = x_est + mu_opt*res;
            x_est = max( min(x_est, 255), 0);
            Count_pre_inner = Count_pre_inner+1;
            if param_MPE_inner.isMPE && Count_pre_inner>param_MPE_inner.Iter_pre
                if MPE_inner_count<= param_MPE_inner.KK+1
                    QQ_MPE_inner(:,MPE_inner_count) = x_est(:);
                    MPE_inner_count = MPE_inner_count+1;
                end
                if MPE_inner_count>param_MPE_inner.KK+1
                    x_MPE_inner_0 = extrapolate(x_MPE_inner_0,param_MPE_inner.KK,...
                        QQ_MPE_inner,param_MPE_inner.method);
                    MPE_inner_count = 1;
                    x_est = reshape(x_MPE_inner_0,size(x_est));
                    x_est = max( min(x_est, 255), 0);
                    if param_MPE_inner.Iter_pre==0
                        x_MPE_inner_0 = x_est(:);
                    end
                    Count_pre_inner = 0;
                end
            elseif param_MPE_inner.isMPE && Count_pre_inner == param_MPE_inner.Iter_pre
                x_MPE_inner_0 = x_est(:);
            end
            
        end
        if isPlot
            resid_norm_set = [resid_norm_set;resid_norm];
        end
        Count_pre = Count_pre+1;
        if param_MPE.isMPE && Count_pre>param_MPE.Iter_pre
            if MPE_count<= param_MPE.KK+1
                QQ_MPE(:,MPE_count) = x_est(:);
                MPE_count = MPE_count+1;
            end
            if MPE_count>param_MPE.KK+1
                if iswindow&&isempty(QQ_temp)
                    x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK,QQ_MPE,param_MPE.method);
                    QQ_temp = QQ_MPE(:,end-window_size+1:end);
                    param_MPE.KK = param_MPE.KK-window_size;
                    QQ_MPE = [];
                else
                    if iswindow
                        x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK+window_size,[QQ_temp QQ_MPE],param_MPE.method);
                        QQ_temp = QQ_MPE(:,end-window_size+1:end);
                    else
                        x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK,QQ_MPE,param_MPE.method);
                    end
                end
                MPE_count = 1;
                x_est = reshape(x_MPE_0,size(x_est));
                x_est = max(min(x_est, 255), 0);
                if param_MPE.Iter_pre==0
                    x_MPE_0 = x_est(:);
                end
                Count_pre = 0;
            end
        elseif param_MPE.isMPE&& Count_pre == param_MPE.Iter_pre
            x_MPE_0 = x_est(:);
        end
    end
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        CPU_time_set = [CPU_time_set;toc(Start_Time)];
        %         fun_val = CostFunc(y, x_est, ForwardFunc,BackwardFunc, input_sigma,...
        %             lambda, effective_sigma);
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma);
        fun_val_set = [fun_val_set;fun_val];
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        psnr_out_set = [psnr_out_set;psnr_out];
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);
        Start_Time = tic;
    end
end

im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
% fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
%             lambda, effective_sigma);
psnr_out = ComputePSNR(orig_im, im_out);
% fun_val_set = [fun_val_set;fun_val];
% psnr_out_set = [psnr_out_set;psnr_out];
CPU_time_set = cumsum(CPU_time_set);
return

