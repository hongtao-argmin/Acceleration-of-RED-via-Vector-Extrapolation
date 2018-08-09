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
%   via steepest descend method.
%   Please refer to Section 4.2 in the paper for more details:
%   "Deploying the Denoising Engine for Solving Inverse Problems -- 
%   Gradient Descent Methods".
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   input_sigma - noise level
%   params.lambda - regularization parameter
%   params.outer_iters - number of total iterations
%   params.effective_sigma - the input noise level to the denoiser
%   orig_im - the original image, used for PSNR evaluation ONLY
%
% Outputs:
%   im_out - the reconstructed image
%   psnr_out - PSNR measured between x_est and orig_im

function [im_out, psnr_out_set,fun_val_set] = RunSD(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params, orig_im,param_MPE)

% print info every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/100);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end

% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
effective_sigma = params.effective_sigma;

% compute step size
mu = 2/(1/(input_sigma^2) + lambda);

% initialization
x_est = InitEstFunc(y);
if param_MPE.isMPE
    QQ_MPE = zeros(length(x_est(:)),param_MPE.KK+1);
    x_MPE_0 = zeros(length(x_est(:)),1); 
    MPE_count = 1;
end
fun_val_set = []; 
psnr_out_set = [];
CPU_time_set = [];
if param_MPE.isMPE && param_MPE.Iter_pre == 0 
    x_MPE_0 = x_est(:);
end

Count_pre = 0;
Start_Time = tic;
for k = 1:1:outer_iters
    
    % denoise
    f_x_est = Denoiser(x_est, effective_sigma);
    
    % update the solution
    grad1 = BackwardFunc(ForwardFunc(x_est) - y)/(input_sigma^2);
    grad2 = lambda*(x_est - f_x_est);
    x_est = x_est - mu*(grad1 + grad2);
    
    Count_pre = Count_pre+1;
    % project to [0,255]
    x_est = max( min(x_est, 255), 0);
    if param_MPE.isMPE && Count_pre>param_MPE.Iter_pre
        if MPE_count<= param_MPE.KK+1
            QQ_MPE(:,MPE_count) = x_est(:); 
            MPE_count = MPE_count+1;
        end
        if MPE_count>param_MPE.KK+1
            x_MPE_0 = extrapolate(x_MPE_0,param_MPE.KK,QQ_MPE,param_MPE.method);  
            MPE_count = 1;
            x_est = reshape(x_MPE_0,size(x_est)); 
            x_est = max(min(x_est, 255),0);
            if param_MPE.Iter_pre==0
                x_MPE_0 = x_est(:);
            end
            Count_pre = 0;
        end
    elseif Count_pre == param_MPE.Iter_pre && param_MPE.isMPE
        x_MPE_0 = x_est(:);           
    end  
    
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        CPU_time_set = [CPU_time_set;toc(Start_Time)];
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fun_val_set = [fun_val_set;fun_val];
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

