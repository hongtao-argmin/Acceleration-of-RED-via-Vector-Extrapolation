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

function [im_out, psnr_out_set,fun_val_set] = RunMultiSD(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params, orig_im,psf,psf_center)

% print info every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end
[im_row,im_column] = size(y);
Pbig = padPSF(psf,[im_row,im_column]);
% Pbig(Pbig<1e-10) = 0;
% BC = 'reflexive';
BC = 'periodic';%'zero';
[Ar,Ac] = kronDecomp(Pbig,psf_center,BC);
Ar(Ar<1e-10) = 0;
Ac(Ac<1e-10) = 0;
Ar = sparse(Ar);
Ac = sparse(Ac);
norm(ForwardFunc(y)-Ac*y*Ar','fro')

% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
effective_sigma = params.effective_sigma;

% compute step size
mu = 2/(1/(input_sigma^2) + lambda);

% initialization
x_est = InitEstFunc(y);
alpha = 1;

for k = 1:1:outer_iters
    
    % denoise
    f_x_est = Denoiser(x_est, effective_sigma);
    
    % update the solution
    grad1 = BackwardFunc(ForwardFunc(x_est) - y)/(input_sigma^2);
    grad2 = lambda*(x_est - f_x_est);
    x_est = x_est - mu*(grad1 + grad2);
    
    % project to [0,255]
    x_est = max( min(x_est, 255), 0);
    if mod(k,3) == 0
        Max_iter = 5;
        inner_iters = 200;
        dir_coarse = ...
            coarse_solverFAS(x_est,y,ForwardFunc,...
            BackwardFunc,input_sigma,lambda,Ac,Ar,effective_sigma,Max_iter,inner_iters);
        %   coarse_solver(x_est,Ht_y,ForwardFunc,BackwardFunc,input_sigma,lambda,Ac,Ar,effective_sigma/4,Max_iter,inner_iters);
        x_k_cc = x_est + alpha*dir_coarse;
        %      alpha = alpha/2;
        x_k_cc = max( min(x_k_cc, 255), 0);
        %             CostFunc(y, x_k_cc, ForwardFunc, input_sigma,...
        %                 lambda, effective_sigma)
        
        % ---------------------------------------------------------
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);
        
        x_est = x_k_cc;
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);
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

function dir_coarse = coarse_solverFAS(x_k,y,ForwardFunc,BackwardFunc,input_sigma,lambda,...
    Ac,Ar,effective_sigma,Max_iter,inner_iters)

% this downsample and prolongation is replicate, not bilinear and
% full-weighting

effective_sigma_coarse = effective_sigma/2;

% assume the dimension of the image is even and square
[im_row,~] = size(x_k);
M = im_row/2;

R = ones(1,2);
R = kron(speye(M,M),R);

% base_mat = [1 0;0.5 0.5];
% Pro = zeros(im_row,M+1);
% count = 1;
% for k=1:2:im_row
%    Pro(k:k+1,count:count+1) =  base_mat;
%    count = count+1;
% end
% Pro = Pro(:,1:end-1);
% Pro(end,1) = 0.5;
% R = Pro';
% R = sparse(R);


% scale down the image
xc_k = R*x_k*R'/4;
xc_est = xc_k;

% formulate the AH_c through Galerkin Coarsing
AcH_1 = R*(Ac'*Ac)*R'/2/input_sigma;
ArH_1 = R*(Ar'*Ar)*R'/2/input_sigma;

AcH_2 = R*R'/2;
ArH_2 = R*R'/2;

Hty = BackwardFunc(y)/(input_sigma^2);
f_x_est = Denoiser(x_k,effective_sigma);
fc = R*(Hty-BackwardFunc(ForwardFunc(x_k))/(input_sigma^2)-lambda*x_k+lambda*f_x_est)*R'/4;
temp_xc = R'*xc_k*R;
f_xc_est = Denoiser(xc_k,effective_sigma_coarse);
Acxc_k =  R*(BackwardFunc(ForwardFunc(temp_xc))/(input_sigma^2)+lambda*temp_xc)*R'/4-lambda*f_xc_est;
fc = fc + Acxc_k;

for k = 1:Max_iter
    
    for j = 1:1:inner_iters
        
        b = fc + lambda*f_xc_est;
        
        ht_h_xc_est = AcH_1*xc_est*ArH_1' + lambda*AcH_2*xc_est*ArH_2';
        
        res = b - ht_h_xc_est;
        A_res = AcH_1*res*ArH_1' + lambda*AcH_2*res*ArH_2';
        mu_opt = mean(res(:).*A_res(:))/mean(A_res(:).*A_res(:));
        xc_est = xc_est + mu_opt*res;
        xc_est = max( min(xc_est, 255), 0);
    end
    if k~= Max_iter
        f_xc_est =  Denoiser(xc_est,effective_sigma_coarse);
    end
end

dir_coarse = R'*(xc_est-xc_k)*R;


return
