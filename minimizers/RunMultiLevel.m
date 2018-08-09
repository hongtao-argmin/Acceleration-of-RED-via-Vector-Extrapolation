% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% https://www.apache.org/licenses/LICENSE-2.0
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

% test code assume the image is even and circular boundary condiiton is
% used.
function [im_out, psnr_out,fun_val_set,CPU_time_set] = RunMultiLevel(y,ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params,orig_im,psf,psf_center)

% print infp every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end

%%%%%%% -----------------------------------------------------------------
% formulate kroneck decomposition
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

%--------------------------------------------------------------------------
% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
inner_iters = params.inner_iters;
effective_sigma = params.effective_sigma;
effective_sigma_c = effective_sigma/2;  % it may become half of effective_sigma.

% initialization
x_est = InitEstFunc(y);
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
% outer iterations
x_k_cc = zeros(size(x_est));
eta = 1e-2;
alpha = 1;
isToCoarse = true;

%%%% preprocess
mu = 2/(1/(input_sigma^2) + lambda);
f_x_est = Denoiser(x_est, effective_sigma);
% update the solution
grad1 = BackwardFunc(ForwardFunc(x_est) - y)/(input_sigma^2);
grad2 = lambda*(x_est - f_x_est);
x_est = x_est - mu*(grad1 + grad2);
% project to [0,255]
x_est = max( min(x_est, 255), 0);
Max_iter = 3;
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

% --------------------------------------------------------
x_est = x_k_cc;

for k = 1:1:outer_iters
    
    % apply the denoising engine
    f_x_est = Denoiser(x_est, effective_sigma);
    
    % solve Az = b by a variant of the SD method, where
    % A = 1/(sigma^2)*H'*H + lambda*I, and
    % b = 1/(sigma^2)*H'*y + lambda*denoiser(x_est)
    if isfield(params,'use_fft') && params.use_fft == true
        %     for kkk=1:2
        b = fft_Ht_y + lambda*fft2(f_x_est);
        A = fft_HtH + lambda;
        x_est = real(ifft2( b./A ));
        x_est = max( min(x_est, 255), 0);
    else
        for j=1:1:inner_iters
            b = Ht_y + lambda*f_x_est;
            ht_h_x_est = BackwardFunc(ForwardFunc(x_est))/(input_sigma^2);
            res = b - ht_h_x_est - lambda*x_est;
            A_res = BackwardFunc(ForwardFunc(res))/(input_sigma^2) + lambda*res;
            mu_opt = mean(res(:).*A_res(:))/mean(A_res(:).*A_res(:));
            x_est = x_est + mu_opt*res;
            x_est = max( min(x_est, 255), 0);
        end
        %         CostFunc(y, x_est, ForwardFunc, input_sigma,...
        %             lambda, effective_sigma)
        
        % decide whether goes to coarse
        %         if mod(k,2) == 0
        %         if alpha>=2e-1 && isToCoarse %&& mod(k,3) == 0
        %             Max_iter = 10;
        % %             CostFunc(y, x_est, ForwardFunc, input_sigma,...
        % %                 lambda, effective_sigma)
        %             %             dir_coarse = ...
        %             %                 coarse_solver_v1(x_est,y,ForwardFunc,BackwardFunc,input_sigma,lambda,Ac,Ar,...
        %             %                 effective_sigma,Max_iter,inner_iters,f_x_est);
        %             dir_coarse = ...
        %                 coarse_solverFAS(x_est,y,ForwardFunc,...
        %                 BackwardFunc,input_sigma,lambda,Ac,Ar,effective_sigma,Max_iter,inner_iters);
        %             %   coarse_solver(x_est,Ht_y,ForwardFunc,BackwardFunc,input_sigma,lambda,Ac,Ar,effective_sigma/4,Max_iter,inner_iters);
        %             x_k_cc = x_est + alpha*dir_coarse;
        %             alpha = alpha/2;
        %             x_k_cc = max( min(x_k_cc, 255), 0);
        % %             CostFunc(y, x_k_cc, ForwardFunc, input_sigma,...
        % %                 lambda, effective_sigma)
        %             x_est = x_k_cc;
        %             isToCoarse = false;
        %         end
        %         end
        
        %         CostFunc(y, x_est + dir_coarse, ForwardFunc, input_sigma,...
        %             lambda, effective_sigma)
    end
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);
    end
end

im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
psnr_out = ComputePSNR(orig_im, im_out);

return


% FAS -- full approximation scheme
% This is a good version
% future version should also consider the design of prolongation and
% restriction.

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


% the code can be optimized to same more computations
% the main concern is to reduce the indifferent denoising.
function dir_coarse = coarse_solver_v1(x_k,y,ForwardFunc,BackwardFunc,input_sigma,lambda,Ac,Ar,effective_sigma,Max_iter,inner_iters,f_x_est)
% assume the dimension of the image is even and square
effective_sigma_coarse = effective_sigma/2;
% the size of the image shoule be even.
[im_row,~] = size(x_k);
M = im_row/2;

% Compute the coarse col and row matrix
R = ones(1,2);
R = kron(speye(M,M),R);
xc_k = R*x_k*R'/4;
xc_est = xc_k;
% Galerkin Coarsing
Ac_H = R*Ac*R'/2;
Ar_H = R*Ar*R'/2;
yc = R*y*R'/4;

Ht_y = Ac_H'*yc*Ar_H/((input_sigma/2)^2);

f_x_est = Denoiser(x_k,effective_sigma);
nDh = (BackwardFunc(ForwardFunc(x_k))-BackwardFunc(y))/(input_sigma^2)+lambda*(x_k-f_x_est);

nD_h = R*nDh*R'/4;
nDH = (Ac_H'*(Ac_H*xc_est*Ar_H')*Ar_H-Ac_H'*yc*Ar_H)/((input_sigma/2)^2)+lambda*(xc_est-Denoiser(xc_est,effective_sigma_coarse));

kappa = 0.49;
if (norm(nDH,'fro')/norm(nDh,'fro') <= kappa)
    fprintf('close coarse correction\n')
    dir_coarse = zeros(size(x_k));
    return;
end

v = nD_h - nDH;
% contain 1/sigma^2 and lambda
for k = 1:Max_iter
    f_xc_est =  Denoiser(xc_est,effective_sigma_coarse);
    for j=1:1:inner_iters
        b = Ht_y - v + lambda*f_xc_est;
        ht_h_xc_est = (Ac_H'*(Ac_H*xc_est*Ar_H')*Ar_H)/((input_sigma/2)^2);
        res = b - ht_h_xc_est - lambda*xc_est;
        A_res = (Ac_H'*(Ac_H*res*Ar_H')*Ar_H)/((input_sigma/2)^2) + lambda*res;
        mu_opt = mean(res(:).*A_res(:))/mean(A_res(:).*A_res(:));
        xc_est = xc_est + mu_opt*res;
        xc_est = max( min(xc_est, 255), 0);
    end
end

dir_coarse = R'*(xc_est-xc_k)*R; %kron(xc_est-x_c_k,ones(2,2));

% eta = 1e-5;
% if norm(dir_coarse,'fro')/numel(dir_coarse)<eta
%     dir_coarse = zeros();
% end
return


% this formulation has some problems
function dir_coarse = coarse_solver_v2(x_k,y,ForwardFunc,BackwardFunc,input_sigma,lambda,Ac,Ar,effective_sigma,Max_iter,inner_iters)

% this downsample and prolongation is replicate, not bilinear and
% full-weighting
effective_sigma_coarse = effective_sigma/2;

% assume the dimension of the image is even and square
[im_row,~] = size(x_k);
M = im_row/2;

R = ones(1,2);
R = kron(speye(M,M),R);

% scale down the image
xc_k = R*x_k*R'/4;
xc_est = xc_k;

% Galerkin Coarsing
Ac_HP = Ac*R';
Ar_HP = Ar*R';

% contain 1/sigma^2 and lambda
x_k_bar = x_k - R'*(R*x_k*R'/4)*R;

f = (R*BackwardFunc(ForwardFunc(x_k_bar))*R' - R*BackwardFunc(y)*R')/(input_sigma^2)+lambda*R*x_k_bar*R';
f = -f;

for k = 1:Max_iter
    f_x_est =  Denoiser(x_k_bar + R'*xc_est*R,effective_sigma);
    
    for j = 1:1:inner_iters
        
        b = f + lambda*R*f_x_est*R';
        ht_h_xc_est = (Ac_HP'*(Ac_HP*xc_est*Ar_HP')*Ar_HP)/(input_sigma^2);
        res = b - ht_h_xc_est - lambda*R*(R'*xc_est*R)*R'/(input_sigma^2);
        A_res = (Ac_HP'*(Ac_HP*res*Ar_HP')*Ar_HP)/(input_sigma^2) + lambda*R*(R'*res*R)*R'/(input_sigma^2);
        mu_opt = mean(res(:).*A_res(:))/mean(A_res(:).*A_res(:));
        xc_est = xc_est + mu_opt*res;
        xc_est = max( min(xc_est, 255), 0);
        
    end
end

dir_coarse = R'*(xc_est-xc_k)*R;

return





% bilinear prolongation and full-weighting restriction
% function fc = Restrict(r)              % Restriction to coarser grid using
% % full-weighted local averaging
% fc = r(1:2:end,1:2:end);
% fc(2:end-1,2:end-1) = 0.25*(fc(2:end-1,2:end-1) + ...
%     0.5*(r(2:2:end-3,3:2:end-2) + r(4:2:end-1,3:2:end-2) + ...
%     r(3:2:end-2,2:2:end-3) + r(3:2:end-2,4:2:end-1)) + ...
%     0.25*(r(2:2:end-3,2:2:end-3) + r(2:2:end-3,4:2:end-1) + ...
%     r(4:2:end-1,2:2:end-3) + r(4:2:end-1,4:2:end-1)));
%
% return
%
% function u = Prolong(uc)                % Bi-linear interpolation
%
% u = zeros(length(uc)*2-1);
% u(1:2:end,1:2:end) = uc;
% u(2:2:end-1, 1:2:end) = 0.5*(u(1:2:end-2,1:2:end) + u(3:2:end,1:2:end));
% u(1:end,2:2:end-1) = 0.5*(u(1:end,1:2:end-2) + u(1:end,3:2:end));
%
% return