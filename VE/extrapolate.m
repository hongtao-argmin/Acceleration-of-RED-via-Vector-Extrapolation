% Extrapolate iterative method F using specified method.
% Use L cycles of k-order extrapolation, starting from x0.
% ======================================================

% x0: the starting point
% Q: the collection matrix contains successively k+1 values.

function [x0] = extrapolate(x0,k,Q,method) %F, k, L,residuals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N = numel(x0);
% Q = zeros(N, k+1);
if size(Q,2)~=k+1
    error('The dimension in Q is not enough\n')
end

switch upper(method)
    case 'MPE', method = @mpe;
    case 'RRE', method = @rre;
    case 'SVDMPE', method = @svdmpe;
    case 'N/A', method = '';
    case '', warning('Extrapolate:PlainIteration', 'No extrapolation');
    otherwise, error('Extrapolate:UnknownMethod', method);
end
% Perform L cycles of extrapolation method
% residuals = zeros(L, 1);
% for t = 1:L
% Compute (k+1) vectors, in addition to x0
%     Q(:, 1) = F(x0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     residuals(t) = norm(x0 - Q(:, 1), 2);
%     for i = 1:k
%         Q(:, i+1) = F( Q(:, i) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     end
if isempty(method)  % No extrapolation.
    x0 = Q(:, end); % Just take the last vector.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         continue;
    return
end
% Compute differences (k+1)

% for i = k:-1:1
%     Q(:, i+1) = Q(:, i+1) - Q(:, i);
% end
Q(:, 2:end) = diff(Q,1,2);

Q(:, 1) = Q(:, 1) - x0;
% Perform QR decomposition
[Q, R] = MGS(Q);
% Perform extrapolation
[gamma] = method(R, k); % s.t. x0 = X * gamma
xi = 1 - cumsum(gamma(1:k)); % s.t. x0' = x0 + U * xi
%     [Q, R] = MGS(Q_temp);
eta = R(1:k, 1:k) * xi(:); % since U = Q * R
x0 = x0 + Q(:, 1:k) * eta; % s.t. x0' = x0 + Q * R * xi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end
% end
return
% Minimal Polynomial Extrapolation


function gamma = mpe(R, k) %, residual
c = backsub(R(1:k, 1:k), -R(1:k, k+1));
c = [c; 1];
gamma = c / sum(c);
% residual = abs(gamma(end)) * R(end, end);
return

% Reduced Rank Extrapolation
function gamma = rre(R, k)
e = ones(k+1, 1);
d = backsub(R, backsub(R', e));
lambda = 1 / sum(d);
gamma = lambda * d;
% residual = sqrt(lambda);
return

function gamma = svdmpe(R, k) %, residual
[~,~,H] = svd(full(R));
c = H(:,k+1);
gamma = c / sum(c);
return


