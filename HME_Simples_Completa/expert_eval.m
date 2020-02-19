function [post,lik,lli] = expert_eval(param,x,y)
% INPUT
% 	param 	parameters
% 	x 	    dxn matrix of n input column vectors
% 	[y] 	kxn vector of class assignments
%
% OUTPUT
% 	post 	kxn fitted class posteriors
% 	lik 	1xn vector of sample likelihoods
%	lli	log likelihood
%

if nargin > 3 && size(y,2) ~= size(x,2), 
  error('Inputs x,y not the same length.'); 
end

k=size(param.B,1); % Numero de saida
[d,n] = size(x); 

% class posteriors
post = zeros (k,n); 

bx=perceptrons(x',param.A,param.B);
bx=bx';

for j = 1:k, 
    post(j,:) = 1 ./ sum(exp(bx - repmat(bx(j,:),k,1)),1);
end
clear bx;


% likelihood of each sample
if nargout > 1,
  y = y ./ repmat(sum(y,1),k,1); % L1-normalize class assignments  
  lik = prod(post.^y,1);
end

% total log likelihood
if nargout > 2,
  lli = sum(log(lik+eps));
end;
