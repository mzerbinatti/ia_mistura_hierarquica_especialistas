function [bx,lik,lli] = mlpexpert_eval(param,x,y)
% INPUT
% 	param 	parameters
% 	x 	    dxn matrix of n input column vectors
% 	[y] 	kxn vector of class assignments
%
% OUTPUT
% 	y 	    kxn matrix
% 	lik 	1xn vector of sample likelihoods


if nargin > 3 && size(y,2) ~= size(x,2), 
  error('Inputs x,y not the same length.'); 
end

k=size(param.B,1); % Numero de saida
[d,n] = size(x); 

bx=perceptrons(x',param.A,param.B);
bx=bx';
var  = param.sig;

% likelihood de cada exemplo
if nargout > 1,
    for j=1:n
        diff = y(:,j)-bx(:,j);
        lik(1,j)=exp(-diff*diff'/(2*var));
    end
end

% total log likelihood
if nargout > 2,
  lli = sum(log(lik+eps));
end;
