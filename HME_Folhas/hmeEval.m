function [yexp,lik,lli] = hmeEval(hme,k,x,y)
%
% Avalia o modelo dado x e y.
%
% INPUTS
%	hme	HME model
%	k	number of classes
%	x	dxn matrix of samples
%	[y]	kxn vector of class memberships
%
% OUTPUTS
%	yexp	kxn matrix of output
%	lik	1xn vector of sample likelihoods (if y is given)
%	lli	log likelihood (if y is given)
%


[n,d] = size(x');

if hme.leaf,
   % evaluate expert
   if nargin > 3,
        switch lower (hme.type)
            case 'linear'
                 [yexp,lik,lli] = logistexpert_eval(hme.param,x,y);
            case 'mlp'
                 [yexp,lik,lli] = mlpexpert_eval(hme.param,x,y);
        end
    else
        switch lower(hme.type)
            case 'linear'
                yexp = logistexpert_eval(hme.param,x);
            case 'mlp'
                yexp = mlpexpert_eval(hme.param,x);
        end    
    end
    
else
    
    % evaluate gating function
    switch lower(hme.type)
        case 'linear'
            gate = logistK_eval(hme.param,x);
        case 'mlp'
            gate=mlpk_eval(hme.param,x);            
        case 'localized'
            disp('Nao implementado')
    end
    
    % combine yexp,lik from children using gating function
    yexp = zeros(k,n);
    lik = zeros(1,n);
    
    for i = 1:length(hme.children),
        if nargin > 3,
            [yexp_i,lik_i] = hmeEval(hme.children{i},k,x,y);
            lik = lik + lik_i .* gate(i,:);
        else
            yexp_i = hmeEval(hme.children{i},k,x);
        end
        yexp = yexp + yexp_i .* repmat(gate(i,:),k,1); %Calcula a saida da mistura
    end
    
    % log-likelihood
    if nargin > 3,
        lli = sum(log(lik+eps));
    end    
end

