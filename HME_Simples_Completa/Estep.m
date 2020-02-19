%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Estima o h
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hme] = Estep(hme,x,y)


if hme.leaf,
    % calcula a likelihood do especialista
    switch lower (hme.type)
        case 'linear'  %modelo linear
            [yexp,lik] = logistexpert_eval(hme.param,x,y);
        case 'mlp'     %redes neurais
            [yexp,lik]=mlpexpert_eval(hme.param,x,y);
    end
    hme.lik = lik;
else    
    % Calcula o likelihood dos filhos
    for i = 1:length(hme.children),
        hme.children{i} = Estep(hme.children{i},x,y);
    end
    
    % avalia a rede gating
    switch lower (hme.type)
        case 'linear'
            gate = logistK_eval(hme.param,x); % bxn matrix
        case 'mlp'
            gate = mlpk_eval(hme.param,x); % bxn matrix
        case 'localized'
            gate = localizedK_eval(hme.param,x); % bxn matrix
    end
    % calcula o likelihood do nó  e normaliz gated
    % likelihood
    hme.clik = zeros(size(gate));
    
    for i = 1:length(hme.children),
        hme.clik(i,:) = gate(i,:).*hme.children{i}.lik;
    end
    hme.lik = sum(hme.clik,1);
    
    for i = 1:length(hme.children),
        hme.clik(i,:) = hme.clik(i,:) ./ (hme.lik+eps);
    end
end