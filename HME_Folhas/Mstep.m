function [hme] = Mstep(hme,x,y,w)

if hme.leaf,
    
    % train expert
    switch lower(hme.type)
        case 'linear'
            hme.param = logistexpert(x,y,w,hme.param);
        case 'mlp'
            hme.param = mlpexpert(x,y,w,hme.h,hme.param);
    end
else    
    % train filhos
    for i = 1:length(hme.children),
        wi = w .* hme.clik(i,:);
        hme.children{i} = Mstep(hme.children{i},x,y,wi);
    end
    % treina rede gating
    switch lower (hme.type)
        case 'linear'
            hme.param = logistK(x,hme.clik,w,hme.param);
        case 'mlp'
            hme.param = mlpK(x,hme.clik,w,hme.h,hme.param);
        case 'localized'
            hme.param = localizedK(x,hme.clik,w,hme.param);
    end
end

% retirar os campos Estep
if hme.leaf,
    hme = rmfield(hme,{'lik'});
else
    hme = rmfield(hme,{'lik','clik'});
end