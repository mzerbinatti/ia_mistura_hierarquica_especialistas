function [hme] = hmeInitLR(hme,x,y)
% [hme] = hmeInitLR(hme,x,y)
%
% Initialize an HME model by fitting the logistic functions in the
% gating network and expert nodes in a top-down manner.  This only
% makes sense if the branching factor of the tree is equal to the
% number of classes.
%
%	hme	HME model to initialize.
%	x	dxn matrix of samples
%	y	kxn matrix of class assignments
% 

[d,n] = size(x);
hme = initTree(hme,x,y,ones(1,n));

function [hme] = initTree(hme,x,y,w)
  
  if hme.leaf,	% expert

    switch lower(hme.type)
        case 'linear'
            hme.param = logistexpert(x,y,w);
        case 'mlp'            
            hme.param = mlpexpert(x,y,w,hme.h);
        case 'Gaussiano'
            disp('Nao implementado')
    end

  else	% gating node
    [k,n] = size(y);
    [d,b] = size(hme.param);
    if b == k,
        switch lower (hme.type)
            case 'linear'
                [hme.param,post] = logistK(x,y,w);
            case 'mlp'
                [hme.param,post] = mlpK(x,y,w,hme.h);
            case 'localized'
                disp('Nao implementado')
        end                                
      for i = 1:length(hme.children),
        wi = w .* post(i,:);
        hme.children{i} = initTree(hme.children{i},x,y,wi);
      end
    else
      warning(sprintf('Can''t initialize gating node since (k=%d) != (b=%d).',k,b));
    end
    
  end
  
