%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Cria uma HME como arvore balanceada
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hme] = hmeCreate(levels,b,d,k, arquitetura) % MICHEL_ARQUIT
% INPUTS
%	levels	number of levels (>=0) in hierarchy
%	b	    branching factor
%	d	    dimensionality of data
%	k	    number of outputs
%
% OUTPUTS
%	hme     HME model

global index_gating

ind_arq = index_gating; 
b = arquitetura(ind_arq);
index_gating = ind_arq + 1;
  
if levels == 0 | b <= 1,
    hme = hmeCreateExpert(d,k); % cria os especialistas
else
    
  children = cell(1,b); %cria os filhos   
    
  for i = 1:b,
    if (levels-1)==0
        % disp(sprintf('\n*******************************************')) % MICHEL
        % disp(sprintf('Rede Especialista do ramo %d',i))				% MICHEL
        % disp(sprintf('*******************************************')) % MICHEL
    end
    children{i} = hmeCreate(levels-1,b,d,k, arquitetura); % MICHEL_FOLHAS
    
  end
  % disp(sprintf('\n*******************************************'))% MICHEL
  % disp(sprintf('Rede Gating do nivel %d',levels))				% MICHEL
  % disp(sprintf('*******************************************')) % MICHEL
  hme = hmeCreateMixture(children,d);
end