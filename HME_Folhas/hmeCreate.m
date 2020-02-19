%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Cria uma HME como arvore balanceada
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hme] = hmeCreate(levels,b,d,k, folhas) % MICHEL_FOLHAS
% INPUTS
%	levels	number of levels (>=0) in hierarchy
%	b	    branching factor
%	d	    dimensionality of data
%	k	    number of outputs
%
% OUTPUTS
%	hme     HME model

global index_gating

if levels == 0,
    hme = hmeCreateExpert(d,k); % cria os especialistas
else
  
  
  if (levels-1)==0
    % EX.: em uma estrutura 2x3, uma arvore completa terá 9 especialistas(FOLHAS) ou seja [3 , 3 , 3] 
    % O Vetor informará quantas folhas cada gating do nivel final terá:
    % Ex.: [2,3,2] informará que o Gating 1 tera 2 especialistas, o Gating 2 terá 3 e o Gating 3 terá 2
    % Quantidade de EXPERTS
    ind_fol = index_gating;
    index_gating = ind_fol + 1;
    b = folhas(ind_fol);
    children = cell(1,b); %cria os filhos   
  else
    children = cell(1,b); %cria os filhos
  end
  
  for i = 1:b,
    if (levels-1)==0
        % disp(sprintf('\n*******************************************')) % MICHEL
        % disp(sprintf('Rede Especialista do ramo %d',i))				% MICHEL
        % disp(sprintf('*******************************************')) % MICHEL
    end
    children{i} = hmeCreate(levels-1,b,d,k, folhas); % MICHEL_FOLHAS
    
  end
  % disp(sprintf('\n*******************************************'))% MICHEL
  % disp(sprintf('Rede Gating do nivel %d',levels))				% MICHEL
  % disp(sprintf('*******************************************')) % MICHEL
  hme = hmeCreateMixture(children,d);
end