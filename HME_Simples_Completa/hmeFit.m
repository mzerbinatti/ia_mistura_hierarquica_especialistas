%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Ajusta um modelo HME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hme] = hmeFit(hme,x,y,iter,tol,split,vis,prefix)
% INPUTS
%	hme	HME model with params initialized
%	x	nxd matrix of samples
%	y	nxk matrix of class assignments
%	[iter]	Number of EM iterations.  If a vector, then perform
%		at least min(iter) and at most max(iter) iterations.
%		Default value is [20 50].  A single number is valid.
%	[tol]	Stop EM when log likelihood increases by a fraction
%		less than tol, i.e. (old-new)/old < tol.  
%		Used when a range of iterations is given.  
%		Default value is 1e-4. 
%	[split] Value in (0,1) giving fraction of data to use as
%		the training set.  The remaining data is used as
%		the test set.  Default value is 0.5.
%	[vis]	Visualization level:
%		  0 = none [default]
%		  1 = plot log likelihood vs. iteration
%		  2 = image model details
%	[prefix] Prefix for visualization output files.
%
% OUTPUTS
%	hme	HME model with params fitted

error(nargchk(3,9,nargin));

if nargin < 5, 
	iter = [10 50]; 
end
if nargin < 6, 
	tol = 1e-4; 
end
if nargin < 7, 
	split = 0.5; 
end
if nargin < 8, 
	vis = 1; 
end
if nargin < 9, 
	prefix = ''; 
end

[n,d]= size(x');
[n,k]=size(y');


% split the samples into training and test sets
n1 = round(n*split); 
n2 = n - n1;
if n1 <= 0, 
    error('N�o h� dados para treinamento!'); end
perm = randperm(n);
ind1 = perm(1:n1); 
ind2 = perm(n1+1:n);
x1 = x(:,ind1); 
x2 = x(:,ind2);
y1 = y(:,ind1); 
y2 = y(:,ind2);

L1 = zeros(1,max(iter));
L2 = zeros(1,max(iter));

if vis > 0,
  h = figure;
  set(h,'DoubleBuffer','on');
end

lli = -inf;
save all
for i = 1:max(iter),

    disp(['INICIO da Iteracao ' num2str(i)]); % MICHEL
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Treinamento usando algoritmo EM
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hme = Estep(hme,x1,y1);
    hme = Mstep(hme,x1,y1,ones(1,n1));
    
    if vis > 1,
        % visualiza o estado atual do modelo
        % hmeVis(hme,k,[0 1],[0 1],x1,prefix); % MICHEL
    end
    
    if n2 > 0 && (i > min(iter) | vis > 0),
        % avalia o modelo usando conjunto de valida��o
        [post2,lik2,lli2] = hmeEval(hme,k,x2,y2);
        L2(i) = lli2/n2;
        lli_prev = lli; lli = lli2/n2;
        % disp(sprintf('hme iter=%d lli=%g',i,lli)); % MICHEL
        hme.lli = lli; % MICHEL
    else
        % disp(sprintf('hme iter=%d ',i)); % MICHEL
    end
    
    if vis > 0,
        % avalia o modelo usando conjunto de treinamento
        [post1,lik1,lli1] = hmeEval(hme,k,x1,y1);
        L1(i) = lli1/n1;
        % disp(sprintf('hme iter=%d lli=%g',i,L1(i))); % MICHEL
        % faz o grafico do log likelihood para conjunto de treinamento e
        % valida��o
		% MICHEL % MICHEL % MICHEL
        % figure(h); hold off; plot(L1(1:i),'b-o');
        % if n2 > 0, hold on; plot(L2(1:i),'r-o'); end
        % if n2 > 0,
        %     axis([1 max(iter) 1.01*min([L1(1:i) L2(1:i)]) 0.99*max([L1(1:i) L2(1:i)])]);
        % else
        %     axis([1 max(iter) 1.01*min(L1(1:i)) 0.99*max(L1(1:i))]);
        % end
        % xlabel('iteration'); ylabel('mean log likelihood'); 
        % if n2 > 0, legend('training set','test set',4); end
        % print(h,'-depsc',[prefix 'lli']);
		% MICHEL % MICHEL % MICHEL
    end
    
    if n2 > 0 && i > min(iter),
        % Para o treinamento
        if (lli_prev-lli)/lli < tol, 
            disp(lli_prev)
            break, 
        end 
    end
    if vis==0
        [post1,lik1,lli1] = hmeEval(hme,k,x1,y1);
        L1 = lli1/n1;
        if abs(L1)<1e-9,
            disp(sprintf('FINALIZANDO LIKELIHOOD = %1.5g',L1))
            break,
        end
    end
end



if vis == 1,
  % visualize the current state of the model
  % hmeVis(hme,k,[0 1],[0 1],x1,prefix);  % MICHEL
end



