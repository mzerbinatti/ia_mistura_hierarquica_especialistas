%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Autor: Clodoaldo Aparecido de Moraes Lima
%%%Disciplina: Inteligencia Computacional
%%%Especialista Linear
%%%             Mlp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-------------------------------------------------------------------------
% Definicao dos parametros
%--------------------------------------------------------------------------
%disp(sprintf('************************************************************'))
%disp(sprintf('\tMistura Hierarquica de Especialista'))
%disp(sprintf('************************************************************'))
% MICHEL
function [hme_fit, yexp, z]= hmeTest(alturamax, larguramax, arquitetura) 
    depth=2; % Profundidade
    factor=2; % fator de ramificação
    nexperts=factor; % fator de ramificação

    % disp(sprintf('Carregando os dados'));
    load dados2
    plot(Y)
    X = load('serie1_lag.txt')
    Y = load('serie1_lag_y.txt')
    plot(Y)
    global index_gating;
    index_gating = 1;
    % arquitetura = [2, 2, 0, 0, 2, 0, 0] % OK
    % arquitetura = [2, 2, 0, 0, 0] % OK
    % arquitetura = [0] % OK
    % arquitetura
    
    [N,dim]=size(X); %J� com bias
    [N,k]=size(Y);
    
    hme = hmeCreate(depth,nexperts,dim,k, arquitetura ); %cria a estrutura
    hme = hmeInitRand(hme,depth,nexperts,dim,k);%Inicializa a estrutura
    hme_fit = hmeFit(hme,X',Y',10,0,0.7,0); %realiza treinamento
    yexp = hmeEval(hme_fit,k,X'); %Realiza o teste
    z=perceptrons(X,hme_fit.param.A,hme_fit.param.B);
    % z=exp(z)./[sum(exp(z),2)*[1 1]];
    % z=exp(z)./[sum(exp(z),2)* ones(1, factor)];
    % subplot(3,1,1)
    % plot(Y)
    % hold on
    % plot(yexp,'r')
    % subplot(3,1,2)
    % plot(z(:,1),'linewidth',2)
    % subplot(3,1,3)
    % plot(z(:,2),'linewidth',2)

endfunction

a=hmeTest(2,3, [2, 2, 0, 0, 0]);
