function [hme]= hmeCreateExpert(d,k,param)
% Cria um modelo HME consistindo de um especialista
%
% INPUTS
%	d	dimensionality of data
%	k	number of classes
%	[param]	dxk matrix of model coefficients
%
% OUTPUTS
%	hme	HME model

hme.leaf = 1;

% disp(sprintf('Escolha o modelo do especialista')) % MICHEL
% disp(sprintf('[1]    - Especialista lineares')) % MICHEL
% disp(sprintf('[2]    - Redes Feedforward')) % MICHEL
%type_expert=input('Escolha a opcao desejada  ');
type_expert=2; % MICHEL

if type_expert==1
    hme.param = rand(d,k);
    hme.type='linear';
    
elseif type_expert==2
    % h=input('\nDigite o numero de neuronios para cada Rede especialista  ');
    h=3; % MICHEL
    hme.param.A = rand(h,d); % Ja tem bias
    hme.param.B = rand(k,h+1);
    hme.param.sig =1;
    hme.type='mlp';
    hme.h=h;
else
    disp('Não configurado')
end