%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Cria um modelo HME consistindo de uma mistura de um
%%% conjunto de modelos HME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hme] = hmeCreateMixture(children,d,param)

% INPUTS
% 	children 	1xb cell array of HME models
%	d		dimensionality of data
%	[param]		dxb matrix of mixing model coefficients
%
% OUTPUTS
%	hme		HME model
%

b = length(children);

hme.leaf = 0;

% disp(sprintf('Modelos de Rede Gating')) % MICHEL
% disp(sprintf('[1] - Gating lineares')) % MICHEL
% disp(sprintf('[2] - Redes Feedforward')) % MICHEL
% disp(sprintf('[3] - Kernels Normalizados')) % MICHEL
% type_gating=input('Escolha a opcao desejada  ');
type_gating=2; % MICHEL

if type_gating==1
    hme.param = rand(d,b);
    hme.type=linear;

elseif type_gating==2
    % nh=input('\nDigite o numero de neuronios para cada Rede Gating  ');
    nh=3; % MICHEL
    hme.param.A = rand(nh,d);
    hme.param.B = rand(b,nh+1);
    hme.type='mlp';
    hme.h=nh;

elseif type_gating==3
    hme.param.m=[];
    hme.param.sigma=0;
    hme.type='localized';
end

hme.children = children;