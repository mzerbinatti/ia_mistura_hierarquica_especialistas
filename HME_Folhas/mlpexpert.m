function [param,yexp,lli] = mlpexpert(x,y,w,h,param)

% k = number of classes
% n = number of samples
% d = dimensionality of samples
%
% INPUT
% 	x 	dxn matrix of n input column vectors
% 	y 	kxn vector of class assignments
% 	[w]	1xn vector of sample weights
%	[beta]	dxk matrix of model coefficients
%
% OUTPUT
% 	param 	dxk matrix of fitted model coefficients
% 	post 	kxn matrix of fitted class posteriors
% 	lli 	log likelihood

error(nargchk(2,5,nargin));

debug = 0;
if debug>0,
    h=figure(1);
    set(h,'DoubleBuffer','on');
end

% get sizes
[d,nx] = size(x);
[k,ny] = size(y);

% check sizes
if nx ~= ny,
    error('Inputs x,y not the same length.');
end

n = nx;


% if sample weights weren't specified, set them to 1
if nargin < 3,
    w = ones(1,n);
end

% if starting beta wasn't specified, initialize randomly

if nargin < 5,
    param.A=rand(h,d)/5;
    param.B=rand(k,h+1)/5;
    param.sig =1;
end

stepsize = 1;

minstepsize = 1e-10;

yexp = perceptrons(x',param.A, param.B);
wn = w/param.sig;
lli = computeLogLik(yexp',y,param.sig,wn);

% for iter = 1:1000,
for iter = 1:500, % MICHEL
    
    %vis(x,y,beta,lli,d,k,iter,debug);
    
    % gradient
    [dJdA,dJdB]=grad_expert(x',y',wn',param.A,param.B,h);
    
    % Transforma para vetor
    vetdJdA=reshape(dJdA,1,h*(d))';
    vetdJdB=reshape(dJdB,1,k*(h+1))';
    
    % Monta o vetor gradiente
    g=[vetdJdA;vetdJdB];
    
    % Normaliza o vetor gradiente
    %g=g/norm(g);
    
    % Transforma em vetor
    vetdJdA=g(1:h*(d));
    vetdJdB=g(h*(d)+1:end);
    
    % Transforma em matriz
    dJdA=reshape(vetdJdA',h,d);
    dJdB=reshape(vetdJdB',k,h+1);
    
    % Atualiza a matriz A e B
    param2.A=param.A-stepsize*dJdA;
    param2.B=param.B-stepsize*dJdB;
    
    lli_prev = lli;
    
    yexp2 = perceptrons(x',param2.A,param2.B);
    lli2 = computeLogLik(yexp2',y,param.sig,wn);
    
    while lli>lli2
        % Atualiza a matriz A e B
        param2.A=param.A-stepsize*dJdA;
        param2.B=param.B-stepsize*dJdB;
        
        % calcula log likelihood
        yexp2 = perceptrons(x',param2.A,param2.B);
        lli2 = computeLogLik(yexp2',y,param.sig,wn);
        
        stepsize = 0.5 * stepsize;
    end
    lli=lli2; % MICHEL
    param.A=param2.A;
    param.B=param2.B;
    yexp=yexp2';
    stepsize=2*stepsize;
    
    %     % stop if the log likelihood has decreased; this shouldn't happen
    if lli < lli_prev,
        warning(['Stopped at iteration ' num2str(iter) ...
            ' because the log likelihood decreased from ' ...
            num2str(lli_prev) ' to ' num2str(lli) '.' ...
            ' This may be a bug.']);
        break
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Atualiza o sigma
diff = yexp - y;
soma = 0;
for i=1:n
    soma = soma + w(1,i)*(diff(:,i)'*diff(:,i));
end
% disp('Atualizando sig')
param.sig = max(0.05,(1/k)*soma/sum(w));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Calcula likelihood
wn = w/param.sig;
lli = computeLogLik(yexp2',y,param.sig,wn);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if debug>0, 
  vis(x,y,param,lli,d,k,iter,2); 
end

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% log likelihood
function lli = computeLogLik(yexp,y,sig,w)
  [k,n] = size(yexp);
  lli = 0;
  for j = 1:n,
    diff = yexp(:,j)-y(:,j);
    lik(1,j)=exp(-diff*diff'/(2*sig));
  end
  lli = sum(w.*log(lik+eps));
  if isnan(lli), 
    error('lli is nan'); 
  end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% visualization
function vis (x,y,beta,lli,d,k,iter,debug)
  debug = 0; % MICHEL
  if debug<=0, return, end

  disp(['iter=' num2str(iter) ' lli=' num2str(lli)]);
  if debug<=1, return, end

  if d~=3 | k>10, return, end

  figure(1);
  res = 100;
  r = abs(max(max(x)));
  dom = linspace(-r,r,res);
  [px,py] = meshgrid(dom,dom);
  xx = px(:); yy = py(:);
 
  points = [xx' ; yy' ; ones(1,res*res)];
  
  func = zeros(k,res*res);
  %for j = 1:k,
  %  func(j,:) = exp(beta(:,j)'*points);
  %end
  func=perceptrons(points',beta.A,beta.B);
  func=func';
  
  [mval,ind] = max(func,[],1);
  hold off; 
  im = reshape(ind,res,res);
  imagesc(xx,yy,im);
  hold on;
  syms = {'w.' 'wx' 'w+' 'wo' 'w*' 'ws' 'wd' 'wv' 'w^' 'w<'};
  for j = 1:k,
    [mval,ind] = max(y,[],1);
    ind = find(ind==j);
    plot(x(1,ind),x(2,ind),syms{j});
  end
  

% eof
