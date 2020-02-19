function [beta,post,lli] = mlpK(x,y,w,h,beta)

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
% 	beta 	dxk matrix of fitted model coefficients 
%		(beta(:,k) are fixed at 0) 
% 	post 	kxn matrix of fitted class posteriors
% 	lli 	log likelihood
%
% Let p(i,j) = exp(beta(:,j)'*x(:,i)),
% Class j posterior for observation i is:
%	post(j,i) = p(i,j) / (p(i,1) + ... p(i,k))
%
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
if k < 2,
  error('Input y must encode at least 2 classes.');
end
if nx ~= ny,
  error('Inputs x,y not the same length.'); 
end

n = nx;

% make sure class assignments have unit L1-norm
sumy = sum(y,1);
if abs(1-sumy) > eps,
  sumy = sum(y,1);
  for i = 1:k, y(i,:) = y(i,:) ./ sumy; end
end
clear sumy;

% if sample weights weren't specified, set them to 1
if nargin < 3, 
  w = ones(1,n);
end

% normalize sample weights so max is 1
w = w / max(w);

% if starting beta wasn't specified, initialize randomly

if nargin < 5,
    beta.A=rand(h,d)/5;
    beta.B=rand(k,h+1)/5;
    % beta = 1e-3*rand(d,k);
    % beta(:,k) = 0;	% fix beta for class k at zero
else
 % if sum(beta(:,k)) ~= 0,
 %   error('beta(:,k) ~= 0');
 % end
end

stepsize = 1;
minstepsize = 1e-10;

post = computePost(beta,x);
lli = computeLogLik(post,y,w);

%figure(1)
%clf
for iter = 1:1000,
   
   % pause
    %disp(sprintf('  logist iter=%d lli=%g',iter,lli));
    %vis(x,y,beta,lli,d,k,iter,debug);
    
    % gradient and hessian
    %[g,h] = derivs(post,x,y,w);
    [dJdA,dJdB]=grad_expert_class(x',y',w',beta.A,beta.B,h);
    
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
    beta2.A=beta.A-stepsize*dJdA;
    beta2.B=beta.B-stepsize*dJdB;
    
    lli_prev = lli;
    
    post2 = computePost(beta2,x);
    lli2 = computeLogLik(post2,y,w);
     
       
    while lli>lli2
       % Atualiza a matriz A e B
        beta2.A=beta.A-stepsize*dJdA;
        beta2.B=beta.B-stepsize*dJdB;
        
        % get the new log likelihood
        post2 = computePost(beta2,x);
        lli2 = computeLogLik(post2,y,w);
    
        stepsize = 0.5 * stepsize;
    end
    lli=lli2;
    post=post2;
    beta=beta2;
    stepsize=2*stepsize;

    % stop if the average log likelihood has gotten small enough
    %if 1-exp(lli/n) < 1e-2, break, end
    
    % stop if the log likelihood changed by a small enough fraction
    %dlli = (lli_prev-lli) / lli;
    %if abs(dlli) < 1e-3, break, end
    
    % stop if the step size has gotten too small
    %if stepsize < minstepsize, break, end
    
    % stop if the log likelihood has decreased; this shouldn't happen
    if lli < lli_prev,
        warning(['Stopped at iteration ' num2str(iter) ...
                ' because the log likelihood decreased from ' ...
                num2str(lli_prev) ' to ' num2str(lli) '.' ...
                ' This may be a bug.']);
        break
    end
end

if debug>0, 
  vis(x,y,beta,lli,d,k,iter,2); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% class posteriors
function post = computePost(beta,x)
  [d,n] = size(x);
  %  [d,k] = size(beta);
  k=size(beta.B,1);  
  post = zeros(k,n);
  bx = zeros(k,n);
  %  for j = 1:k, 
  %    bx(j,:) = beta(:,j)'*x; 
  %  end
  bx=perceptrons(x',beta.A,beta.B);
  bx=bx'; % Realiza transposicao

  for j = 1:k, 
    post(j,:) = 1 ./ sum(exp(bx - repmat(bx(j,:),k,1)),1);
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% log likelihood
function lli = computeLogLik(post,y,w)
  [k,n] = size(post);
  lli = 0;
  for j = 1:k,
    lli = lli + sum(w.*y(j,:).*log(post(j,:)+eps));
  end
  if isnan(lli), 
    error('lli is nan'); 
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% gradient and hessian
%% These are computed in what seems a verbose manner, but it is
%% done this way to use minimal memory.  x should be transposed
%% to make it faster.
function [g,h] = derivs(post,x,y,w)

  [k,n] = size(post);
  [d,n] = size(x);

  % first derivative of likelihood w.r.t. beta
  g = zeros(d,k-1);
  for j = 1:k-1,
    wyp = w .* (y(j,:) - post(j,:));
    for ii = 1:d, 
      g(ii,j) = x(ii,:) * wyp'; 
    end
  end
  g = reshape(g,d*(k-1),1);

  % hessian of likelihood w.r.t. beta
  h = zeros(d*(k-1),d*(k-1)); 
  for i = 1:k-1,	% diagonal
    wt = w .* post(i,:) .* (1 - post(i,:));
    hii = zeros(d,d);
    for a = 1:d,
      wxa = wt .* x(a,:);
      for b = a:d,
        hii_ab = wxa * x(b,:)';
        hii(a,b) = hii_ab;
        hii(b,a) = hii_ab;
      end
    end
    h( (i-1)*d+1 : i*d , (i-1)*d+1 : i*d ) = -hii;
  end
  for i = 1:k-1,	% off-diagonal
    for j = i+1:k-1,
      wt = w .* post(j,:) .* post(i,:);
      hij = zeros(d,d);
      for a = 1:d,
        wxa = wt .* x(a,:);
        for b = a:d,
          hij_ab = wxa * x(b,:)';
          hij(a,b) = hij_ab;
          hij(b,a) = hij_ab;
        end
      end
      h( (i-1)*d+1 : i*d , (j-1)*d+1 : j*d ) = hij;
      h( (j-1)*d+1 : j*d , (i-1)*d+1 : i*d ) = hij;
    end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% debug/visualization
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
  pause(0.1);

% eof
