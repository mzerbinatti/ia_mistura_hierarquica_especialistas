function [dJdA,dJdB]=grad_expert(X,Y,w,A,B,h)

N=length(X);
Nm=size(B,1);
Z=X*A';
V=tanh(Z);
yexp=[V,ones(N,1)]*B';

erro=repmat(w,1,Nm).*(yexp-Y);
dJdB=erro'*[V,ones(N,1)];
sig=erro*B(:,1:h).*(1-V.*V);
dJdA=sig'*X;