function [dJdA,dJdB]=grad_expert_class(X,Y,w,A,B,h)
N=length(X);
Nm=size(B,1);
Z=X*A';
V=tanh(Z);
T=[V,ones(N,1)]*B';
g1=exp(T);
g=g1./(sum(g1')'*ones(1,Nm));
erro=repmat(w,1,Nm).*(g-Y);

dJdB=erro'*[V,ones(N,1)];
sig=erro*B(:,1:h).*(1-V.*V);
%dJdA=sig'*[X,ones(N,1)];
dJdA=sig'*X;

