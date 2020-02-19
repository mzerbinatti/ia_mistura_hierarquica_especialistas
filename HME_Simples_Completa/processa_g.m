function [dJdA,dJdB]=processa(X,A,B,saida,h)

N=length(X);

Nm=size(B,1);

Z=[X,ones(N,1)]*A';

V=tanh(Z);

T=[V,ones(N,1)]*B';

g1=exp(T);

g=g1./(sum(g1')'*ones(1,Nm));

erro=g-saida;

dJdB=erro'*[V,ones(N,1)];

sig=erro*B(:,1:h).*(1-V.*V);

dJdA=sig'*[X,ones(N,1)];

