function g=perceptrons(X,A,B)

N=length(X);
Nm=size(B,1);
Z=[X,ones(N,1)]*A';
V=tanh(Z);
T=[V,ones(N,1)]*B';
g1=exp(T);
g=g1./(sum(g1')'*ones(1,Nm));

