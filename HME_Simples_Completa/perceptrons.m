function Y=perceptrons(X,A,B)
N=length(X);
Z = X*A';
V=tanh(Z);
Y=[V,ones(N,1)]*B';