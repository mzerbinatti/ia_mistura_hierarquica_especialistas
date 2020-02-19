% generate some data
k = 3;
n = 500;
alpha = 0.3;
X = [ 
    alpha*randn(n,1)-1 alpha*randn(n,1)+1; %randn(n,1);
    alpha*randn(n,1)+0 alpha*randn(n,1)+1; %randn(n,1);
    alpha*randn(n,1)+1 alpha*randn(n,1)+1; %randn(n,1);
    alpha*randn(n,1)-1 alpha*randn(n,1)+0; %randn(n,1);
    alpha*randn(n,1)+0 alpha*randn(n,1)+0; %randn(n,1);
    alpha*randn(n,1)+1 alpha*randn(n,1)+0; %randn(n,1);
    alpha*randn(n,1)-1 alpha*randn(n,1)-1; %randn(n,1);
    alpha*randn(n,1)+0 alpha*randn(n,1)-1; %randn(n,1);
    alpha*randn(n,1)+1 alpha*randn(n,1)-1; %randn(n,1);
    ];
X = (X +1.5) / 3;
X = [X ones(9*n,1)];

a = [ones(n,1) zeros(n,1) zeros(n,1)];
b = [zeros(n,1) ones(n,1) zeros(n,1)];
c = [zeros(n,1) zeros(n,1) ones(n,1)];
Yvec = [a;b;a;b;c;b;c;a;c];

%a = [ones(n,1) zeros(n,1)];
%b = [zeros(n,1) ones(n,1)];
%Yvec = [a;b;a;b;a;b;a;b;a];

[v,Y] = max(Yvec,[],2);

kolor = ['r' 'b' 'k' 'g'];
figure(1); clf;
hold on;
for i = 1:k
  ind = find(Y == i);
  plot(X(ind,1),X(ind,2),[kolor(i) '.']);
end;
axis([0 1 0 1]); axis square;
