function [hme] = hmeInitRand(hme,levels,b,d,k)

global cont trk dstr

if hme.leaf, % Rede Especialista
    cont=cont+1;    
    switch lower(hme.type)       
        case 'linear'
            % Initialize with random parameters of small magnitude so that
            % splits start out very soft.
            [d,k] = size(hme.param);
            hme.param = 1e-3 * rand(d,k);
            hme.param(:,k) = 0;
        case 'mlp'
            h=hme.h;
            hme.param.A = rand(h,d)/2; % Ja tem bias
            hme.param.B = rand(k,h+1)/2;
        case 'localized'
            disp('Nao implementado')
        case 'svm_lssvm'
            switch lower (hme.type.child{1})
                case 'svm'
                    disp(sprintf('Inicializando o %s %d usando Kmeans',get_name(hme.type.child{1}.algorithm),cont));              
                    Ind=find(trk.X==cont);               % Pontos correspondente para o especialista cont
                    ds=get(dstr,Ind);                   % Copia somente alguns pontos do objeto ds
                    %k1= kernel('rbf',1);
                    %hme.type=svm_lssvm(svm({k1,'optimizer="decomp"'}));
                    %hme.type=a.child{1};
                    hme.type=a.expert;
                    [tr1,hme.type]=train(hme.type,ds);  % Treina um svm
                    clear ds tr1;
                case 'lssvm'
                    disp(sprintf('Inicializando o %s %d usando Kmeans',get_name(hme.type.child{1}.algorithm),cont));              
                    Ind=find(trk.X==cont);               % Pontos correspondente para o especialista cont
                    %Ind=find(trk==cont)
                    ds=get(dstr,Ind);                   % Copia somente alguns pontos do objeto ds
                    %k1= kernel('rbf',1);
                    %hme.type=svm_lssvm(lssvm({k1,'optimizer="matlab"'}));
                    %hme.type=a.child{1};
                    hme.type=a.expert;
                    [tr1,hme.type]=train(hme.type,ds);  % Treina um svm
                    clear ds tr1;
            end
    end

else % Rede Gating
    for i = 1:b,
        hme.children{i} = hmeInitRand(hme.children{i},levels-1,b,d,k);    
    end    
  
  switch lower (hme.type)
      case 'linear'
          % Initialize with a gating function that will divide the postive
          % unit orthant into approximately equal regions with soft
          % transitions.
          [d,b] = size(hme.param);
          hme.param = zeros(d,b);
          for i = 1:b-1,
              % create vector at random orientation
              while 1,
                  v = 2*rand(d-1,1) - 1;
                  if sum(v.^2) < 1, break, end
              end
              v = v ./ norm(v);
              % pick random location in middle of orthant
              x0 = 0.4 + 0.2*rand(d-1,1);
              % compute line intercept (line eqn: (x-x0).v=0)
              icpt = -x0'*v;
              hme.param(:,i) = [v;icpt] * 3;
          end
          hme.param(:,b) = 0;
      case 'mlp'          
          h=hme.h;
          hme.param.A = rand(h,d)/4;
          hme.param.B = rand(b,h+1)/4;
          
      case 'localized'
          disp('Nao implementado')
  end
  
end