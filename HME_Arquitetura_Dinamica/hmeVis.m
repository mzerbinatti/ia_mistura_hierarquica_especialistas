function hmeVis(hme,k,xrange,yrange,Xdata,prefix)
% INPUTS
%	hme	HME model
%	k	number of classes
%	xrange	1x2 vector giving x range for grid sampling
%	yrange	1x2 vector giving y range for grid sampling
%	[Xdata]	dxn matrix of samples
%	[prefix] for output files
%
% OUPUTS
%	none


error(nargchk(4,6,nargin));
if nargin < 6, prefix = ''; end

%dim = size(hme.param,1);
dim=size(Xdata,2)+1;

if dim > 3 && nargin < 5,
  error('dim>3 and Xdata not provided.');
end

plotNumber = 30; % depth plots starting with this number

if dim == 3,
  % we will sample the HME on a grid
  res = 100;
  [x,y] = meshgrid( ...
      linspace(xrange(1),xrange(2),res), ...
      linspace(yrange(1),yrange(2),res));
  
  x = x(:);
  y = y(:);
  n = length(x);

  % get the class posteriors for the grid
  %X = [x y ones(n,1)]';
  X=[x y];
  post = hmeEval(hme,k,X);
  post=post';
  % for each grid point, show the dominant class
  [mval,ind] = max(post,[],1);
  set(figure(plotNumber),'DoubleBuffer','on'); 
  plotNumber = plotNumber+1;
  nplot = k+1;
  nx = ceil(sqrt(nplot));
  ny = ceil(nplot/nx);
  subplot(nx,ny,1); hold off;
  imagesc(x,y,reshape(ind,res,res));
  title('dominant classes'); 
  axis xy; axis image;
  
  % show the posterior for each class
  for i = 1:k,
    subplot(nx,ny,i+1); 
    imagesc(x,y,reshape(post(i,:),res,res),[0 1]);
    title(['class ' num2str(i) ' posterior']);
    axis xy; axis image;
  end
else
  x = [];
  y = [];
end

depth = getDepth(hme);

global G_plotCount;
G_plotCount = zeros(1,depth);
countPlots(hme,1);

global G_histData;
if nargin > 4,
  G_histData = cell(1,depth);
  for i = 1:depth,
   G_histData{i} = zeros(G_plotCount(i),size(Xdata,2));
 %   G_histData{i} = zeros(G_plotCount(i),size(Xdata,1));
  end
end

global G_plotHandles;
G_plotHandles = cell(1,depth-1);
if dim == 3,
  for level = 2:depth, 
    G_plotHandles{level-1} = figure(plotNumber);
    plotNumber = plotNumber+1;
    set(G_plotHandles{level-1},'DoubleBuffer','on');
  end
end

global G_plotCounter;
G_plotCounter = zeros(1,depth);
if nargin > 4,
  hmeVisGate(hme,k,1,x,y,ones(1,length(x)),Xdata,ones(1,size(Xdata,2)));
%  hmeVisGate(hme,k,1,x,y,ones(1,length(x)),Xdata,ones(1,size(Xdata,1)));
else
  if dim == 3,
    hmeVisGate(hme,k,1,x,y,ones(1,n));
  end
end

if nargin > 4,
  h = figure(plotNumber);
  plotNumber = plotNumber+1;
  set(h,'DoubleBuffer','on');
  for i = 2:depth,
    h_max = zeros(1,G_plotCount(i));
    h_sum = zeros(1,G_plotCount(i));
    [maxval,ind] = max(G_histData{i},[],1);
    for j = 1:G_plotCount(i), h_max(j) = sum(ind==j); end
    for j = 1:G_plotCount(i), h_sum(j) = sum(G_histData{i}(j,:)); end
    h_max = h_max ./ sum(h_max);
    h_sum = h_sum ./ sum(h_sum);
    figure(h); subplot(1,depth-1,i-1); 
    bar([h_max;h_sum]',1); 
    axis([0.25 G_plotCount(i)+0.75 0 1]);
    legend('max','sum');
    title(sprintf('data split level %d',i));
    ylabel('fraction of data'); xlabel('expert');
  end
end

% save all plots to EPS files
if dim == 3,
  for level = 2:depth, 
    fn = sprintf('%sgate%d',prefix,level);
    print(G_plotHandles{level-1},'-depsc',fn);
  end
end
if nargin > 4,
  print(h,'-depsc',[prefix 'experts']);
end

% clear out globals
clear G_plotCount G_histData G_plotHandles G_plotCounter;

%%% END hmeVis %%%

function countPlots(hme,level)
  global G_plotCount;
  G_plotCount(level) = G_plotCount(level) + 1;
  if ~hme.leaf,
    for i = 1:length(hme.children),
      countPlots(hme.children{i},level+1);
    end
  end

function hmeVisGate(hme,k,level,x,y,w,Xdata,Wdata)
  global G_plotCount G_plotCounter G_plotHandles G_histData;
  G_plotCounter(level) = G_plotCounter(level) + 1;

  %dim = size(hme.param,1);
  dim=size(Xdata,2)+1;

  % plot gating weights on the grid
  if dim == 3 && level > 1,
    nx = ceil(sqrt(G_plotCount(level)));
    ny = ceil(G_plotCount(level)/nx);
    figure(G_plotHandles{level-1});
    subplot(nx,ny,G_plotCounter(level));
    res = sqrt(length(x));
    imagesc(x,y,reshape(w,res,res),[0 1]);
    title(sprintf('gate %d (level=%d)',G_plotCounter(level),level));
    axis xy; axis image;
  end
  
  % collect data for histograms
  if nargin > 6,
      G_histData{level}(G_plotCounter(level),:) = Wdata;
  end

  % recurse
  if ~hme.leaf,
    if dim == 3,
        switch lower (hme.type)
        %switch lower(get_name(hme.type.algorithm))
            case 'linear'
                gate = logistK_eval(hme.param,[x y ones(size(x))]');
            case 'mlp'
                %gate = mlpK_eval(hme.param,[x y ones(size(x))]');
                gate=mlpk_eval(hme.type,[x y]);
                %gate=gate';
            case 'Localized'
                gate = localizedK_eval(hme.param,[x y ones(size(x))]');
        end
    end
    if nargin > 6,
        switch lower (hme.type)
        %switch lower(get_name(hme.type.algorithm))
            case 'linear'
                gateData = logistK_eval(hme.param,Xdata);
            case 'mlp'
                gateData = mlpk_eval(hme.param,Xdata);
                %gateData=mlpk_eval(hme.type,Xdata);
                %gateData=gateData';
            case 'Localized'
                gateData = localizedK_eval(hme.param,Xdata);
        end
    end
    for i = 1:length(hme.children),
      if dim == 3,
        wi = w .* gate(i,:);
      else
        wi = w;
      end
      if nargin > 6,
        save all
        wiData = Wdata .* gateData(i,:);
        hmeVisGate(hme.children{i},k,level+1,x,y,wi,Xdata,wiData);
      else
        hmeVisGate(hme.children{i},k,level+1,x,y,wi);
      end
    end
  end

function [depth] = getDepth(hme)
  if hme.leaf,
    depth = 1;
  else
    depth = 0;
    for i = 1:length(hme.children),
      depth = max(depth,1+getDepth(hme.children{i}));
    end
  end
