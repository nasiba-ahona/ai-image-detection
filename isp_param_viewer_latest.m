function isp_param_viewer_latest()
% ISP_PARAM_VIEWER - measures 3 camera-ISP parameters and plots them (standalone)
%

%
% Parameters (one per ISP stage)
%   rc_disp   IQR of flat-area lag-1 residual correlation   demosaic + noise reduction  [own hypothesis]
%   nlf_var   (max-min)/mean of noise-vs-brightness curve   denoise + tone curve        [Immerkaer CVIU 1996; Liu et al. TPAMI 2008]
%   cl_ratio  chroma noise / luma noise at half resolution  chroma noise reduction      [own hypothesis]
%
% Each image: centre crop to O.cropSide px (offsets multiple of 16, never resized) ->
% JPEG re-save at O.rejpeg so all images share one format -> measure.
% Region map (right): blue/red = one class holds >=O.purity of the local 2D density,
% grey = overlap. Descriptive only.


O.cropSide=512;      % centre-crop side (px); images smaller than this are used whole
O.rejpeg=90;          % JPEG re-save quality before measuring | 0 = off
O.purity=0.8;         % region map purity
O.blk=16;             % block size (px) for noise estimates
O.minBlocksBin=15;    % min blocks per brightness bin
O.maxEdges=4000;      % max edges used in the halo test
O.minC=25;            % min edge contrast (DN out of 255)
O.progressEvery=50;   % images between progress updates
O.params={'rc_disp','nlf_var','cl_ratio'};   % display names (order fixed by isp_params)

fig=figure('Name','ISP parameter viewer','NumberTitle','off','Units','normalized','Position',[0.04 0.07 0.92 0.84]);
lbl={'Add REAL folder','Add AI folder','Add single image(s)','Export CSV','Clear images'};
cbs={@(s,e)add_folder(fig,1),@(s,e)add_folder(fig,2),@(s,e)add_images(fig),@(s,e)export_csv(fig),@(s,e)clear_images(fig)};
for i=1:numel(lbl)
    uicontrol(fig,'Style','pushbutton','String',lbl{i},'Units','normalized','Position',[0.01 0.90-0.07*(i-1) 0.13 0.055],'Callback',cbs{i},'BusyAction','cancel','Interruptible','off');
end
uicontrol(fig,'Style','text','String','region map X','Units','normalized','Position',[0.01 0.52 0.13 0.03]);
A.hx=uicontrol(fig,'Style','popupmenu','String',O.params,'Value',1,'Units','normalized','Position',[0.01 0.48 0.13 0.04],'Callback',@(s,e)redraw(fig));
uicontrol(fig,'Style','text','String','region map Y','Units','normalized','Position',[0.01 0.43 0.13 0.03]);
A.hy=uicontrol(fig,'Style','popupmenu','String',O.params,'Value',2,'Units','normalized','Position',[0.01 0.39 0.13 0.04],'Callback',@(s,e)redraw(fig));
A.hs=uicontrol(fig,'Style','text','String','','Units','normalized','Position',[0.01 0.03 0.13 0.33],'HorizontalAlignment','left');
A.ax=cell(1,numel(O.params));
for k=1:numel(O.params)
    c=mod(k-1,2);
    r=floor((k-1)/2);
    A.ax{k}=axes('Parent',fig,'Position',[0.19+c*0.21 0.56-r*0.48 0.16 0.37]);
end
A.axr=axes('Parent',fig,'Position',[0.64 0.08 0.34 0.82]);
A.O=O;
A.X=zeros(0,numel(O.params));
A.cls=zeros(0,1);
A.path=cell(0,1);
A.isJ=false(0,1);
A.dirs=cell(0,1);
setappdata(fig,'A',A);
redraw(fig);
end

%% ===================== callbacks =====================
function add_folder(fig,cls,d)
A=getappdata(fig,'A');
lab={'REAL','AI'};
if nargin<3
    d=uigetdir(pwd,sprintf('Select the %s image folder',lab{cls}));
    if isequal(d,0)
        return;
    end
end
key=norm_path(d);
if any(cellfun(@(p)same_path(p,key),A.dirs))
    warndlg('This folder is already loaded.');
    return;
end
names=list_images(d);
if isempty(names)
    warndlg('No .jpg .jpeg .png .tif .tiff .bmp images in this folder.');
    return;
end
set(A.hs,'String',{'measuring...',d});
drawnow;
[X,J]=measure_folder(d,names,A.O);
if ~ishandle(fig)
    return;
end
A=getappdata(fig,'A');
n=numel(names);
A.X=[A.X;X];
A.cls=[A.cls;cls*ones(n,1)];
A.path=[A.path;cellfun(@(f)fullfile(d,f),names(:),'UniformOutput',false)];
A.isJ=[A.isJ;J(:)];
A.dirs{end+1,1}=key;
setappdata(fig,'A',A);
fprintf('\nadded %s folder %s: %d images (original JPEG %.0f%%)\n',lab{cls},d,n,100*mean(J));
print_summary(A);
redraw(fig);
end

function add_images(fig,files)
A=getappdata(fig,'A');
if nargin<2
    [f,p]=uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Images'},'Select image(s)','MultiSelect','on');
    if isequal(f,0)
        return;
    end
    if ischar(f)
        f={f};
    end
    files=cellfun(@(x)fullfile(p,x),f,'UniformOutput',false);
end
r=A.cls==1;
a=A.cls==2;
for i=1:numel(files)
    set(A.hs,'String',{'measuring...',files{i}});
    drawnow;
    [x,isJ]=one_image(files{i},A.O);
    A.X=[A.X;x];
    A.cls=[A.cls;3];
    A.path=[A.path;files(i)];
    A.isJ=[A.isJ;isJ];
    idx=nnz(A.cls==3);
    fprintf('\n[%d] %s  (original JPEG:%d  re-saved q%d)\n',idx,files{i},isJ,A.O.rejpeg);
    for k=1:numel(A.O.params)
        if isnan(x(k))
            fprintf('  %-9s       N/A  (not measurable in this image)\n',A.O.params{k});
            continue;
        end
        s='';
        if any(r)
            s=[s,sprintf('  above %5.1f%% of real',rank_pct(A.X(r,k),x(k)))];
        end
        if any(a)
            s=[s,sprintf('  above %5.1f%% of AI',rank_pct(A.X(a,k),x(k)))];
        end
        fprintf('  %-9s %9.4f%s\n',A.O.params{k},x(k),s);
    end
print_verdict(A,x);
end
setappdata(fig,'A',A);
redraw(fig);
end

function export_csv(fig)
A=getappdata(fig,'A');
if isempty(A.cls)
    warndlg('Nothing to export yet.');
    return;
end
[f,p]=uiputfile('*.csv','Save parameters','isp_params.csv');
if isequal(f,0)
    return;
end
cl={'real','AI','single'};
fid=fopen(fullfile(p,f),'w');
fprintf(fid,'file,class,original_jpeg,%s\n',strjoin(A.O.params,','));
for i=1:numel(A.cls)
    fprintf(fid,'"%s",%s,%d',A.path{i},cl{A.cls(i)},A.isJ(i));
    fprintf(fid,',%.6g',A.X(i,:));
    fprintf(fid,'\n');
end
fclose(fid);
fprintf('\nsaved %s\n',fullfile(p,f));
end

function clear_images(fig)
% removes only the single images (the stars); the loaded folders stay
A=getappdata(fig,'A');
k=A.cls~=3;
A.X=A.X(k,:);
A.cls=A.cls(k);
A.path=A.path(k);
A.isJ=A.isJ(k);
setappdata(fig,'A',A);
redraw(fig);
end

%% ===================== display =====================
function redraw(fig)
A=getappdata(fig,'A');
P=A.O.params;
r=A.cls==1;
a=A.cls==2;
s=find(A.cls==3);
set(A.hs,'String',{sprintf('crop side %d px',A.O.cropSide),sprintf('JPEG re-save q%d',A.O.rejpeg),'',...
    sprintf('real images: %d',nnz(r)),sprintf('AI images: %d',nnz(a)),sprintf('single images: %d',numel(s))});
set(0,'CurrentFigure',fig);
for k=1:numel(P)
    set(fig,'CurrentAxes',A.ax{k});
    cla reset;
    hold on;
    v=A.X(:,k);
    [x1,y1]=strip_xy(1,v(r));
    [x2,y2]=strip_xy(2,v(a));
    plot(x1,y1,'.',x2,y2,'.');
    vs=v(s);
    plot(3*ones(numel(s),1),vs,'k*');
    for i=1:numel(s)
        if ~isnan(vs(i))
            text(3.1,vs(i),sprintf('%d',i));
            if numel(s)<=3
                plot([0.5 3.5],[vs(i) vs(i)],'k:');
            end
        end
    end
    th=zone_thr(A,k);
    for i=1:numel(th)
        if ~isnan(th(i))
            plot([0.5 3.5],[th(i) th(i)],'k--');
        end
    end
    hold off;
    xlim([0.5 3.5]);
    set(gca,'XTick',[1 2 3],'XTickLabel',{'real','AI','images'});
    lim=value_lim(v([find(r);find(a)]),vs);
    if ~isempty(lim)
        ylim(lim);
    end
    ttl=strrep(P{k},'_','\_');
    if all(~isnan(zone_thr(A,k)))
        ttl=sprintf('%s  thr %.4g / %.4g',ttl,min(zone_thr(A,k)),max(zone_thr(A,k)));
    end
    if nnz(~isnan(v(r)))>=5&&nnz(~isnan(v(a)))>=5
        Au=auc_rank(v(r&~isnan(v)),v(a&~isnan(v)));
        ttl=sprintf('%s  (AUC %.3f)',ttl,max(Au,1-Au));
    end
    title(ttl);
end

set(fig,'CurrentAxes',A.axr);
cla reset;
i=get(A.hx,'Value');
j=get(A.hy,'Value');
nx=strrep(P{i},'_','\_');
ny=strrep(P{j},'_','\_');
if i==j
    title('choose two different parameters for X and Y');
    return;
end
gr=r&~isnan(A.X(:,i))&~isnan(A.X(:,j));
ga=a&~isnan(A.X(:,i))&~isnan(A.X(:,j));
xr=A.X(gr,i);
yr=A.X(gr,j);
xa=A.X(ga,i);
ya=A.X(ga,j);
xs=A.X(s,i);
ys=A.X(s,j);
lx=value_lim([xr;xa],xs);
ly=value_lim([yr;ya],ys);
hold on;
hh=[];
lab={};
ttl={sprintf('%s vs %s',nx,ny)};
if numel(xr)>=10&&numel(xa)>=10
    nb=120;
    xc=lx(1)+diff(lx)/nb*((1:nb)-0.5);
    yc=ly(1)+diff(ly)/nb*((1:nb)-0.5);
    Pr=dens2(xr,yr,lx,ly,nb);
    Pa=dens2(xa,ya,lx,ly,nb);
    occ=Pr>=hdr_level(Pr,0.95)|Pa>=hdr_level(Pa,0.95);
    post=Pr./(Pr+Pa+realmin);
    Rg=zeros(nb);
    Rg(occ&post>=A.O.purity)=1;
    Rg(occ&post<=1-A.O.purity)=2;
    Rg(occ&post>1-A.O.purity&post<A.O.purity)=3;
    fr=region_share(xr,yr,Rg,lx,ly,nb);
    fa=region_share(xa,ya,Rg,lx,ly,nb);
    cmap=[1 1 1;0.80 0.88 1;1 0.83 0.78;0.86 0.86 0.86];
    imagesc(xc,yc,Rg);
    colormap(cmap);
    caxis([-0.5 3.5]);
    hh=[patch(NaN,NaN,cmap(2,:)),patch(NaN,NaN,cmap(3,:)),patch(NaN,NaN,cmap(4,:))];
    lab={'real region','AI region','overlap'};
    ttl{2}=sprintf('blue: real only | red: AI only | grey: overlap (%.0f%% purity)',100*A.O.purity);
    ttl{3}=sprintf('real %.0f%% own region / %.0f%% overlap | AI %.0f%% own region / %.0f%% overlap',fr(1),fr(2),fa(3),fa(2));
end
hd=plot(nan_if_empty(xr),nan_if_empty(yr),'.',nan_if_empty(xa),nan_if_empty(ya),'.');
hh=[hh,hd(:)'];
lab=[lab,{sprintf('real (%d)',numel(xr)),sprintf('AI (%d)',numel(xa))}];
g=find(~isnan(xs)&~isnan(ys));
if ~isempty(g)
    hi=plot(xs(g),ys(g),'k*');
    for t=g(:)'
        text(xs(t),ys(t),sprintf('  %d',t));
    end
    hh=[hh,hi];
    lab{end+1}='single images';
end
hold off;
set(gca,'YDir','normal');
if ~isempty(lx)
    xlim(lx);
end
if ~isempty(ly)
    ylim(ly);
end
xlabel(nx);
ylabel(ny);
title(ttl);
if ~isempty(A.cls)
    legend(hh,lab);
end
end

function print_summary(A)
r=A.cls==1;
a=A.cls==2;
fprintf('\n=== loaded: %d real  %d AI  %d single ===\n',nnz(r),nnz(a),nnz(A.cls==3));
fprintf('  %-9s  %-30s  %-30s  %s\n','parameter','real  median [IQR]  n','AI  median [IQR]  n','AUC (separation)');
for k=1:numel(A.O.params)
    au='-';
    vr=A.X(r&~isnan(A.X(:,k)),k);
    va=A.X(a&~isnan(A.X(:,k)),k);
    if numel(vr)>=5&&numel(va)>=5
        x=auc_rank(vr,va);
        au=sprintf('%.3f',max(x,1-x));
    end
    fprintf('  %-9s  %-30s  %-30s  %s\n',A.O.params{k},stat_str(vr),stat_str(va),au);
end
end

function s=stat_str(v)
if isempty(v)
    s='-';
else
    s=sprintf('%.4g [%.4g %.4g]  %d',median(v),pct(v,0.25),pct(v,0.75),numel(v));
end
end

function th=zone_thr(A,k)
% two decision lines for parameter k: 85th percentile of the loaded real values
% and 25th percentile of the loaded AI values
th=[NaN,NaN];
vr=A.X(A.cls==1,k);
va=A.X(A.cls==2,k);
vr=vr(~isnan(vr));
va=va(~isnan(va));
if numel(vr)>=5
    th(1)=pct(vr,0.85);
end
if numel(va)>=5
    th(2)=pct(va,0.25);
end
end

function z=zone_label(A,k,v)
% real / AI / uncertain for one parameter of one image
z='uncertain';
if isnan(v)
    z='uncertain (N/A)';
    return;
end
th=zone_thr(A,k);
if any(isnan(th))
    z='uncertain (load both folders)';
    return;
end
if v<=th(2)&&v<th(1)
    z='real';
elseif v>=th(1)&&v>th(2)
    z='AI';
end
end

function print_verdict(A,x)
% per-parameter verdicts plus the combined one (cl_ratio decides; rc_disp only
% when cl_ratio is in the overlap zone; uncertain if cl_ratio is N/A or two of
% the three parameters are N/A)
z=cell(1,numel(A.O.params));
for k=1:numel(A.O.params)
    z{k}=zone_label(A,k,x(k));
    fprintf('  %-9s verdict: %s\n',A.O.params{k},upper(z{k}));
end
c=find(strcmp(A.O.params,'cl_ratio'));
d=find(strcmp(A.O.params,'rc_disp'));
v='UNCERTAIN';
if isempty(c)||isempty(d)
    v='UNCERTAIN';
elseif isnan(x(c))||nnz(isnan(x))>=2
    v='UNCERTAIN';
elseif strcmp(z{c},'real')||strcmp(z{c},'AI')
    v=upper(z{c});
elseif strcmp(z{d},'real')||strcmp(z{d},'AI')
    v=upper(z{d});
end
fprintf('  COMBINED verdict: %s\n',v);
end

function p=rank_pct(v,x)
v=v(~isnan(v));
p=NaN;
if ~isempty(v)
    p=100*mean(v<x);
end
end

function [x,y]=strip_xy(c,v)
% jittered strip coordinates; a NaN placeholder keeps the colour order when a class is empty
if isempty(v)
    x=c;
    y=NaN;
else
    x=c+linspace(-0.2,0.2,numel(v))';
    y=v;
end
end

function v=nan_if_empty(v)
if isempty(v)
    v=NaN;
end
end

function l=value_lim(v,extra)
v=v(~isnan(v));
e=extra(~isnan(extra));
if isempty(v)&&isempty(e)
    l=[];
    return;
end
if isempty(v)
    l=[min(e),max(e)];
else
    l=[pct(v,0.005),pct(v,0.995)];
    if ~isempty(e)
        l=[min(l(1),min(e)),max(l(2),max(e))];
    end
end
l=pad_lim(l);
end

function l=pad_lim(l)
if diff(l)<=0
    l=l+[-1 1]*max(abs(l(1)),1)*0.01;
end
l=l+[-0.05 0.05]*diff(l);
end

function P=dens2(x,y,lx,ly,nb)
in=x>=lx(1)&x<=lx(2)&y>=ly(1)&y<=ly(2);
ix=min(nb,floor((x(in)-lx(1))/diff(lx)*nb)+1);
iy=min(nb,floor((y(in)-ly(1))/diff(ly)*nb)+1);
H=accumarray([iy,ix],1,[nb,nb]);
g=exp(-((-9:9).^2)/18);
P=conv2(g,g,H,'same')/max(numel(x),1);
end

function t=hdr_level(P,m)
% density level whose enclosed cells hold fraction m of the class mass
s=sort(P(:),'descend');
c=cumsum(s);
if c(end)<=0
    t=Inf;
    return;
end
t=s(find(c>=m*c(end),1));
end

function f=region_share(x,y,Rg,lx,ly,nb)
ix=min(nb,max(1,floor((x-lx(1))/diff(lx)*nb)+1));
iy=min(nb,max(1,floor((y-ly(1))/diff(ly)*nb)+1));
r=Rg(sub2ind([nb,nb],iy,ix));
f=100*[mean(r==1),mean(r==3),mean(r==2),mean(r==0)];
end


%% ===================== measuring =====================
function [X,J]=measure_folder(d,names,O)
n=numel(names);
X=nan(n,3);
J=false(n,1);
M=pool_size();
fprintf('\n%s: measuring %d images (parallel workers %d)\n',d,n,M);
wb=waitbar(0,sprintf('0/%d images',n));
t0=tic;
for a=1:O.progressEvery:n
    idx=a:min(n,a+O.progressEvery-1);
    m=numel(idx);
    Xc=nan(m,3);
    Jc=false(m,1);
    parfor (ii=1:m,M)
        [xi,ji]=one_image(fullfile(d,names{idx(ii)}),O);
        Xc(ii,:)=xi;
        Jc(ii)=ji;
    end
    X(idx,:)=Xc;
    J(idx)=Jc;
    done=idx(end);
    el=toc(t0);
    msg=sprintf('%d/%d images  (~%.0f s left)',done,n,el/done*(n-done));
    fprintf('  %s\n',msg);
    if ishandle(wb)
        waitbar(done/n,wb,msg);
    end
end
if ishandle(wb)
    close(wb);
end
end

function [x,isJ]=one_image(f,O)
x=nan(1,3);
isJ=false;
try
    [I,isJ]=load_crop(f,O.cropSide);
    x=isp_params(resave_jpeg(I,O.rejpeg),O);
catch err
    fprintf('  skipped %s: %s\n',f,err.message);
end
end

function x=isp_params(I,O)
% returns [rc_disp nlf_var cl_ratio]; NaN = not measurable in this image
if size(I,3)>=3
    R=I(:,:,1);
    G=I(:,:,2);
    B=I(:,:,3);
    Y=0.299*R+0.587*G+0.114*B;
    Cb=-0.168736*R-0.331264*G+0.5*B;
    Cr=0.5*R-0.418688*G-0.081312*B;
else
    Y=I(:,:,1);
    Cb=[];
    Cr=[];
end
x=[p_rc_disp(Y,O),p_nlf_var(Y,O),p_cl_ratio(Y,Cb,Cr,O)];
end

% ---- rc_disp: demosaic + noise reduction shape the residual noise ----
% Residual = image - 3x3 mean. In the flattest 25% of 32x32 blocks: lag-1 correlation
% (mean of horizontal and vertical). rc_disp = IQR of that correlation over the blocks.
function v=p_rc_disp(Y,O)
v=NaN;
B=2*O.blk;
r=Y-filter2(ones(3)/9,Y);
r=r(2:end-1,2:end-1);
Yc=Y(2:end-1,2:end-1);
Ph=zeros(size(r));
Pv=Ph;
Ph(:,1:end-1)=r(:,1:end-1).*r(:,2:end);
Pv(1:end-1,:)=r(1:end-1,:).*r(2:end,:);
e0=bstat(r.^2,B,'mean');
eh=bstat(Ph,B,'mean');
ev=bstat(Pv,B,'mean');
S=filter2(ones(5)/25,Yc);
tex=sqrt(max(0,bstat(S.^2,B,'mean')-bstat(S,B,'mean').^2));
ok=bstat(Yc,B,'max')<250&bstat(Yc,B,'min')>5&e0>0.25;
if nnz(ok)<20
    return;
end
f=ok&tex<=pct(tex(ok),0.25);
rho=(eh(f)./e0(f)+ev(f)./e0(f))/2;
if numel(rho)<10
    return;
end
v=pct(rho,0.75)-pct(rho,0.25);
end

% ---- nlf_var: denoise + tone curve shape the noise-vs-brightness curve ----
% Immerkaer noise sigma per 16x16 block; per brightness bin (16 DN wide) the lowest 10%
% (flattest blocks). nlf_var = (max-min)/mean of that curve.
function v=p_nlf_var(Y,O)
v=NaN;
[sig,mu,mx,mn]=noise_blocks(Y,O.blk);
ok=mx<250&mn>5;
ed=0:16:256;
S=nan(numel(ed)-1,1);
for k=1:numel(ed)-1
    in=ok&mu>=ed(k)&mu<ed(k+1);
    if nnz(in)>=O.minBlocksBin
        S(k)=pct(sig(in),0.10);
    end
end
S=S(~isnan(S));
if numel(S)>=5&&mean(S)>0.2
    v=(max(S)-min(S))/mean(S);
end
end

% ---- cl_ratio: chroma noise reduction ----
% Y/Cb/Cr 2x2-averaged first so 4:2:0 subsampling alone does not decide it. In the
% flattest 25% of blocks: median chroma noise / median luma noise.
function v=p_cl_ratio(Y,Cb,Cr,O)
v=NaN;
if isempty(Cb)
    return;
end
B=O.blk;
h=2*floor(size(Y,1)/2);
w=2*floor(size(Y,2)/2);
Yh=half2(Y,h,w);
[sY,~,mx,mn]=noise_blocks(Yh,B);
sB=noise_blocks(half2(Cb,h,w),B);
sR=noise_blocks(half2(Cr,h,w),B);
S=filter2(ones(5)/25,Yh(2:end-1,2:end-1));
tex=sqrt(max(0,bstat(S.^2,B,'mean')-bstat(S,B,'mean').^2));
ok=mx<250&mn>5;
if nnz(ok)<20
    return;
end
f=ok&tex<=pct(tex(ok),0.25);
m=median(sY(f));
if m>0.2
    v=median((sB(f)+sR(f))/2)/m;
end
end

%% ===================== helpers =====================
function [sig,mu,mx,mn]=noise_blocks(X,B)
% Immerkaer: sigma=sqrt(pi/2)*mean|X*M|/6 with M=[1 -2 1;-2 4 -2;1 -2 1]
Rl=abs(filter2([1 -2 1;-2 4 -2;1 -2 1],X,'valid'));
sig=sqrt(pi/2)*bstat(Rl,B,'mean')/6;
if nargout>1
    Xc=X(2:end-1,2:end-1);
    mu=bstat(Xc,B,'mean');
    mx=bstat(Xc,B,'max');
    mn=bstat(Xc,B,'min');
end
end

function M=bstat(X,B,fun)
[h,w]=size(X);
nr=floor(h/B);
nc=floor(w/B);
X4=reshape(X(1:nr*B,1:nc*B),B,nr,B,nc);
switch fun
    case 'mean'
        M=sum(sum(X4,1),3)/(B*B);
    case 'max'
        M=max(max(X4,[],1),[],3);
    case 'min'
        M=min(min(X4,[],1),[],3);
end
M=reshape(M,nr*nc,1);
end

function Z=half2(X,h,w)
Z=(X(1:2:h,1:2:w)+X(2:2:h,1:2:w)+X(1:2:h,2:2:w)+X(2:2:h,2:2:w))/4;
end

function y=pct(x,p)
x=sort(x(:));
y=x(max(1,round(p*numel(x))));
end

function A=auc_rank(a,b)
x=[a(:);b(:)];
[~,i]=sort(x);
rk=zeros(size(x));
rk(i)=1:numel(x);
[~,~,g]=unique(x);
rk=accumarray(g,rk,[],@mean);
rk=rk(g);
na=numel(a);
nb=numel(b);
A=(sum(rk(1:na))-na*(na+1)/2)/(na*nb);
end

function names=list_images(d)
L=dir(d);
L=L(~[L.isdir]);
names={};
for i=1:numel(L)
    [~,~,e]=fileparts(L(i).name);
    if any(strcmpi(e,{'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}))
        names{end+1}=L(i).name;
    end
end
end

function M=pool_size()
M=0;
try
    if exist('gcp','file')==2&&license('test','Distrib_Computing_Toolbox')
        p=gcp;
        if ~isempty(p)
            M=p.NumWorkers;
        end
    end
catch
    M=0;
end
end

function [I,isJ]=load_crop(f,S)
isJ=is_jpeg(f);
[I,map]=imread(f);
[h,w,~]=size(I);
ch=min(h,S);
cw=min(w,S);
r0=16*floor((h-ch)/32);
c0=16*floor((w-cw)/32);
I=I(r0+1:r0+ch,c0+1:c0+cw,:);
if ~isempty(map)
    I=round(255*ind2rgb(I,map));
elseif isa(I,'uint16')
    I=double(I)*255/65535;
elseif islogical(I)||isfloat(I)
    I=double(I)*255;
else
    I=double(I);
end
if size(I,3)>3
    I=I(:,:,1:3);
elseif size(I,3)==2
    I=I(:,:,1);
end
end

function I=resave_jpeg(I,q)
if q>0
    tf=[tempname,'.jpg'];
    imwrite(uint8(I),tf,'Quality',q);
    I=double(imread(tf));
    delete(tf);
end
end

function t=is_jpeg(f)
t=false;
fid=fopen(f,'r');
if fid<0
    return;
end
b=fread(fid,2,'uint8');
fclose(fid);
t=numel(b)==2&&b(1)==255&&b(2)==216;
end

function p=norm_path(p)
p=strrep(p,'/',filesep);
p=strrep(p,'\',filesep);
absol=(ispc&&numel(p)>=2&&(p(2)==':'||strcmp(p(1:2),'\\')))||(~ispc&&~isempty(p)&&p(1)=='/');
if ~absol
    p=fullfile(pwd,p);
end
p=strrep(p,[filesep,'.',filesep],filesep);
while numel(p)>1&&p(end)==filesep&&~(ispc&&numel(p)==3)
    p(end)=[];
end
end

function t=same_path(a,b)
if ispc
    t=strcmpi(a,b);
else
    t=strcmp(a,b);
end
end

