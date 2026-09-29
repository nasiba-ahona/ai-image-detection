function R = spectral_core(img)
%SPECTRAL_CORE  Core DSP of the spectral group (sp_grid_dB, sp_slope, sp_rolloff).
%   R = spectral_core(img)   img = imread() output (gray or RGB, any class)
%
%   R.ok        false if the image is smaller than 96 px on a side
%   R.features  [sp_grid_dB sp_slope sp_rolloff]   (raw feature values)
%   R.score     fused logistic score (higher = more AI-like)
%   R.p         P(AI) = sigmoid(score)
%   R.verdict   'AI-GENERATED' | 'REAL' | 'UNCERTAIN'
%   R.votes     1x3 logical, true = that feature votes AI
%   R.nAI       number of features voting AI (0..3)
%
%   Needs: Image Processing Toolbox (medfilt2). Nothing else.
%   All helper names start with sp_ so they will not clash when merged.

R = struct('ok',false,'features',nan(1,3),'score',NaN,'p',NaN, ...
           'verdict','','votes',false(1,3),'nAI',NaN);

% ---- fixed settings ------------------------------------------------------
TILE    = 256;      % analysis window, never resized
NTILES  = 5;        % centre + 4 quadrant tiles
MINSIDE = 96;       % smaller images are skipped

% ---- calibrated constants (from the calibration set) ---------------------
thr    = [10.5176 -28.7970 -7.65724];     % per-feature thresholds
dir_   = [-1 1 1];                        % +1: high = AI, -1: low = AI
med    = [10.6326 -27.3962 -8.93522];     % imputation values for NaN features
mu     = [11.2144 -28.5317 -8.82774];     % z-score mean
sd     = [ 3.49992 9.79536  3.44455];     % z-score std
w      = [-0.378905 0.439417 0.199329];   % ridge-logistic weights
b      = -0.047246;                       % ridge-logistic bias
thr_s  = 0.2115;                          % AI when score >= thr_s
band_s = 0.0591;                          % +/- band around thr_s = UNCERTAIN

% ---- pre-processing ------------------------------------------------------
img = im2double(img);
if size(img,3)==1, img = repmat(img,[1 1 3]); end
if size(img,3)>3,  img = img(:,:,1:3);        end
[H,W,~] = size(img);
if H<MINSIDE || W<MINSIDE, return; end

S  = min([TILE, 2*floor(H/2), 2*floor(W/2)]);
TC = sp_tile_coords(H,W,S,NTILES);
g  = img(:,:,2);                          % green channel

% ---- feature extraction (mean over tiles) --------------------------------
E = nan(size(TC,1),3);
for t = 1:size(TC,1)
    r = TC(t,1)+(0:S-1);
    c = TC(t,2)+(0:S-1);
    try
        E(t,:) = sp_f_spectral(g(r,c));
    catch
    end
end
v = sp_colmean_nan(E);

% ---- fusion + decision ---------------------------------------------------
x = v;
bad = ~isfinite(x);
x(bad) = med(bad);
z = (x-mu)./sd;
z = max(min(z,8),-8);
s = z*w(:) + b;
votes = (dir_.*x) >= (dir_.*thr);

if     s >= thr_s+band_s, verdict = 'AI-GENERATED';
elseif s <= thr_s-band_s, verdict = 'REAL';
else,                     verdict = 'UNCERTAIN';
end

R.ok       = true;
R.features = v;
R.score    = s;
R.p        = 1/(1+exp(-s));
R.verdict  = verdict;
R.votes    = votes;
R.nAI      = sum(votes);
end

% =========================================================================
%  CORE DSP
% =========================================================================
function out = sp_f_spectral(x)
%SP_F_SPECTRAL  Upsampler grid energy and radial profile of the spectrum.
out = nan(1,3);
[h,w] = size(x);
r = x - medfilt2(x,[3 3],'symmetric');              % noise residual

% --- energy at the canonical upsampling frequencies ----------------------
P  = abs(fft2(r .* sp_hann2(h,w)));
md = median(P(:)) + eps;
kk = [h/2 0; 0 w/2; h/2 w/2; h/4 0; 0 w/4; h/4 w/4; 3*h/4 0; 0 3*w/4];
gv = nan(1,size(kk,1));
for i = 1:size(kk,1)
    ri = mod(round(kk(i,1)),h) + 1;
    ci = mod(round(kk(i,2)),w) + 1;
    rr = mod((ri-2:ri),h) + 1;
    cc = mod((ci-2:ci),w) + 1;
    gv(i) = max(max(P(rr,cc)));
end
out(1) = 20*log10(mean(gv)/md);                     % sp_grid_dB

% --- radial structure of the image spectrum ------------------------------
L = 20*log10(abs(fftshift(fft2((x-mean(x(:))).*sp_hann2(h,w)))) + eps);
[prof,rc] = sp_radial_profile(L,48);
ok  = isfinite(prof);
mid = ok & rc>=0.25 & rc<=0.65;
hi  = ok & rc> 0.80;
lo  = ok & rc>=0.40 & rc<=0.60;
if nnz(mid) >= 5
    cfit = polyfit(log10(rc(mid)),prof(mid),1);
    out(2) = cfit(1);                               % sp_slope
end
if nnz(hi) >= 3 && nnz(lo) >= 3
    out(3) = mean(prof(hi)) - mean(prof(lo));       % sp_rolloff
end
end

function W = sp_hann2(h,w)
W = (0.5*(1-cos(2*pi*(0:h-1)'/max(h-1,1)))) * (0.5*(1-cos(2*pi*(0:w-1)/max(w-1,1))));
end

function [prof,rc] = sp_radial_profile(L,nb)
[h,w] = size(L);
[X,Y] = meshgrid((1:w)-(floor(w/2)+1),(1:h)-(floor(h/2)+1));
Rr   = sqrt(X.^2 + Y.^2);
rmax = min(h,w)/2;
idx  = discretize(Rr(:),linspace(1,rmax,nb+1));
ok   = isfinite(idx);
s = accumarray(idx(ok),L(ok),[nb 1]);
c = accumarray(idx(ok),1,    [nb 1]);
prof = (s./max(c,1))';
prof(c==0) = NaN;
e  = linspace(1,rmax,nb+1);
rc = (e(1:end-1)+e(2:end))/2 / rmax;
end

function TC = sp_tile_coords(H,W,S,n)
fr = [0.5 0.22 0.78 0.22 0.78];
fc = [0.5 0.22 0.22 0.78 0.78];
n  = min(n,numel(fr));
TC = zeros(n,2);
for i = 1:n
    TC(i,1) = min(max(round(fr(i)*(H-S))+1,1),max(H-S+1,1));
    TC(i,2) = min(max(round(fc(i)*(W-S))+1,1),max(W-S+1,1));
end
TC = unique(TC,'rows','stable');                    % keeps the centre tile first
end

function m = sp_colmean_nan(A)
m = nan(1,size(A,2));
for j = 1:size(A,2)
    v = A(:,j); v = v(isfinite(v));
    if ~isempty(v), m(j) = mean(v); end
end
end
