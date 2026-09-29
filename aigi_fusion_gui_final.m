function aigi_fusion_gui_2()
% AIGI_FUSION_GUI  Real-vs-AI image detector: six DSP tests fused into one calibrated verdict.
%
% TESTS (19 features, one shared pre-processing path)
%   1 Spectral           sp_grid_dB sp_slope sp_rolloff        (spectral_core.m)
%   2 Nyquist CFA        nyq_dB                                (Test07)
%   3 Cross-channel/CFA  cc_corrRG cc_corrGB cc_corrRB cc_Gratio cfa_phaseG cfa_phaseRB   (Test10)
%   4 Colour histogram   co_clip co_full co_comb co_satmean co_satstd                  (Test15)
%   5 Texture contrast   tex_score                             (PatchCraft-style, texture detector)
%   6 ISP noise          rc_disp nlf_var cl_ratio              (isp_param_viewer)
%
% PRE-PROCESSING (every image, identical): centre crop O.cropSide px (offset multiple of 16,
% never resized) -> JPEG re-save at O.rejpeg (removes the PNG-vs-JPEG format confound) -> measure.
%
% CALIBRATION (button "Calibrate / recalibrate")
%   * per feature : AUC, learned direction, best balanced-accuracy threshold
%   * per test    : ridge-logistic model on that test's features (lambda by 5-fold CV)
%   * fusion      : (a) ridge-logistic on all 19 features and (b) stacked model on the 6
%                   out-of-fold test scores (non-negative weights = weightage per test).
%                   The one with the higher cross-validated AUC is used.
%   * decision    : threshold with best out-of-fold balanced accuracy + an UNCERTAIN band
%                   sized so that confident verdicts reach O.targetAcc (keeping >= O.minCoverage).
%   Robust z-scores (median/MAD), NaN -> median, class-balanced weights. All reported
%   accuracies are OUT-OF-FOLD (honest), not training accuracy.
%
% PERSISTENCE
%   After calibrating, all measured features + the model are saved to
%   aigi_fusion_calibration.mat next to this file and reloaded automatically at start.
%   To extend: (auto-loaded or "Load calibration data") + add new folders + Calibrate.
%   Images already in the data are skipped, old images are never re-measured.
%
% TESTING (after calibration)
%   "Add REAL/AI test folder": images are scored with the saved calibration only (never used
%   to fit it; images that were in the calibration set are excluded). The command window shows
%   how many were detected correctly / wrongly / UNCERTAIN, the efficiency per folder, and a
%   running total over all test folders (confusion matrix, overall and balanced efficiency).
%   Single images get a REAL / AI-GENERATED / UNCERTAIN verdict from the same calibration.
%
% Needs: Image Processing Toolbox. Parallel Computing Toolbox is used if present.

O  = default_options();
FT = feature_table();
assert(numel(FT.names)==O.nFeat,'feature table and O.nFeat disagree');

colNames  = [FT.names, cellfun(@(g)['test: ' g],FT.gnames,'UniformOutput',false), {'FUSED score'}];
nF = numel(FT.names); nG = numel(FT.gnames);
plotCol   = [nF+nG+1, nF+(1:nG), 1:nF];          % popup order: fused, tests, features
plotNames = colNames(plotCol);

fig = figure('Name','AIGI fusion detector','NumberTitle','off','Units','normalized', ...
             'Position',[0.03 0.06 0.94 0.86]);

lbl = {'Add REAL folder','Add AI folder','Add REAL + AI folders','Load calibration data','Calibrate / recalibrate', ...
       'Save calibration as...','Add REAL test folder','Add AI test folder', ...
       'Add single image(s)','Export CSV','Clear single images'};
fns = {@()add_folder(fig,1),@()add_folder(fig,2),@()add_both(fig),@()load_cal(fig,''),@()calibrate(fig), ...
       @()save_cal(fig,''),@()test_folder(fig,1),@()test_folder(fig,2), ...
       @()add_images(fig),@()export_csv(fig),@()clear_images(fig)};
for i = 1:numel(lbl)
    f = fns{i};
    uicontrol(fig,'Style','pushbutton','String',lbl{i},'Units','normalized', ...
        'Position',[0.01 0.94-0.043*(i-1) 0.12 0.04], ...
        'Callback',@(s,e)run_cb(fig,f),'BusyAction','cancel','Interruptible','off');
end

A.hp = gobjects(1,4);
A.ax = cell(1,4);
defSel = [1 2 4 7];                                % FUSED, Spectral, Cross-channel, ISP noise
for k = 1:4
    c = mod(k-1,2); r = floor((k-1)/2);
    x0 = 0.175 + c*0.215;  y0 = 0.555 - r*0.48;
    A.hp(k) = uicontrol(fig,'Style','popupmenu','String',plotNames,'Value',defSel(k), ...
        'Units','normalized','Position',[x0 y0+0.355 0.17 0.035], ...
        'Callback',@(s,e)run_cb(fig,@()redraw(fig)));
    A.ax{k} = axes('Parent',fig,'Position',[x0 y0 0.17 0.30]);
end
uicontrol(fig,'Style','text','String','region map X','Units','normalized','Position',[0.01 0.475 0.12 0.022]);
A.hx = uicontrol(fig,'Style','popupmenu','String',plotNames,'Value',find(strcmp(plotNames,'nlf_var')), ...
    'Units','normalized','Position',[0.01 0.443 0.12 0.032],'Callback',@(s,e)run_cb(fig,@()redraw(fig)));
uicontrol(fig,'Style','text','String','region map Y','Units','normalized','Position',[0.01 0.41 0.12 0.022]);
A.hy = uicontrol(fig,'Style','popupmenu','String',plotNames,'Value',find(strcmp(plotNames,'cl_ratio')), ...
    'Units','normalized','Position',[0.01 0.378 0.12 0.032],'Callback',@(s,e)run_cb(fig,@()redraw(fig)));
A.hs  = uicontrol(fig,'Style','text','String','','Units','normalized', ...
    'Position',[0.01 0.02 0.12 0.35],'HorizontalAlignment','left');
A.axr = axes('Parent',fig,'Position',[0.635 0.08 0.35 0.80]);

A.O = O; A.FT = FT;
A.colNames = colNames; A.plotCol = plotCol; A.plotNames = plotNames;
A.X = zeros(0,nF); A.D = zeros(0,nF+nG+1);
A.cls = zeros(0,1); A.path = cell(0,1); A.src = cell(0,1);
A.isJ = false(0,1); A.npix = zeros(0,1);
A.Cal = []; A.stale = false;
A.T = empty_tests(nF);                             % images from the test folders (never used to calibrate)
A.calFile = fullfile(fileparts(mfilename('fullpath')),'aigi_fusion_calibration.mat');
setappdata(fig,'A',A);

fprintf('\n==================================================================\n');
fprintf('  AIGI FUSION DETECTOR  (6 tests, 19 features, calibrated fusion)\n');
fprintf('==================================================================\n');
if exist(A.calFile,'file')
    fprintf('found saved calibration: %s\n',A.calFile);
    run_cb(fig,@()load_cal(fig,A.calFile));
else
    fprintf('no saved calibration yet: add a REAL and an AI folder, then press Calibrate.\n');
end
redraw(fig);
end

%% =====================================================================================
%  SETTINGS
%  =====================================================================================
function O = default_options()
O.cropSide     = 512;        % centre-crop side (px); smaller images used whole
O.rejpeg       = 90;         % JPEG re-save quality before measuring | 0 = off
O.minSide      = 96;         % images smaller than this are skipped
O.blk          = 16;         % ISP noise block size
O.minBlocksBin = 15;         % ISP noise: min blocks per brightness bin
O.nFeat        = 19;
O.kfold        = 5;          % cross-validation folds
O.seed         = 1;          % CV split seed
O.lambdas      = 10.^(-4:0.5:1);   % ridge strengths searched by CV
O.zclip        = 6;          % robust z-scores clipped to +/- this
O.targetAcc    = 0.95;       % wanted balanced accuracy of the confident (non-UNCERTAIN) verdicts
O.minCoverage  = 0.70;       % ...but at least this fraction of images must get a verdict
O.purity       = 0.8;        % region-map purity
O.chunk        = 48;         % images per parallel chunk (progress granularity)
O.exts = {'.jpg','.jpeg','.jfif','.png','.tif','.tiff','.bmp','.webp'};
end

function FT = feature_table()
FT.names = {'sp_grid_dB','sp_slope','sp_rolloff','nyq_dB', ...
            'cc_corrRG','cc_corrGB','cc_corrRB','cc_Gratio','cfa_phaseG','cfa_phaseRB', ...
            'co_clip','co_full','co_comb','co_satmean','co_satstd', ...
            'tex_score','rc_disp','nlf_var','cl_ratio'};
FT.grp   = [1 1 1 2 3 3 3 3 3 3 4 4 4 4 4 5 6 6 6];
FT.gnames = {'Spectral','Nyquist CFA','Cross-channel/CFA','Colour histogram','Texture contrast','ISP noise'};
end

%% =====================================================================================
%  CALLBACKS
%  =====================================================================================
function run_cb(fig,fn)
try
    fn();
catch err
    fprintf(2,'\n[!] %s\n',err.message);
    for q = 1:min(4,numel(err.stack))
        fprintf(2,'    in %s (line %d)\n',err.stack(q).name,err.stack(q).line);
    end
    if ishandle(fig)
        errordlg(err.message,'AIGI fusion detector');
        try, redraw(fig); catch, end
    end
end
end

function add_folder(fig,cls)
% pick all folders first, then measure them one after another (no dialog in between)
lab={'REAL','AI'};
D=pick_folders(sprintf('Select %s image folder(s) - Ctrl/Shift-click for several (subfolders included)',lab{cls}));
if isempty(D), return; end
fprintf('\n%d %s folder(s) selected:\n',numel(D),lab{cls});
fprintf('  %s\n',D{:});
for k=1:numel(D)
    if ~ishandle(fig), return; end
    add_one_folder(fig,cls,D{k});
end
end

function add_both(fig)
% pick REAL folders, then AI folders, then measure all of them (no dialog in between)
DR=pick_folders('Select REAL image folder(s) - Ctrl/Shift-click for several (subfolders included)');
p0=pwd;
if ~isempty(DR), p0=fileparts(DR{1}); end
DA=pick_folders('Select AI image folder(s) - Ctrl/Shift-click for several (subfolders included)',p0);
if isempty(DR) && isempty(DA), return; end
fprintf('\n%d REAL folder(s) selected:\n',numel(DR));
if ~isempty(DR), fprintf('  %s\n',DR{:}); end
fprintf('%d AI folder(s) selected:\n',numel(DA));
if ~isempty(DA), fprintf('  %s\n',DA{:}); end
D=[DR(:);DA(:)];
C=[ones(numel(DR),1);2*ones(numel(DA),1)];
for k=1:numel(D)
    if ~ishandle(fig), return; end
    add_one_folder(fig,C(k),D{k});
end
end

function D = pick_folders(ttl,p0)
% multi-select folder dialog; falls back to repeated uigetdir (Cancel when done)
if nargin<2 || isempty(p0), p0=pwd; end
D={};
if usejava('swing')
    try
        jc=javaObjectEDT('javax.swing.JFileChooser',p0);
        jc.setDialogTitle(ttl);
        jc.setFileSelectionMode(javax.swing.JFileChooser.DIRECTORIES_ONLY);
        jc.setMultiSelectionEnabled(true);
        if jc.showOpenDialog([])~=javax.swing.JFileChooser.APPROVE_OPTION, return; end
        sel=jc.getSelectedFiles();
        for k=1:numel(sel)
            D{end+1,1}=char(sel(k).getAbsolutePath()); %#ok<AGROW>
        end
        if isempty(D)                               % opened a folder and pressed Open
            f=jc.getSelectedFile();
            if ~isempty(f), D={char(f.getAbsolutePath())}; end
        end
        D=unique(D,'stable');
        return;
    catch
        D={};
    end
end
while true
    d=uigetdir(p0,sprintf('%s  [%d chosen, Cancel when done]',ttl,numel(D)));
    if isequal(d,0), break; end
    D{end+1,1}=d; %#ok<AGROW>
end
D=unique(D,'stable');
end

function add_one_folder(fig,cls,d)
A = getappdata(fig,'A');
lab = {'REAL','AI'};
files = list_images(d,A.O.exts);
if isempty(files)
    warndlg(sprintf('No supported images in this folder or its subfolders:\n%s',d)); return;
end
isNew = ~ismember(lower(files),lower(A.path));
if any(~isNew)
    fprintf('\n%d image(s) from %s are already loaded - skipped (not re-measured)\n',nnz(~isNew),d);
end
files = files(isNew);
if isempty(files), return; end
set_status(A,{'measuring...',d});
[X,J,NP,ok] = measure_files(files,A.O,sprintf('%s folder: %s',lab{cls},d));
if ~ishandle(fig), return; end
A = getappdata(fig,'A');
if any(~ok), fprintf('  %d image(s) could not be measured and were left out\n',nnz(~ok)); end
n = nnz(ok);
A.X    = [A.X; X(ok,:)];
A.cls  = [A.cls; cls*ones(n,1)];
A.path = [A.path; files(ok)];
A.src  = [A.src; repmat({d},n,1)];
A.isJ  = [A.isJ; J(ok)];
A.npix = [A.npix; NP(ok)];
A.stale = ~isempty(A.Cal) || A.stale;
A.D = derived_all(A);
setappdata(fig,'A',A);
fprintf('added %s folder %s: %d images (originally JPEG: %.0f%%)\n',lab{cls},d,n,100*mean(J(ok)));
print_loaded(A);
if isempty(A.Cal)
    fprintf('-> add the other class if needed, then press "Calibrate / recalibrate".\n');
else
    fprintf('-> new data is not used by the model until you press "Calibrate / recalibrate".\n');
end
redraw(fig);
end

function add_images(fig)
A = getappdata(fig,'A');
flt = {'*.jpg;*.jpeg;*.jfif;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'};
[f,p] = uigetfile(flt,'Select image(s) to test','MultiSelect','on');
if isequal(f,0), return; end
if ischar(f), f = {f}; end
files = cellfun(@(x)fullfile(p,x),f(:),'UniformOutput',false);
set_status(A,{'measuring single images...'});
[X,J,NP,ok] = measure_files(files,A.O,'single image(s)');
if ~ishandle(fig), return; end
A = getappdata(fig,'A');
for i = find(~ok(:))', fprintf('  [!] could not measure %s\n',files{i}); end
n0 = size(A.X,1); nPrev = nnz(A.cls==3); n = nnz(ok);
A.X    = [A.X; X(ok,:)];
A.cls  = [A.cls; 3*ones(n,1)];
A.path = [A.path; files(ok)];
A.src  = [A.src; repmat({'single'},n,1)];
A.isJ  = [A.isJ; J(ok)];
A.npix = [A.npix; NP(ok)];
A.D = derived_all(A);
setappdata(fig,'A',A);
for i = 1:n
    print_single(A,n0+i,nPrev+i);
end
print_single_summary(A,n0+(1:n),nPrev+(1:n));
redraw(fig);
end

function clear_images(fig)
A = getappdata(fig,'A');
k = A.cls~=3;
A.X = A.X(k,:); A.D = A.D(k,:); A.cls = A.cls(k); A.path = A.path(k);
A.src = A.src(k); A.isJ = A.isJ(k); A.npix = A.npix(k);
setappdata(fig,'A',A);
fprintf('\nsingle images cleared (calibration data kept)\n');
redraw(fig);
end

function calibrate(fig)
A = getappdata(fig,'A');
tr = find(A.cls==1 | A.cls==2);
y = double(A.cls(tr)==2);                         % 1 = AI
if nnz(y==0) < 10 || nnz(y==1) < 10
    warndlg(sprintf('Need at least 10 REAL and 10 AI images (have %d / %d).',nnz(y==0),nnz(y==1)));
    return;
end
set_status(A,{'calibrating...'});
t0 = tic;
Cal = fit_calibration(A.X(tr,:),y,A.FT,A.O);
Cal.trainPaths = A.path(tr);
Cal.trainSrc   = A.src(tr);
Cal.date = char(datetime('now','Format','yyyy-MM-dd HH:mm'));
Cal.secs = toc(t0);
A.Cal = Cal; A.stale = false;
A.D = derived_all(A);
setappdata(fig,'A',A);
print_calibration(A);
save_cal(fig,A.calFile);
redraw(fig);
% re-print verdicts of single images already on screen with the new model
s = find(A.cls==3);
if ~isempty(s)
    fprintf('\nsingle images re-scored with the new calibration:\n');
    for i = 1:numel(s), print_single(A,s(i),i); end
    print_single_summary(A,s,1:numel(s));
end
if ~isempty(A.T.cls)
    fprintf('\ntest folders re-scored with the new calibration:\n');
    print_test_totals(A);
end
end

function save_cal(fig,f)
A = getappdata(fig,'A');
k = A.cls==1 | A.cls==2;
if ~any(k), warndlg('No REAL/AI data to save yet.'); return; end
if isempty(f)
    [fn,p] = uiputfile('*.mat','Save calibration data',A.calFile);
    if isequal(fn,0), return; end
    f = fullfile(p,fn);
end
S.version   = 1;
S.O         = A.O;
S.featNames = A.FT.names;
S.X    = A.X(k,:);   S.cls  = A.cls(k);  S.path = A.path(k);
S.src  = A.src(k);   S.isJ  = A.isJ(k);  S.npix = A.npix(k);
S.Cal  = A.Cal;      S.stale = A.stale;
S.saved = char(datetime('now','Format','yyyy-MM-dd HH:mm'));
save(f,'-struct','S','-v7');
fprintf('\ncalibration data saved: %s  (%d real, %d AI%s)\n',f,nnz(S.cls==1),nnz(S.cls==2), ...
    tern(isempty(S.Cal),', no model yet',''));
end

function load_cal(fig,f)
A = getappdata(fig,'A');
if isempty(f)
    [fn,p] = uigetfile('*.mat','Load calibration data',A.calFile);
    if isequal(fn,0), return; end
    f = fullfile(p,fn);
end
S = load(f);
if ~all(isfield(S,{'X','cls','path','featNames','O'})) || ~isequal(S.featNames,A.FT.names)
    error('%s was not made by this version of aigi_fusion_gui (feature list differs).',f);
end
if ~isfield(S,'src'), S.src = cellfun(@fileparts,S.path,'UniformOutput',false); end
hadCalData = any(A.cls==1 | A.cls==2);
if hadCalData
    if ~same_opts(A.O,S.O)
        error(['The loaded data used different pre-processing (crop %d / JPEG q%d) than the ' ...
               'current data (crop %d / JPEG q%d); they cannot be mixed.'], ...
               S.O.cropSide,S.O.rejpeg,A.O.cropSide,A.O.rejpeg);
    end
else
    if ~same_opts(A.O,S.O) && any(A.cls==3)
        fprintf('pre-processing changes to the file''s settings: single images cleared, add them again.\n');
        k = A.cls~=3;
        A.X = A.X(k,:); A.cls = A.cls(k); A.path = A.path(k); A.src = A.src(k);
        A.isJ = A.isJ(k); A.npix = A.npix(k);
    end
    if ~same_opts(A.O,S.O) && ~isempty(A.T.cls)
        fprintf('pre-processing changes to the file''s settings: test-folder results cleared, add them again.\n');
        A.T = empty_tests(numel(A.FT.names));
    end
    fl = {'cropSide','rejpeg','minSide','blk','minBlocksBin'};
    for i = 1:numel(fl), A.O.(fl{i}) = S.O.(fl{i}); end
end
isNew = ~ismember(lower(S.path),lower(A.path));
n = nnz(isNew);
A.X    = [A.X; S.X(isNew,:)];
A.cls  = [A.cls; S.cls(isNew)];
A.path = [A.path; S.path(isNew)];
A.src  = [A.src; S.src(isNew)];
A.isJ  = [A.isJ; S.isJ(isNew)];
A.npix = [A.npix; S.npix(isNew)];
if ~hadCalData
    A.Cal = S.Cal;
    A.stale = isfield(S,'stale') && S.stale;
else
    A.stale = true;                               % merged two data sets: model must be refit
end
A.D = derived_all(A);
setappdata(fig,'A',A);
fprintf('\nloaded %s: %d new rows (%d real, %d AI), %d already present\n',f,n, ...
    nnz(S.cls(isNew)==1),nnz(S.cls(isNew)==2),nnz(~isNew));
if ~isempty(A.Cal) && ~A.stale
    print_calibration(A);
elseif A.stale
    fprintf('-> data changed since the stored model: press "Calibrate / recalibrate".\n');
end
if ~isempty(A.T.cls) && ~isempty(A.Cal) && ~hadCalData
    print_test_totals(A);
end
redraw(fig);
end

%% =====================================================================================
%  TEST FOLDERS (evaluation of the calibrated detector on new, labelled images)
%  =====================================================================================
function T = empty_tests(nF)
T.X = zeros(0,nF); T.cls = zeros(0,1); T.path = cell(0,1); T.src = cell(0,1);
end

function test_folder(fig,cls)
A = getappdata(fig,'A');
lab = {'REAL','AI'};
if isempty(A.Cal)
    warndlg('Calibrate first (or load a calibration file) - test folders are scored with the calibration.');
    return;
end
d = uigetdir(pwd,sprintf('Select a %s TEST folder (subfolders included)',lab{cls}));
if isequal(d,0), return; end
files = list_images(d,A.O.exts);
if isempty(files), warndlg('No supported images in this folder or its subfolders.'); return; end
inCal = ismember(lower(files),lower(A.Cal.trainPaths));
if any(inCal)
    fprintf(['\n[!] %d image(s) in %s were used for calibration and are left out of the test\n' ...
             '    (testing on calibration images would overstate the efficiency)\n'],nnz(inCal),d);
end
files = files(~inCal);
done = ismember(lower(files),lower(A.T.path));
if any(done), fprintf('%d image(s) were already tested earlier - skipped\n',nnz(done)); end
files = files(~done);
if isempty(files), return; end
set_status(A,{'testing...',d});
[X,~,~,ok] = measure_files(files,A.O,sprintf('%s TEST folder: %s',lab{cls},d));
if ~ishandle(fig), return; end
A = getappdata(fig,'A');
nFail = nnz(~ok);
X = X(ok,:); files = files(ok);
A.T.X    = [A.T.X; X];
A.T.cls  = [A.T.cls; cls*ones(numel(files),1)];
A.T.path = [A.T.path; files];
A.T.src  = [A.T.src; repmat({d},numel(files),1)];
setappdata(fig,'A',A);
print_test_folder(A,cls,d,X,files,nFail);
print_test_totals(A);
redraw(fig);
end

function print_test_folder(A,cls,d,X,files,nFail)
C = A.Cal; FT = A.FT; nG = numel(FT.gnames);
lab = {'REAL','AI'};
n = size(X,1);
L1 = repmat('=',1,96); L2 = repmat('-',1,96);
fprintf('\n%s\n TEST RESULT - %s folder: %s\n',L1,lab{cls},d);
fprintf(' calibration of %s (%s fusion, %d real + %d AI)%s\n',C.date,C.kind,C.n(1),C.n(2), ...
    tern(A.stale,'  [data added since - not yet recalibrated]',''));
fprintf('%s\n',L1);
if n == 0, fprintf(' no image could be measured (%d failed)\n%s\n',nFail,L1); return; end
[G,s,p] = apply_model(C,X);
v = arrayfun(@(t)verdict_of(C,t),s,'UniformOutput',false);
want = tern(cls==2,'AI-GENERATED','REAL');
nOK  = nnz(strcmp(v,want));
nUnc = nnz(strcmp(v,'UNCERTAIN'));
nBad = n - nOK - nUnc;
nDec = nOK + nBad;
forced = mean((s >= C.thr) == (cls==2));
fprintf(' images tested            : %d%s\n',n,tern(nFail>0,sprintf('   (%d could not be read, not counted)',nFail),''));
fprintf(' detected correctly as %-4s: %5d   (%5.1f%%)\n',tern(cls==2,'AI','REAL'),nOK,100*nOK/n);
fprintf(' detected wrongly as %-6s: %5d   (%5.1f%%)\n',tern(cls==2,'REAL','AI'),nBad,100*nBad/n);
fprintf(' UNCERTAIN                : %5d   (%5.1f%%)\n',nUnc,100*nUnc/n);
fprintf('%s\n',L2);
fprintf(' EFFICIENCY (correct / all tested)        : %5.1f%%\n',100*nOK/n);
fprintf(' efficiency on decided images (no UNCERT.) : %5.1f%%   (%d decided)\n',100*nOK/max(nDec,1),nDec);
fprintf(' forced decision, no uncertain band        : %5.1f%%\n',100*forced);
fprintf(' mean P(AI)                                : %.3f\n',mean(p));
fprintf('%s\n',L2);
fprintf(' each test on its own (share called %s correctly at its calibrated threshold):\n',lab{cls});
for g = 1:nG
    fprintf('   TEST %d %-22s %5.1f%%   (weight %4.1f%%)\n',g,FT.gnames{g}, ...
        100*mean((G(:,g) >= C.grp(g).thr) == (cls==2)),C.testW(g));
end
wrong = find(~strcmp(v,want));
if ~isempty(wrong)
    [~,o] = sort(abs(s(wrong)-C.thr),'descend');
    wrong = wrong(o);
    nShow = min(15,numel(wrong));
    fprintf('%s\n not detected correctly (%d, most confident mistakes first%s):\n',L2,numel(wrong), ...
        tern(numel(wrong)>nShow,sprintf(', first %d shown',nShow),''));
    for i = 1:nShow
        k = wrong(i);
        [~,nm,ex] = fileparts(files{k});
        fprintf('   %-13s P(AI) %.3f   %s%s\n',v{k},p(k),nm,ex);
    end
end
fprintf('%s\n',L1);
end

function print_test_totals(A)
% all test images so far, scored with the current calibration
C = A.Cal; T = A.T;
if isempty(C) || isempty(T.cls), return; end
keep = ~ismember(lower(T.path),lower(C.trainPaths));      % never count calibration images
if any(~keep)
    fprintf(' [i] %d test image(s) are now part of the calibration data and are not counted\n',nnz(~keep));
end
X = T.X(keep,:); cls = T.cls(keep);
if isempty(cls), return; end
[~,s] = apply_model(C,X);
v = arrayfun(@(t)verdict_of(C,t),s,'UniformOutput',false);
isR = cls==1; isA = cls==2;
M = zeros(2,3);                                   % rows REAL/AI, cols called REAL / AI / UNCERTAIN
cats = {'REAL','AI-GENERATED','UNCERTAIN'};
for j = 1:3
    M(1,j) = nnz(isR & strcmp(v,cats{j}));
    M(2,j) = nnz(isA & strcmp(v,cats{j}));
end
nR = nnz(isR); nA = nnz(isA);
L1 = repmat('#',1,96);
fprintf('\n%s\n ALL TEST FOLDERS SO FAR  (%d REAL, %d AI test images)\n%s\n',L1,nR,nA,repmat('-',1,96));
fprintf('                    called REAL   called AI   UNCERTAIN\n');
fprintf('   REAL images      %10d  %10d  %10d\n',M(1,:));
fprintf('   AI images        %10d  %10d  %10d\n',M(2,:));
fprintf('%s\n',repmat('-',1,96));
eR = M(1,1)/max(nR,1); eA = M(2,2)/max(nA,1);
if nR > 0, fprintf('   REAL detected correctly : %5d / %-5d  = %5.1f%%\n',M(1,1),nR,100*eR); end
if nA > 0, fprintf('   AI   detected correctly : %5d / %-5d  = %5.1f%%\n',M(2,2),nA,100*eA); end
tot = nR + nA; dec = tot - sum(M(:,3));
fprintf('   OVERALL EFFICIENCY      : %5d / %-5d  = %5.1f%%\n',M(1,1)+M(2,2),tot,100*(M(1,1)+M(2,2))/tot);
if nR > 0 && nA > 0
    fprintf('   BALANCED EFFICIENCY     : %5.1f%%   (average of REAL and AI, fair when counts differ)\n',50*(eR+eA));
end
fprintf('   on decided images only  : %5.1f%%   (%d of %d decided, %d uncertain)\n', ...
    100*(M(1,1)+M(2,2))/max(dec,1),dec,tot,sum(M(:,3)));
fr = (s >= C.thr) == isA;
fprintf('   forced decision (no band): %5.1f%%', 100*mean(fr));
if nR > 0 && nA > 0
    fprintf('   balanced %5.1f%%   AUC %.3f',50*(mean(fr(isR))+mean(fr(isA))),auc_rank(s(isA),s(isR)));
end
fprintf('\n%s\n',L1);
end

function print_single_summary(A,rows,nums)
if isempty(rows), return; end
fprintf('\n---------------- SINGLE IMAGE VERDICTS ----------------\n');
for i = 1:numel(rows)
    [~,nm,ex] = fileparts(A.path{rows(i)});
    if isempty(A.Cal)
        fprintf(' [%d] %-14s %s%s\n',nums(i),'NO VERDICT',nm,ex);
    else
        [v,p] = verdict_of(A.Cal,A.D(rows(i),end));
        fprintf(' [%d] %-14s P(AI) %.3f   %s%s\n',nums(i),v,p,nm,ex);
    end
end
if isempty(A.Cal)
    fprintf(' (calibrate or load a calibration to get REAL / AI / UNCERTAIN verdicts)\n');
end
fprintf('-------------------------------------------------------\n');
end

function export_csv(fig)
A = getappdata(fig,'A');
if isempty(A.cls), warndlg('Nothing to export yet.'); return; end
[f,p] = uiputfile('*.csv','Save results','aigi_fusion_results.csv');
if isequal(f,0), return; end
cl = {'real','AI','single'};
fid = fopen(fullfile(p,f),'w');
if fid < 0, error('cannot write %s',fullfile(p,f)); end
c = onCleanup(@()fclose(fid));
fprintf(fid,'file,class,source,original_jpeg,megapixels,%s,verdict,P_AI\n',strjoin(strrep(A.colNames,',',' '),','));
for i = 1:numel(A.cls)
    fprintf(fid,'"%s",%s,"%s",%d,%.3f',A.path{i},cl{A.cls(i)},A.src{i},A.isJ(i),A.npix(i)/1e6);
    fprintf(fid,',%.6g',A.D(i,:));
    if isempty(A.Cal)
        fprintf(fid,',,\n');
    else
        [v,pp] = verdict_of(A.Cal,A.D(i,end));
        fprintf(fid,',%s,%.4f\n',v,pp);
    end
end
% images from the test folders
if ~isempty(A.T.cls)
    nG = numel(A.FT.gnames); Dt = [A.T.X nan(numel(A.T.cls),nG+1)];
    if ~isempty(A.Cal), [G,sc] = apply_model(A.Cal,A.T.X); Dt(:,end-nG:end) = [G sc]; end
    tl = {'test_real','test_AI'};
    for i = 1:numel(A.T.cls)
        fprintf(fid,'"%s",%s,"%s",,',A.T.path{i},tl{A.T.cls(i)},A.T.src{i});
        fprintf(fid,',%.6g',Dt(i,:));
        if isempty(A.Cal)
            fprintf(fid,',,\n');
        else
            [v,pp] = verdict_of(A.Cal,Dt(i,end));
            fprintf(fid,',%s,%.4f\n',v,pp);
        end
    end
end
fprintf('\nsaved %s  (fused/test scores of calibration images are out-of-fold)\n',fullfile(p,f));
end

%% =====================================================================================
%  MEASUREMENT (parallel)
%  =====================================================================================
function [X,J,NP,ok] = measure_files(files,O,label)
n = numel(files);
X = nan(n,O.nFeat); J = false(n,1); NP = nan(n,1); ok = false(n,1);
M = pool_size(n);
fprintf('\n%s\n  measuring %d image(s)  (parallel workers: %d)\n',label,n,M);
wb = [];
if n > 1, wb = waitbar(0,sprintf('0/%d images',n),'Name','measuring'); end
cw = onCleanup(@()close_if(wb));
t0 = tic;
step = max(O.chunk,4*M);
for a = 1:step:n
    idx = a:min(n,a+step-1);
    m = numel(idx);
    fc = files(idx);
    Xc = nan(m,O.nFeat); Jc = false(m,1); Nc = nan(m,1); Oc = false(m,1);
    parfor (ii = 1:m, M)
        [xi,ji,ni,oi] = one_image(fc{ii},O);
        Xc(ii,:) = xi; Jc(ii) = ji; Nc(ii) = ni; Oc(ii) = oi;
    end
    X(idx,:) = Xc; J(idx) = Jc; NP(idx) = Nc; ok(idx) = Oc;
    done = idx(end); el = toc(t0);
    msg = sprintf('%d/%d images  (%.0f s, ~%.0f s left)',done,n,el,el/done*(n-done));
    if n > 1, fprintf('  %s\n',msg); end
    if ~isempty(wb) && ishandle(wb), waitbar(done/n,wb,msg); end
end
end

function close_if(h)
if ~isempty(h) && ishandle(h), close(h); end
end

function M = pool_size(n)
M = 0;
if n < 4, return; end                       % not worth the pool for a few images
try
    if license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'))
        p = gcp('nocreate');
        if isempty(p)
            fprintf('  starting parallel pool (first time only)...\n');
            p = parpool;
        end
        M = p.NumWorkers;
    end
catch err
    fprintf('  parallel pool unavailable (%s) - running serially\n',err.message);
    M = 0;
end
end

function s = workers_str()
s = 'not available';
try
    if license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'))
        p = gcp('nocreate');
        if isempty(p), s = 'starts with 1st folder'; else, s = sprintf('%d',p.NumWorkers); end
    end
catch
end
end

function [x,isJ,np,ok] = one_image(f,O)
x = nan(1,O.nFeat); isJ = false; np = NaN; ok = false;
try
    [I,isJ,np] = load_crop(f,O.cropSide);
    if min(size(I,1),size(I,2)) < O.minSide
        error('smaller than %d px',O.minSide);
    end
    I = resave_jpeg(I,O.rejpeg);
    x = all_features(I,O);
    ok = any(isfinite(x));
catch err
    fprintf('  skipped %s: %s\n',f,err.message);
end
end

function x = all_features(I,O)
% I: double 0..255, 1 or 3 channels
if size(I,3) == 1, rgb = repmat(I,[1 1 3])/255; else, rgb = I/255; end
g = rgb(:,:,2);
x = nan(1,O.nFeat);
x(1:3)   = safe_call(@()feat_spectral(g),3);
x(4)     = safe_call(@()feat_nyquist(g),1);
x(5:10)  = safe_call(@()feat_crosschannel(rgb),6);
x(11:15) = safe_call(@()feat_colour(rgb),5);
x(16)    = safe_call(@()feat_texture(g),1);
x(17:19) = safe_call(@()isp_params(I,O),3);
end

function v = safe_call(fn,n)
try
    v = double(fn());
    v = v(:)';
    if numel(v) ~= n, v = nan(1,n); end
catch
    v = nan(1,n);
end
end

%% =====================================================================================
%  TEST 1: SPECTRAL  (spectral_core.m, raw features only)
%  =====================================================================================
function v = feat_spectral(g)
[H,W] = size(g);
S  = min([256, 2*floor(H/2), 2*floor(W/2)]);
TC = sp_tile_coords(H,W,S,5);
E  = nan(size(TC,1),3);
for t = 1:size(TC,1)
    try
        E(t,:) = sp_f_spectral(g(TC(t,1)+(0:S-1),TC(t,2)+(0:S-1)));
    catch
    end
end
v = sp_colmean_nan(E);
end

function out = sp_f_spectral(x)
out = nan(1,3);
[h,w] = size(x);
r = x - medfilt2(x,[3 3],'symmetric');
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
out(1) = 20*log10(mean(gv)/md);
L = 20*log10(abs(fftshift(fft2((x-mean(x(:))).*sp_hann2(h,w)))) + eps);
[prof,rc] = sp_radial_profile(L,48);
ok  = isfinite(prof);
mid = ok & rc>=0.25 & rc<=0.65;
hi  = ok & rc> 0.80;
lo  = ok & rc>=0.40 & rc<=0.60;
if nnz(mid) >= 5
    cfit = polyfit(log10(rc(mid)),prof(mid),1);
    out(2) = cfit(1);
end
if nnz(hi) >= 3 && nnz(lo) >= 3
    out(3) = mean(prof(hi)) - mean(prof(lo));
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
TC = unique(TC,'rows','stable');
end

function m = sp_colmean_nan(A)
m = nan(1,size(A,2));
for j = 1:size(A,2)
    v = A(:,j); v = v(isfinite(v));
    if ~isempty(v), m(j) = mean(v); end
end
end

%% =====================================================================================
%  TEST 2: NYQUIST CFA  (Test07)
%  =====================================================================================
function score = feat_nyquist(g)
LAP = [0 1 0; 1 -4 1; 0 1 0];
e = conv2(g,LAP,'same');
p = e.^2;
p = p - conv2(p,ones(9)/81,'same');
p = p(9:end-8,9:end-8);
[h,w] = size(p);
h = 2*floor(h/2); w = 2*floor(w/2);
p = p(1:h,1:w);
P = abs(fft2(p .* sp_hann2(h,w)));
iy = h/2 + 1; ix = w/2 + 1;
peakVal = max(max(P(iy-1:iy+1,ix-1:ix+1)));
score = 20*log10((peakVal + eps)/(median(P(:)) + eps));
end

%% =====================================================================================
%  TEST 3: CROSS-CHANNEL + CFA PHASE  (Test10)
%  =====================================================================================
function v = feat_crosschannel(rgb)
res = zeros(size(rgb));
for k = 1:3
    res(:,:,k) = rgb(:,:,k) - medfilt2(rgb(:,:,k),[3 3],'symmetric');
end
rR = res(:,:,1); rG = res(:,:,2); rB = res(:,:,3);
cRG = safecorr(rR,rG); cGB = safecorr(rG,rB); cRB = safecorr(rR,rB);
eR = mean(rR(:).^2); eG = mean(rG(:).^2); eB = mean(rB(:).^2);
Gratio = 10*log10((eG + eps)/(0.5*(eR + eB) + eps));
K = [0 .25 0; .25 0 .25; 0 .25 0];
phG  = phase_imbalance(rgb(:,:,2),K);
phRB = 0.5*(phase_imbalance(rgb(:,:,1),K) + phase_imbalance(rgb(:,:,3),K));
v = [cRG cGB cRB Gratio phG phRB];
end

function c = safecorr(a,b)
a = a(:); b = b(:);
if numel(a) > 200000
    s = round(linspace(1,numel(a),200000));
    a = a(s); b = b(s);
end
if std(a) < eps || std(b) < eps, c = NaN; return; end
R = corrcoef(a,b);
c = R(1,2);
end

function v = phase_imbalance(x,K)
e = abs(x - conv2(x,K,'same'));
e = e(3:end-2,3:end-2);
[h,w] = size(e);
h = 2*floor(h/2); w = 2*floor(w/2);
e = e(1:h,1:w);
p = [mean(mean(e(1:2:end,1:2:end))), mean(mean(e(1:2:end,2:2:end))), ...
     mean(mean(e(2:2:end,1:2:end))), mean(mean(e(2:2:end,2:2:end)))];
v = (max(p) - min(p))/(max(p) + min(p) + eps);
end

%% =====================================================================================
%  TEST 4: COLOUR HISTOGRAM  (Test15)
%  =====================================================================================
function v = feat_colour(rgb)
u = im2uint8(rgb);
clip = mean(any(u == 0 | u == 255,3),'all');
full = 0;
for k = 1:3
    hk = histcounts(double(u(:,:,k)),-0.5:1:255.5);
    full = full + mean(hk > 0);
end
full = full/3;
yv = double(im2uint8(rgb2gray(rgb)));
hy = histcounts(yv,-0.5:1:255.5);
hy = hy/(sum(hy) + eps);
Hf = abs(fft(hy - mean(hy)));
comb = max(Hf(3:129))/(sum(hy) + eps);
mx = max(rgb,[],3); mn = min(rgb,[],3);
sat = (mx - mn)./(mx + eps);
v = [clip full comb mean(sat(:)) std(sat(:))];
end

%% =====================================================================================
%  TEST 5: TEXTURE CONTRAST  (PatchCraft-style, rich vs poor patch SRM energy)
%  =====================================================================================
function score = feat_texture(g)
ps = 32; np = 192;
SRM = [-1 2 -2 2 -1; 2 -6 8 -6 2; -2 8 -12 8 -2; 2 -6 8 -6 2; -1 2 -2 2 -1]/12;
[H,W] = size(g);
if H <= ps || W <= ps, score = NaN; return; end
rs = RandStream('mt19937ar','Seed',0);         % private stream: safe inside parfor
P = cell(1,np); d = zeros(1,np);
for k = 1:np
    i = randi(rs,H-ps); j = randi(rs,W-ps);
    q = g(i:i+ps-1,j:j+ps-1);
    P{k} = q;
    d(k) = sum(abs(diff(q,1,2)),'all') + sum(abs(diff(q,1,1)),'all') + ...
           sum(abs(q(2:end,2:end)-q(1:end-1,1:end-1)),'all') + ...
           sum(abs(q(2:end,1:end-1)-q(1:end-1,2:end)),'all');
end
[~,ord] = sort(d);
third = floor(np/3);
E  = @(C) mean(cellfun(@(q) mean(abs(conv2(q,SRM,'valid')),'all'),C));
ep = E(P(ord(1:third)));
er = E(P(ord(end-third+1:end)));
score = -(er - ep)/(er + ep + eps);
end

%% =====================================================================================
%  TEST 6: ISP NOISE PARAMETERS  (isp_param_viewer, 0..255 scale)
%  =====================================================================================
function x = isp_params(I,O)
if size(I,3) >= 3
    R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
    Y  = 0.299*R + 0.587*G + 0.114*B;
    Cb = -0.168736*R - 0.331264*G + 0.5*B;
    Cr = 0.5*R - 0.418688*G - 0.081312*B;
else
    Y = I(:,:,1); Cb = []; Cr = [];
end
x = [p_rc_disp(Y,O), p_nlf_var(Y,O), p_cl_ratio(Y,Cb,Cr,O)];
end

function v = p_rc_disp(Y,O)
v = NaN;
B = 2*O.blk;
r = Y - filter2(ones(3)/9,Y);
r = r(2:end-1,2:end-1);
Yc = Y(2:end-1,2:end-1);
Ph = zeros(size(r)); Pv = Ph;
Ph(:,1:end-1) = r(:,1:end-1).*r(:,2:end);
Pv(1:end-1,:) = r(1:end-1,:).*r(2:end,:);
e0 = bstat(r.^2,B,'mean');
eh = bstat(Ph,B,'mean');
ev = bstat(Pv,B,'mean');
S = filter2(ones(5)/25,Yc);
tex = sqrt(max(0,bstat(S.^2,B,'mean') - bstat(S,B,'mean').^2));
ok = bstat(Yc,B,'max') < 250 & bstat(Yc,B,'min') > 5 & e0 > 0.25;
if nnz(ok) < 20, return; end
f = ok & tex <= pct(tex(ok),0.25);
rho = (eh(f)./e0(f) + ev(f)./e0(f))/2;
if numel(rho) < 10, return; end
v = pct(rho,0.75) - pct(rho,0.25);
end

function v = p_nlf_var(Y,O)
v = NaN;
[sig,mu,mx,mn] = noise_blocks(Y,O.blk);
ok = mx < 250 & mn > 5;
ed = 0:16:256;
S = nan(numel(ed)-1,1);
for k = 1:numel(ed)-1
    in = ok & mu >= ed(k) & mu < ed(k+1);
    if nnz(in) >= O.minBlocksBin
        S(k) = pct(sig(in),0.10);
    end
end
S = S(~isnan(S));
if numel(S) >= 5 && mean(S) > 0.2
    v = (max(S) - min(S))/mean(S);
end
end

function v = p_cl_ratio(Y,Cb,Cr,O)
v = NaN;
if isempty(Cb), return; end
B = O.blk;
h = 2*floor(size(Y,1)/2); w = 2*floor(size(Y,2)/2);
Yh = half2(Y,h,w);
[sY,~,mx,mn] = noise_blocks(Yh,B);
sB = noise_blocks(half2(Cb,h,w),B);
sR = noise_blocks(half2(Cr,h,w),B);
S = filter2(ones(5)/25,Yh(2:end-1,2:end-1));
tex = sqrt(max(0,bstat(S.^2,B,'mean') - bstat(S,B,'mean').^2));
ok = mx < 250 & mn > 5;
if nnz(ok) < 20, return; end
f = ok & tex <= pct(tex(ok),0.25);
m = median(sY(f));
if m > 0.2
    v = median((sB(f) + sR(f))/2)/m;
end
end

function [sig,mu,mx,mn] = noise_blocks(X,B)
Rl = abs(filter2([1 -2 1;-2 4 -2;1 -2 1],X,'valid'));
sig = sqrt(pi/2)*bstat(Rl,B,'mean')/6;
if nargout > 1
    Xc = X(2:end-1,2:end-1);
    mu = bstat(Xc,B,'mean'); mx = bstat(Xc,B,'max'); mn = bstat(Xc,B,'min');
end
end

function M = bstat(X,B,fun)
[h,w] = size(X);
nr = floor(h/B); nc = floor(w/B);
X4 = reshape(X(1:nr*B,1:nc*B),B,nr,B,nc);
switch fun
    case 'mean', M = sum(sum(X4,1),3)/(B*B);
    case 'max',  M = max(max(X4,[],1),[],3);
    case 'min',  M = min(min(X4,[],1),[],3);
end
M = reshape(M,nr*nc,1);
end

function Z = half2(X,h,w)
Z = (X(1:2:h,1:2:w) + X(2:2:h,1:2:w) + X(1:2:h,2:2:w) + X(2:2:h,2:2:w))/4;
end

%% =====================================================================================
%  CALIBRATION
%  =====================================================================================
function Cal = fit_calibration(X,y,FT,O)
[n,nF] = size(X); nG = numel(FT.gnames);
folds = strat_folds(y,O.kfold,O.seed);
Cal = struct();
Cal.names = FT.names; Cal.gnames = FT.gnames; Cal.O = O;
Cal.n = [nnz(y==0) nnz(y==1)];
Cal.y = y;

% ---- single features -----------------------------------------------------
Cal.fAUC = nan(1,nF); Cal.fDir = ones(1,nF); Cal.fThr = nan(1,nF);
Cal.fBacc = nan(1,nF); Cal.fNaN = nan(1,nF);
Cal.fStatR = nan(3,nF); Cal.fStatA = nan(3,nF);
for k = 1:nF
    v = X(:,k); ok = isfinite(v);
    vr = v(ok & y==0); va = v(ok & y==1);
    Cal.fNaN(k) = 100*mean(~ok);
    Cal.fStatR(:,k) = q3(vr); Cal.fStatA(:,k) = q3(va);
    if numel(vr) >= 3 && numel(va) >= 3
        au = auc_rank(va,vr);                     % P(AI value > REAL value)
        Cal.fAUC(k) = au;
        Cal.fDir(k) = 1 - 2*(au < 0.5);
        [Cal.fThr(k),Cal.fBacc(k)] = best_cut(vr,va,Cal.fDir(k));
    end
end

% ---- one model per test --------------------------------------------------
oofG = nan(n,nG);
for g = 1:nG
    cols = find(FT.grp==g);
    [oofG(:,g),lam] = cv_lr(X(:,cols),y,folds,O.lambdas,false,O.zclip);
    P = prep_fit(X(:,cols),O.zclip);
    [w,b] = fit_lr(prep_apply(P,X(:,cols)),y,lam,false);
    Cal.grp(g).cols = cols; Cal.grp(g).P = P; Cal.grp(g).w = w; Cal.grp(g).b = b;
    Cal.grp(g).lam  = lam;
    Cal.grp(g).auc  = auc_rank(oofG(y==1,g),oofG(y==0,g));
    [Cal.grp(g).thr,Cal.grp(g).bacc] = best_cut(oofG(y==0,g),oofG(y==1,g),1);
end

% ---- fusion A: ridge-logistic on all features -----------------------------
[oofF,lamF] = cv_lr(X,y,folds,O.lambdas,false,O.zclip);
P = prep_fit(X,O.zclip);
[w,b] = fit_lr(prep_apply(P,X),y,lamF,false);
Cal.full = struct('P',P,'w',w,'b',b,'lam',lamF, ...
                  'auc',auc_rank(oofF(y==1),oofF(y==0)),'bacc',NaN);
[~,Cal.full.bacc] = best_cut(oofF(y==0),oofF(y==1),1);

% ---- fusion B: stacked (test scores -> non-negative meta weights) ---------
folds2 = strat_folds(y,O.kfold,O.seed+1);
[oofM,lamM] = cv_lr(oofG,y,folds2,O.lambdas,true,O.zclip);
Pm = prep_fit(oofG,O.zclip);
[wm,bm] = fit_lr(prep_apply(Pm,oofG),y,lamM,true);
Cal.meta = struct('P',Pm,'w',wm,'b',bm,'lam',lamM, ...
                  'auc',auc_rank(oofM(y==1),oofM(y==0)),'bacc',NaN);
[~,Cal.meta.bacc] = best_cut(oofM(y==0),oofM(y==1),1);

% ---- choose, threshold, uncertain band, probability scaling --------------
if Cal.meta.auc >= Cal.full.auc - 0.002
    Cal.kind = 'stacked'; s = oofM; Cal.aucChosen = Cal.meta.auc;
else
    Cal.kind = 'full';    s = oofF; Cal.aucChosen = Cal.full.auc;
end
[Cal.thr,Cal.bacc] = best_cut(s(y==0),s(y==1),1);
[Cal.band,Cal.cov,Cal.cacc] = pick_band(s,y,Cal.thr,O);
[a,c] = fit_logreg(s,y,1e-6);
Cal.platt = [a c];

% ---- weightage -----------------------------------------------------------
Cal.featW = 100*abs(Cal.full.w(:)')/max(sum(abs(Cal.full.w)),eps);
if strcmp(Cal.kind,'stacked')
    tw = max(Cal.meta.w(:)',0);
else
    tw = accumarray(FT.grp(:),abs(Cal.full.w(:)),[nG 1])';
end
Cal.testW = 100*tw/max(sum(tw),eps);

Cal.oofG = oofG; Cal.oofS = s; Cal.oofF = oofF; Cal.oofM = oofM;
end

function [G,s,p] = apply_model(Cal,X)
n = size(X,1); nG = numel(Cal.grp);
G = nan(n,nG);
for g = 1:nG
    M = Cal.grp(g);
    G(:,g) = prep_apply(M.P,X(:,M.cols))*M.w + M.b;
end
if strcmp(Cal.kind,'stacked')
    s = prep_apply(Cal.meta.P,G)*Cal.meta.w + Cal.meta.b;
else
    s = prep_apply(Cal.full.P,X)*Cal.full.w + Cal.full.b;
end
p = 1./(1 + exp(-(Cal.platt(1)*s + Cal.platt(2))));
end

function [v,p] = verdict_of(C,s)
p = 1/(1 + exp(-(C.platt(1)*s + C.platt(2))));
if ~isfinite(s),               v = 'UNCERTAIN';
elseif s >= C.thr + C.band,    v = 'AI-GENERATED';
elseif s <  C.thr - C.band,    v = 'REAL';
else,                          v = 'UNCERTAIN';
end
end

function D = derived_all(A)
nF = numel(A.FT.names); nG = numel(A.FT.gnames); n = size(A.X,1);
D = [A.X nan(n,nG+1)];
if isempty(A.Cal) || n == 0, return; end
[G,s] = apply_model(A.Cal,A.X);
D(:,nF+1:end) = [G s];
% images the model was trained on: show their honest out-of-fold scores
[tf,loc] = ismember(lower(A.path),lower(A.Cal.trainPaths));
tf = tf & A.cls ~= 3;
D(tf,nF+1:end) = [A.Cal.oofG(loc(tf),:) A.Cal.oofS(loc(tf))];
end

function f = strat_folds(y,K,seed)
rs = RandStream('mt19937ar','Seed',seed);
f = zeros(numel(y),1);
for c = [0 1]
    id = find(y==c);
    id = id(randperm(rs,numel(id)));
    f(id) = mod(0:numel(id)-1,K)' + 1;
end
end

function [oof,lam] = cv_lr(X,y,folds,L,nonneg,clip)
n = numel(y); K = max(folds);
S = nan(n,numel(L));
for f = 1:K
    te = folds==f; tr = ~te;
    P = prep_fit(X(tr,:),clip);
    Ztr = prep_apply(P,X(tr,:)); Zte = prep_apply(P,X(te,:));
    for j = 1:numel(L)
        [w,b] = fit_lr(Ztr,y(tr),L(j),nonneg);
        S(te,j) = Zte*w + b;
    end
end
loss = arrayfun(@(j)bal_logloss(S(:,j),y),1:numel(L));
[~,jb] = min(loss);
lam = L(jb); oof = S(:,jb);
end

function [w,b] = fit_lr(Z,y,lam,nonneg)
d = size(Z,2); w = zeros(d,1); act = true(d,1);
while true
    [wa,b] = fit_logreg(Z(:,act),y,lam);
    if ~nonneg || all(wa >= 0) || nnz(act) == 1
        w(act) = wa; break;
    end
    ia = find(act); [~,m] = min(wa); act(ia(m)) = false;   % drop the most negative test
end
end

function [w,b] = fit_logreg(Z,y,lam)
% class-balanced ridge logistic regression, Newton / IRLS (bias not penalised)
[n,d] = size(Z);
y = double(y(:));
sw = zeros(n,1);
n1 = nnz(y==1); n0 = n - n1;
sw(y==1) = 0.5/max(n1,1); sw(y==0) = 0.5/max(n0,1);
Xa = [Z ones(n,1)];
R = lam*diag([ones(d,1); 0]);
beta = zeros(d+1,1);
for it = 1:100
    p  = 1./(1 + exp(-(Xa*beta)));
    gr = Xa'*(sw.*(p - y)) + R*beta;
    Wd = sw.*max(p.*(1-p),1e-10);
    Hs = Xa'*(Xa.*Wd) + R + 1e-9*eye(d+1);
    st = Hs\gr;
    if norm(st) > 10, st = st*10/norm(st); end
    beta = beta - st;
    if max(abs(st)) < 1e-8, break; end
end
w = beta(1:d); b = beta(end);
end

function L = bal_logloss(s,y)
sp = @(t) max(t,0) + log1p(exp(-abs(t)));
L = 0.5*(mean(sp(-s(y==1))) + mean(sp(s(y==0))));
end

function P = prep_fit(X,clip)
d = size(X,2);
P.med = zeros(1,d); P.sc = ones(1,d); P.clip = clip;
for k = 1:d
    v = X(isfinite(X(:,k)),k);
    if isempty(v), continue; end
    m = median(v); s = 1.4826*median(abs(v - m));
    if s < 1e-12, s = std(v); end
    if ~(s > 1e-12), s = 1; end
    P.med(k) = m; P.sc(k) = s;
end
end

function Z = prep_apply(P,X)
Z = (X - P.med)./P.sc;
Z(~isfinite(Z)) = 0;                              % missing feature -> calibration median
Z = max(min(Z,P.clip),-P.clip);
end

function [t,ba] = best_cut(r,a,d)
% threshold with the best balanced accuracy; AI when d*x >= d*t
t = NaN; ba = NaN;
r = r(isfinite(r)); a = a(isfinite(a));
if numel(r) < 2 || numel(a) < 2, return; end
v = unique([r(:); a(:)]);
if numel(v) > 1500, v = unique(pct([r(:); a(:)],linspace(0,1,1500)')); end
if numel(v) < 2, t = v(1); ba = 0.5; return; end
c  = (v(1:end-1) + v(2:end))/2;
cs = d*c(:)';
bal = 0.5*(mean(d*a(:) >= cs,1) + mean(d*r(:) < cs,1));
[ba,i] = max(bal);
t = c(i);
end

function [band,cov,cacc] = pick_band(s,y,thr,O)
dist = abs(s - thr); pred = s >= thr;
cand = unique([0; pct(dist,(0.01:0.01:0.5)')]);
res = nan(numel(cand),2);
for i = 1:numel(cand)
    m = dist > cand(i) | cand(i) == 0;
    res(i,:) = [mean(m), bacc_of(pred,y,m)];
end
okc = res(:,1) >= O.minCoverage & isfinite(res(:,2));
hit = find(okc & res(:,2) >= O.targetAcc,1,'first');
if isempty(hit)
    tmp = res(:,2); tmp(~okc) = -Inf;
    [~,hit] = max(tmp);
end
band = cand(hit); cov = res(hit,1); cacc = res(hit,2);
end

function b = bacc_of(pred,y,m)
p1 = pred(m & y==1); p0 = pred(m & y==0);
if isempty(p1) || isempty(p0), b = NaN; return; end
b = 0.5*(mean(p1) + mean(~p0));
end

%% =====================================================================================
%  COMMAND-WINDOW REPORTS
%  =====================================================================================
function print_loaded(A)
r = A.cls==1; a = A.cls==2;
fprintf('\n=== loaded: %d real  %d AI  %d single ===\n',nnz(r),nnz(a),nnz(A.cls==3));
fprintf('  %-12s  %-30s  %-30s  %s\n','feature','real  median [IQR]  n','AI  median [IQR]  n','AUC');
for k = 1:numel(A.FT.names)
    vr = A.X(r & isfinite(A.X(:,k)),k); va = A.X(a & isfinite(A.X(:,k)),k);
    au = '-';
    if numel(vr) >= 5 && numel(va) >= 5
        x = auc_rank(va,vr); au = sprintf('%.3f',max(x,1-x));
    end
    fprintf('  %-12s  %-30s  %-30s  %s\n',A.FT.names{k},stat_str(vr),stat_str(va),au);
end
end

function print_calibration(A)
C = A.Cal; FT = A.FT; nG = numel(FT.gnames);
y = C.y; s = C.oofS;
L1 = repmat('=',1,112); L2 = repmat('-',1,112);
fprintf('\n%s\n',L1);
fprintf(' CALIBRATION %s  |  REAL %d  AI %d  |  %d-fold stratified CV  |  crop %d px, JPEG re-save q%d\n', ...
    C.date,C.n(1),C.n(2),C.O.kfold,C.O.cropSide,C.O.rejpeg);
fprintf(' all accuracies below are OUT-OF-FOLD (each image scored by a model that never saw it)\n');
fprintf('%s\n',L1);

fprintf(' 1) SINGLE FEATURES   dir: side of the threshold that means AI | weight: share in the all-feature model\n');
fprintf(' %-12s %-18s %-27s %-27s %6s %3s %10s %7s %7s %5s\n', ...
    'feature','test','REAL median [IQR]','AI median [IQR]','AUC','dir','threshold','balAcc','weight','NaN%');
fprintf('%s\n',L2);
for g = 1:nG
    for k = find(FT.grp==g)
        fprintf(' %-12s %-18s %-27s %-27s %6.3f %3s %10.4g %6.1f%% %6.1f%% %5.1f\n', ...
            FT.names{k},FT.gnames{g},qstr(C.fStatR(:,k)),qstr(C.fStatA(:,k)), ...
            max(C.fAUC(k),1-C.fAUC(k)),dsym(C.fDir(k)),C.fThr(k),100*C.fBacc(k),C.featW(k),C.fNaN(k));
    end
end

fprintf('\n 2) TESTS   (ridge-logistic per test; score > threshold = AI)\n');
fprintf(' %-20s %6s %8s %11s %8s %9s %9s\n','test','#feat','CV AUC','threshold','balAcc','lambda','WEIGHT');
fprintf('%s\n',L2);
for g = 1:nG
    fprintf(' %-20s %6d %8.3f %11.3f %7.1f%% %9.2g %8.1f%%\n',FT.gnames{g},numel(C.grp(g).cols), ...
        C.grp(g).auc,C.grp(g).thr,100*C.grp(g).bacc,C.grp(g).lam,C.testW(g));
end

fprintf('\n 3) FUSION\n');
fprintf('    all-feature ridge model  : CV AUC %.3f   balAcc %.1f%%   (lambda %.2g)\n',C.full.auc,100*C.full.bacc,C.full.lam);
fprintf('    stacked (tests -> meta)  : CV AUC %.3f   balAcc %.1f%%   (lambda %.2g)\n',C.meta.auc,100*C.meta.bacc,C.meta.lam);
fprintf('    -> USING: %s\n',upper(C.kind));
fprintf('    decision threshold       : %.4f   (P(AI) there = %.2f)\n',C.thr,1/(1+exp(-(C.platt(1)*C.thr+C.platt(2)))));
fprintf('    uncertain band           : +/- %.4f  -> %.1f%% of images get a verdict, balanced accuracy %.1f%%\n', ...
    C.band,100*C.cov,100*C.cacc);
pr = s >= C.thr;
TN = nnz(~pr & y==0); FP = nnz(pr & y==0); FN = nnz(~pr & y==1); TP = nnz(pr & y==1);
fprintf('\n    confusion at threshold (no band):   called REAL   called AI\n');
fprintf('                     REAL images   %10d  %10d\n',TN,FP);
fprintf('                     AI images     %10d  %10d\n',FN,TP);
fprintf('    REAL correct %.1f%%  |  AI correct %.1f%%  |  accuracy %.1f%%  |  BALANCED %.1f%%\n', ...
    100*TN/max(TN+FP,1),100*TP/max(TP+FN,1),100*(TN+TP)/numel(y),100*C.bacc);

fprintf('\n 4) PER SOURCE FOLDER (out-of-fold, at threshold)\n');
src = C.trainSrc; u = unique(src,'stable');
for i = 1:numel(u)
    m = strcmp(src,u{i});
    ym = y(m); cor = mean((s(m) >= C.thr) == (ym==1));
    fprintf('    %-5s %6d images  %5.1f%% correct   %s\n',tern(ym(1)==1,'AI','REAL'),nnz(m),100*cor,u{i});
end

fprintf('\n 5) CONFOUND CHECK\n');
[tf,loc] = ismember(lower(C.trainPaths),lower(A.path));
if all(tf)
    J = A.isJ(loc); NP = A.npix(loc);
    fprintf('    originally JPEG : REAL %.0f%%   AI %.0f%%\n',100*mean(J(y==0)),100*mean(J(y==1)));
    fprintf('    median size     : REAL %.2f MP   AI %.2f MP\n',median(NP(y==0))/1e6,median(NP(y==1))/1e6);
    if abs(mean(J(y==0)) - mean(J(y==1))) > 0.5
        fprintf('    [!] file formats differ strongly between classes; the JPEG re-save reduces this, but\n');
        fprintf('        try to add AI images saved as JPEG / real images saved as PNG.\n');
    end
end
if numel(unique(src(y==1))) < 2
    fprintf('    [i] only one AI source folder: accuracy on other generators may be lower.\n');
end
if C.bacc < 0.6
    fprintf('    [!] balanced accuracy under 60%%: the tests barely separate these folders.\n');
end
fprintf('%s\n',L1);
end

function print_single(A,row,num)
FT = A.FT; nF = numel(FT.names); nG = numel(FT.gnames);
x = A.X(row,:); d = A.D(row,:);
r = A.cls==1; a = A.cls==2; C = A.Cal; hasC = ~isempty(C);
L1 = repmat('=',1,100); L2 = repmat('-',1,100);
fprintf('\n%s\n [%d] %s\n',L1,num,A.path{row});
fprintf('     %.2f MP, originally JPEG: %s, analysed as %d px centre crop re-saved JPEG q%d\n', ...
    A.npix(row)/1e6,tern(A.isJ(row),'yes','no'),A.O.cropSide,A.O.rejpeg);
if ~hasC
    fprintf('     NOT CALIBRATED - raw values and ranks vs the loaded folders only\n');
elseif A.stale
    fprintf('     note: data was added after the last calibration - press Calibrate to use it\n');
end
fprintf('%s\n',L2);
fprintf(' %-30s %10s %8s %8s %12s   %s\n','test / feature','value','>%real','>%AI','AI if','result');
nAI = 0;
for g = 1:nG
    if hasC
        sg = d(nF+g); tg = C.grp(g).thr; isAI = sg >= tg; nAI = nAI + isAI;
        fprintf(' TEST %d %-23s %10.3f %8s %8s %12s   %-4s  P(AI) %.2f | CV AUC %.3f | weight %4.1f%%\n', ...
            g,FT.gnames{g},sg,rp(A.D(r,nF+g),sg),rp(A.D(a,nF+g),sg),sprintf('>= %.3f',tg), ...
            tern(isAI,'AI','REAL'),1/(1+exp(-sg)),C.grp(g).auc,C.testW(g));
    else
        fprintf(' TEST %d %s\n',g,FT.gnames{g});
    end
    for k = find(FT.grp==g)
        ts = '-'; vt = '';
        if ~isfinite(x(k))
            vt = 'n/a';
        elseif hasC && isfinite(C.fThr(k))
            ts = sprintf('%s %.4g',dsym(C.fDir(k)),C.fThr(k));
            vt = tern(C.fDir(k)*x(k) >= C.fDir(k)*C.fThr(k),'ai','real');
        end
        fprintf('     %-26s %10.4g %8s %8s %12s   %s\n',FT.names{k},x(k),rp(A.X(r,k),x(k)),rp(A.X(a,k),x(k)),ts,vt);
    end
end
if hasC
    s = d(end); [v,p] = verdict_of(C,s);
    fprintf('%s\n',L2);
    fprintf(' FINAL (%s fusion): score %.3f   threshold %.3f   uncertain band +/- %.3f\n',C.kind,s,C.thr,C.band);
    fprintf(' tests voting AI: %d of %d    P(AI) = %.3f\n',nAI,nG,p);
    fprintf(' >>> VERDICT: %s\n',v);
    if strcmp(v,'UNCERTAIN')
        fprintf('     (score is inside the uncertain band around the threshold: the tests disagree or the\n');
        fprintf('      image looks like neither class strongly - leaning %s, P(AI) %.2f)\n',tern(s>=C.thr,'AI','REAL'),p);
    end
else
    fprintf('%s\n >>> VERDICT: none - calibrate (or load a calibration) first\n',L2);
end
fprintf('%s\n',L1);
end

function s = rp(v,x)
v = v(isfinite(v));
if isempty(v) || ~isfinite(x), s = '-'; else, s = sprintf('%.1f',100*mean(v < x)); end
end

function s = qstr(q)
if any(~isfinite(q)), s = '-'; else, s = sprintf('%.4g [%.4g %.4g]',q(1),q(2),q(3)); end
end

function q = q3(v)
if isempty(v), q = nan(3,1); else, q = [median(v); pct(v,0.25); pct(v,0.75)]; end
end

function s = stat_str(v)
if isempty(v), s = '-';
else, s = sprintf('%.4g [%.4g %.4g]  %d',median(v),pct(v,0.25),pct(v,0.75),numel(v)); end
end

function s = dsym(d)
if d > 0, s = '>'; else, s = '<'; end
end

function s = tern(c,a,b)
if c, s = a; else, s = b; end
end

%% =====================================================================================
%  DISPLAY
%  =====================================================================================
function set_status(A,str)
if ishandle(A.hs), set(A.hs,'String',str); drawnow; end
end

function update_status(A)
t = {sprintf('crop side %d px',A.O.cropSide),sprintf('JPEG re-save q%d',A.O.rejpeg), ...
     sprintf('workers: %s',workers_str()),'', ...
     sprintf('real images: %d',nnz(A.cls==1)),sprintf('AI images: %d',nnz(A.cls==2)), ...
     sprintf('single images: %d',nnz(A.cls==3)), ...
     sprintf('test images: %d real, %d AI',nnz(A.T.cls==1),nnz(A.T.cls==2)),''};
if isempty(A.Cal)
    t{end+1} = 'NOT CALIBRATED';
else
    C = A.Cal;
    t = [t, {sprintf('calibrated %s',C.date),sprintf('on %d real + %d AI',C.n(1),C.n(2)), ...
             sprintf('fusion: %s',C.kind),sprintf('CV AUC %.3f',C.aucChosen), ...
             sprintf('CV bal. acc %.1f%%',100*C.bacc),sprintf('thr %.3f  band %.3f',C.thr,C.band)}];
    if A.stale, t{end+1} = '>> new data: press Calibrate'; end
end
set(A.hs,'String',t);
end

function [t,b] = plot_thr(A,col)
t = NaN; b = 0; C = A.Cal;
if isempty(C), return; end
nF = numel(A.FT.names); nG = numel(A.FT.gnames);
if col <= nF,           t = C.fThr(col);
elseif col <= nF + nG,  t = C.grp(col-nF).thr;
else,                   t = C.thr; b = C.band;
end
end

function redraw(fig)
if ~ishandle(fig), return; end
A = getappdata(fig,'A');
r = A.cls==1; a = A.cls==2; s = find(A.cls==3);
update_status(A);
cR = [0 0.447 0.741]; cA = [0.850 0.325 0.098];
nF = numel(A.FT.names);

for k = 1:4
    ax = A.ax{k};
    cla(ax,'reset'); hold(ax,'on');
    j = get(A.hp(k),'Value'); col = A.plotCol(j); nm = A.plotNames{j};
    v = A.D(:,col);
    [x1,y1] = strip_xy(1,v(r)); [x2,y2] = strip_xy(2,v(a));
    plot(ax,x1,y1,'.','Color',cR,'MarkerSize',7);
    plot(ax,x2,y2,'.','Color',cA,'MarkerSize',7);
    vs = v(s);
    if ~isempty(s), plot(ax,3*ones(numel(s),1),vs,'k*'); end
    for i = 1:numel(s)
        if isfinite(vs(i)), text(ax,3.12,vs(i),sprintf('%d',i),'FontSize',8); end
    end
    [th,bd] = plot_thr(A,col);
    if isfinite(th)
        plot(ax,[0.5 3.5],[th th],'k--','LineWidth',1.1);
        if bd > 0, plot(ax,[0.5 3.5],th+[bd bd],'k:',[0.5 3.5],th-[bd bd],'k:'); end
    end
    hold(ax,'off');
    xlim(ax,[0.5 3.5]);
    set(ax,'XTick',1:3,'XTickLabel',{'real','AI','images'});
    lim = value_lim(v(r|a),[vs; th; th+bd; th-bd]);
    if ~isempty(lim), ylim(ax,lim); end
    sub = '';
    vr = v(r & isfinite(v)); va = v(a & isfinite(v));
    if numel(vr) >= 5 && numel(va) >= 5
        au = auc_rank(va,vr);
        sub = sprintf('AUC %.3f',max(au,1-au));
        if col > nF, sub = [sub ' (out-of-fold)']; end
    end
    if isfinite(th), sub = sprintf('%s  thr %.4g',sub,th); end
    title(ax,{strrep(nm,'_','\_'),sub},'FontSize',9);
    grid(ax,'on'); box(ax,'on');
end

% ---- region map -----------------------------------------------------------
ax = A.axr;
cla(ax,'reset');
i = A.plotCol(get(A.hx,'Value')); j = A.plotCol(get(A.hy,'Value'));
nx = strrep(A.colNames{i},'_','\_'); ny = strrep(A.colNames{j},'_','\_');
if i == j
    title(ax,'choose two different quantities for X and Y'); return;
end
gr = r & isfinite(A.D(:,i)) & isfinite(A.D(:,j));
ga = a & isfinite(A.D(:,i)) & isfinite(A.D(:,j));
xr = A.D(gr,i); yr = A.D(gr,j); xa = A.D(ga,i); ya = A.D(ga,j);
xs = A.D(s,i);  ys = A.D(s,j);
lx = value_lim([xr;xa],xs); ly = value_lim([yr;ya],ys);
hold(ax,'on');
hh = []; lab = {};
ttl = {sprintf('%s vs %s',nx,ny)};
if numel(xr) >= 10 && numel(xa) >= 10
    nb = 120;
    xc = lx(1) + diff(lx)/nb*((1:nb)-0.5);
    yc = ly(1) + diff(ly)/nb*((1:nb)-0.5);
    Pr = dens2(xr,yr,lx,ly,nb); Pa = dens2(xa,ya,lx,ly,nb);
    occ = Pr >= hdr_level(Pr,0.95) | Pa >= hdr_level(Pa,0.95);
    post = Pr./(Pr + Pa + realmin);
    Rg = zeros(nb);
    Rg(occ & post >= A.O.purity) = 1;
    Rg(occ & post <= 1-A.O.purity) = 2;
    Rg(occ & post > 1-A.O.purity & post < A.O.purity) = 3;
    fr = region_share(xr,yr,Rg,lx,ly,nb); fa = region_share(xa,ya,Rg,lx,ly,nb);
    cmap = [1 1 1; 0.80 0.88 1; 1 0.83 0.78; 0.86 0.86 0.86];
    imagesc(ax,xc,yc,Rg);
    colormap(ax,cmap); set(ax,'CLim',[-0.5 3.5]);
    hh = [patch(ax,NaN,NaN,cmap(2,:)),patch(ax,NaN,NaN,cmap(3,:)),patch(ax,NaN,NaN,cmap(4,:))];
    lab = {'real region','AI region','overlap'};
    ttl{2} = sprintf('blue: real only | red: AI only | grey: overlap (%.0f%% purity)',100*A.O.purity);
    ttl{3} = sprintf('real %.0f%% own region / %.0f%% overlap | AI %.0f%% own region / %.0f%% overlap',fr(1),fr(2),fa(3),fa(2));
end
h1 = plot(ax,nan_if_empty(xr),nan_if_empty(yr),'.','Color',cR,'MarkerSize',7);
h2 = plot(ax,nan_if_empty(xa),nan_if_empty(ya),'.','Color',cA,'MarkerSize',7);
hh = [hh,h1,h2];
lab = [lab,{sprintf('real (%d)',numel(xr)),sprintf('AI (%d)',numel(xa))}];
g = find(isfinite(xs) & isfinite(ys));
if ~isempty(g)
    hi = plot(ax,xs(g),ys(g),'k*','MarkerSize',8);
    for t = g(:)', text(ax,xs(t),ys(t),sprintf('  %d',t),'FontSize',9); end
    hh = [hh,hi]; lab{end+1} = 'single images';
end
hold(ax,'off');
set(ax,'YDir','normal');
if ~isempty(lx), xlim(ax,lx); end
if ~isempty(ly), ylim(ax,ly); end
xlabel(ax,nx); ylabel(ax,ny); title(ax,ttl); box(ax,'on');
if ~isempty(A.cls), legend(ax,hh,lab,'Location','best'); end
end

function [x,y] = strip_xy(c,v)
if isempty(v), x = c; y = NaN; return; end
n = numel(v);
x = c + 0.2*(2*mod((1:n)'*0.6180339887,1) - 1);    % low-discrepancy jitter
y = v;
end

function v = nan_if_empty(v)
if isempty(v), v = NaN; end
end

function l = value_lim(v,extra)
v = v(isfinite(v)); e = extra(isfinite(extra));
if isempty(v) && isempty(e), l = []; return; end
if isempty(v)
    l = [min(e) max(e)];
else
    l = [pct(v,0.005) pct(v,0.995)];
    if ~isempty(e), l = [min(l(1),min(e)) max(l(2),max(e))]; end
end
if diff(l) <= 0, l = l + [-1 1]*max(abs(l(1)),1)*0.01; end
l = l + [-0.05 0.05]*diff(l);
end

function P = dens2(x,y,lx,ly,nb)
in = x >= lx(1) & x <= lx(2) & y >= ly(1) & y <= ly(2);
ix = min(nb,floor((x(in)-lx(1))/diff(lx)*nb)+1);
iy = min(nb,floor((y(in)-ly(1))/diff(ly)*nb)+1);
H = accumarray([iy(:) ix(:)],1,[nb nb]);
g = exp(-((-9:9).^2)/18);
P = conv2(g,g,H,'same')/max(numel(x),1);
end

function t = hdr_level(P,m)
s = sort(P(:),'descend'); c = cumsum(s);
if c(end) <= 0, t = Inf; return; end
t = s(find(c >= m*c(end),1));
end

function f = region_share(x,y,Rg,lx,ly,nb)
ix = min(nb,max(1,floor((x-lx(1))/diff(lx)*nb)+1));
iy = min(nb,max(1,floor((y-ly(1))/diff(ly)*nb)+1));
r = Rg(sub2ind([nb nb],iy,ix));
f = 100*[mean(r==1),mean(r==3),mean(r==2),mean(r==0)];
end

%% =====================================================================================
%  GENERAL HELPERS
%  =====================================================================================
function y = pct(x,p)
% percentile(s), p as fraction(s) 0..1, NaN ignored
x = sort(x(isfinite(x(:))));
if isempty(x), y = nan(size(p)); return; end
y = reshape(x(max(1,min(numel(x),round(p*numel(x))))),size(p));
end

function A = auc_rank(a,b)
% P(a > b), ties count half
a = a(isfinite(a)); b = b(isfinite(b));
na = numel(a); nb = numel(b);
if na == 0 || nb == 0, A = NaN; return; end
x = [a(:); b(:)];
[~,i] = sort(x);
rk = zeros(size(x)); rk(i) = 1:numel(x);
[~,~,g] = unique(x);
rk = accumarray(g,rk,[],@mean); rk = rk(g);
A = (sum(rk(1:na)) - na*(na+1)/2)/(na*nb);
end

function t = same_opts(a,b)
fl = {'cropSide','rejpeg','minSide','blk','minBlocksBin'};
t = true;
for i = 1:numel(fl)
    if ~isfield(a,fl{i}) || ~isfield(b,fl{i}) || ~isequal(a.(fl{i}),b.(fl{i})), t = false; return; end
end
end

function files = list_images(d,exts)
L = dir(fullfile(d,'**','*'));
L = L(~[L.isdir]);
files = {};
if isempty(L), return; end
names = {L.name}; folders = {L.folder};
[~,~,e] = cellfun(@fileparts,names,'UniformOutput',false);
keep = ismember(lower(e),exts);
if ~any(keep), return; end
files = fullfile(folders(keep),names(keep));
files = files(:);
end

function [I,isJ,np] = load_crop(f,S)
isJ = is_jpeg(f);
[I,map] = imread(f);
[h,w,~] = size(I);
np = h*w;
ch = min(h,S); cw = min(w,S);
r0 = 16*floor((h-ch)/32); c0 = 16*floor((w-cw)/32);
I = I(r0+1:r0+ch,c0+1:c0+cw,:);
if ~isempty(map)
    I = round(255*ind2rgb(I,map));
elseif isa(I,'uint16')
    I = double(I)*255/65535;
elseif islogical(I) || isfloat(I)
    I = double(I)*255;
else
    I = double(I);
end
if size(I,3) > 3
    I = I(:,:,1:3);
elseif size(I,3) == 2
    I = I(:,:,1);
end
end

function I = resave_jpeg(I,q)
if q > 0
    tf = [tempname '.jpg'];
    imwrite(uint8(I),tf,'Quality',q);
    I = double(imread(tf));
    delete(tf);
end
end

function t = is_jpeg(f)
t = false;
fid = fopen(f,'r');
if fid < 0, return; end
b = fread(fid,2,'uint8');
fclose(fid);
t = numel(b) == 2 && b(1) == 255 && b(2) == 216;
end
