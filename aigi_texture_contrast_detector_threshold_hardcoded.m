function aigi_texture_contrast_detector
% AIGI_TEXTURE_CONTRAST_DETECTOR  (PURE DSP, SINGLE-TRACE TEXTURE CONTRAST)
%
% Decides whether the rich-texture and poor-texture regions of one image
% carry the SAME high-frequency energy, or whether they diverge the way a
% generator's output tends to.
%
% Method after: Zhong, Xu, Li, Qian & Zhang, "PatchCraft: Exploring Texture
% Patch for Efficient AI-generated Image Detection", CVPR 2024,
% arXiv:2311.12397.  This file keeps the Smash&Reconstruction step exactly
% as in the paper's DSP analogue, but reduces the comparator to ONE trace:
% SRM high-frequency ENERGY CONTRAST between the richest and poorest third
% of sampled patches.  No correlation trace, no spectral trace, no
% kurtosis trace, no machine learning -- one number, one cutoff.
%
% PREMISE
%   A real photograph's rich and poor regions passed through one lens, one
%   sensor and one ISP, so their fine-scale energy tracks together as
%   texture complexity changes.  A generator solves two differently hard
%   problems -- high-entropy texture is much harder to synthesise than
%   low-entropy flat content -- so its energy CONTRAST between the two
%   region types tends to diverge from what a single real pipeline produces.
%
% THE ONE TRACE
%   TEXTURE CONTRAST  = (er - ep) / (er + ep), where er/ep are the mean
%   SRM-filtered high-frequency energies of the richest/poorest thirds of
%   192 sampled 32x32 patches.  Reported SCORE = -contrast (matches the
%   convention F.tex_contrast uses in aigi_features).
%
% HOW THE DECISION IS MADE
%   The trace has a cutoff AND A DIRECTION, both derived in mode 4 by
%   sweeping your own folders.  Direction matters: on real data the trace
%   can come out inverted, and forcing the assumed direction turns a
%   useful trace into a source of false positives.
%   An image is SYNTHETIC when the score fires past the cutoff in the
%   learned direction.

clc; close all;
if ~license('test','image_toolbox')
    error('This code requires the Image Processing Toolbox.');
end

fprintf('=================================================================\n');
fprintf('   TEXTURE CONTRAST CONSISTENCY  (PURE DSP, SINGLE-TRACE)\n');
fprintf('=================================================================\n');
print_rule(decision_thresholds());

while true
    modeChoice = menu('Select Analysis Mode', ...
                      'Analyze Single Image (plot poor/rich reconstruction)', ...
                      'Compare Single Pair (1 Real vs 1 AI)', ...
                      'Batch Folder Analysis (Real Folder vs AI Folder)', ...
                      'Calibrate Threshold From Folders  (run this first)', ...
                      'Show Current Decision Rule', ...
                      'Exit');
    try
        switch modeChoice
            case 1, run_single_image();
            case 2, run_single_pair();
            case 3, run_batch_folders();
            case 4, run_tune_threshold();
            case 5, print_rule(decision_thresholds());
            otherwise
                disp('Exiting.'); return;
        end
    catch ME
        fprintf('\n [!] %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('     (in %s, line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('     Returning to the menu.\n\n');
    end
end
end

% =========================================================================
% THE DECISION RULE
%   thr    cutoff on SCORE = -contrast
%   dir    +1 means SYNTHETIC when score is ABOVE the cutoff
%          -1 means SYNTHETIC when score is BELOW  (an inverted trace)
% =========================================================================
function T = decision_thresholds()
T.name   = 'texture_contrast';
T.thr    = -0.550;
T.dir    = -1;
T.active = true;
T.source = 'built-in seed value -- run mode 4 before reporting anything';
end

function print_rule(T)
fprintf('\n--------------------- DECISION RULE ---------------------\n');
fprintf(' source : %s\n', T.source);
fprintf(' ---------------------------------------------------------\n');
if ~T.active
    fprintf(' %-16s  OFF  (too close to chance to be trusted)\n', T.name);
else
    fprintf(' %-16s  fires when score %s %10.5f\n', T.name, dirsym(T.dir), T.thr);
end
fprintf(' ---------------------------------------------------------\n');
fprintf(' SYNTHETIC when the trace fires\n');
if isfield(T,'perf')
    fprintf(' ---------------------------------------------------------\n');
    fprintf(' On the folders it was tuned on:\n');
    fprintf('   balanced accuracy : %.1f%%\n', 100*T.perf.bacc);
    fprintf('   AUC               : %.3f\n', T.perf.auc);
    if abs(T.perf.auc - 0.5) < 0.05
        fprintf('   [i] AUC is within 0.05 of chance on this data.\n');
    end
end
fprintf('==========================================================\n\n');
end

function s = dirsym(d)
if d > 0, s = '>'; else, s = '<'; end
end

% =========================================================================
% FIRING / VERDICT
% =========================================================================
function tf = fires(score, T)
if ~isfinite(score), tf = false; return; end
if ~T.active, tf = false; return; end
if T.dir > 0
    tf = score > T.thr;
else
    tf = score < T.thr;
end
end

function tf = is_synthetic(F, T)
tf = fires(F.score, T);
end

function s = yn(b)
if b, s = 'FIRES'; else, s = '-'; end
end

% =========================================================================
% MODE 1: SINGLE IMAGE  (same plot/output as visualize_texture_contrast.m)
% =========================================================================
function run_single_image()
[imgFile, imgPath] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'}, 'Select an image');
if isequal(imgFile,0), disp('Cancelled.'); return; end

T = decision_thresholds();
fprintf('\nProcessing image: %s...\n', imgFile);
F = compute_texture_features(im2double(imread(fullfile(imgPath, imgFile))));
if ~F.ok
    fprintf('\n[!] Image is too small for 32x32 patch sampling.\n\n');
    return;
end

grid_size = ceil(sqrt(floor(192/3)));

fprintf('\n--- TEXTURE CONTRAST RESULTS ---\n');
fprintf('Patch Size (ps)  : 32 x 32 pixels\n');
fprintf('Total Patches    : 192\n');
fprintf('Patches per set  : %d (arranged in %dx%d grid)\n', floor(192/3), grid_size, grid_size);
fprintf('--------------------------------\n');
fprintf('Poor Energy (ep) : %f\n', F.ep);
fprintf('Rich Energy (er) : %f\n', F.er);
fprintf('Contrast Ratio   : %f\n', F.contrast);
fprintf('Final Score      : %f\n', F.score);
fprintf('--------------------------------\n');

fires_ = is_synthetic(F, T);
fprintf('Trace fires?     : %s\n', yn(fires_));
if fires_
    fprintf('Assessment       : SPLIT SYNTHESIS SIGNATURE.\n');
else
    fprintf('Assessment       : CONSISTENT PIPELINE SIGNATURE.\n');
end
fprintf('================================\n\n');

figure('Name', 'PatchCraft: Smash & Reconstruction', 'Position', [100, 100, 1000, 350]);
subplot(1,3,1);
imshow(F.g);
title(sprintf('Original Image (Green Channel)\n%s', imgFile), 'Interpreter','none');

subplot(1,3,2);
imshow(F.poor);
title(sprintf('Poor Texture Reconstruction\n(ep: %.4f)', F.ep));

subplot(1,3,3);
imshow(F.rich);
title(sprintf('Rich Texture Reconstruction\n(er: %.4f)', F.er));

sgtitle(sprintf('Score %.4f  |  %s', F.score, tern(fires_,'SYNTHETIC','AUTHENTIC-LOOKING')));
end

function s = tern(c,a,b)
if c, s = a; else, s = b; end
end

% =========================================================================
% MODE 2: SINGLE PAIR
% =========================================================================
function run_single_pair()
[realFile, realPath] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'}, 'Select ONE REAL image');
if isequal(realFile,0), disp('Cancelled.'); return; end
[aiFile, aiPath] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'}, 'Select ONE AI image');
if isequal(aiFile,0), disp('Cancelled.'); return; end

T = decision_thresholds();

fprintf('\nProcessing REAL image: %s...\n', realFile);
R = compute_texture_features(im2double(imread(fullfile(realPath, realFile))));
fprintf('Processing AI image:   %s...\n', aiFile);
A = compute_texture_features(im2double(imread(fullfile(aiPath, aiFile))));

if ~R.ok || ~A.ok
    fprintf('\n[!] One of the images is too small for 32x32 patch sampling.\n\n');
    return;
end

fprintf('\n----------------------- FORENSIC METRICS -----------------------\n');
fprintf('%-26s | %-14s | %-14s\n', 'Metric', 'REAL Image', 'AI Image');
fprintf('----------------------------------------------------------------\n');
fprintf('%-26s | %-14.5f | %-14.5f\n', '1. Poor Energy (ep)',  R.ep,       A.ep);
fprintf('%-26s | %-14.5f | %-14.5f\n', '2. Rich Energy (er)',  R.er,       A.er);
fprintf('%-26s | %-14.5f | %-14.5f\n', '3. Contrast Ratio',    R.contrast, A.contrast);
fprintf('%-26s | %-14.5f | %-14.5f\n', '4. FINAL SCORE',       R.score,    A.score);
fprintf('----------------------------------------------------------------\n');

rFires = is_synthetic(R, T);
aFires = is_synthetic(A, T);
fprintf('%-26s | %-14s | %-14s\n', 'Trace fires?', yn(rFires), yn(aFires));
fprintf('----------------------------------------------------------------\n');

fprintf('\nForensic Assessment:\n');
fprintf('REAL Image: ');
if rFires
    fprintf('SUSPICIOUS. Rich/poor energy contrast crosses the cutoff.\n');
else
    fprintf('AUTHENTIC. Rich and poor regions share one pipeline signature.\n');
end
fprintf('AI Image  : ');
if aFires
    fprintf('SYNTHETIC SIGNATURE. Rich/poor energy contrast shows split synthesis.\n');
else
    fprintf('INCONCLUSIVE. Regions remain consistent (heavy post-processing?).\n');
end
fprintf('================================================================\n\n');

figure('Name','Smash & Reconstruction: Rich vs Poor Texture','Position',[80 220 1100 560]);
subplot(2,3,1); imshow(R.g); title(sprintf('REAL: green channel\n%s', realFile),'Interpreter','none');
subplot(2,3,2); imshow(R.poor); title(sprintf('REAL: POOR (ep %.4f)', R.ep));
subplot(2,3,3); imshow(R.rich); title(sprintf('REAL: RICH (er %.4f)  score %.4f', R.er, R.score));
subplot(2,3,4); imshow(A.g); title(sprintf('AI: green channel\n%s', aiFile),'Interpreter','none');
subplot(2,3,5); imshow(A.poor); title(sprintf('AI: POOR (ep %.4f)', A.ep));
subplot(2,3,6); imshow(A.rich); title(sprintf('AI: RICH (er %.4f)  score %.4f', A.er, A.score));
end

% =========================================================================
% MODE 3: BATCH
% =========================================================================
function run_batch_folders()
realDirs = pick_folders('REAL');
if isempty(realDirs), disp('Cancelled.'); return; end
aiDirs = pick_folders('AI', realDirs{end});
if isempty(aiDirs), disp('Cancelled.'); return; end
if any(ismember(lower(realDirs), lower(aiDirs)))
    fprintf('\n [!] A folder appears in both lists.  Nothing to compare.\n\n'); return;
end

T = decision_thresholds();

fprintf('\nAnalyzing %d REAL folder(s)...\n', numel(realDirs));
[resReal, srcR, nameR] = collect_folders(realDirs, 'REAL', T);
fprintf('\nAnalyzing %d AI folder(s)...\n', numel(aiDirs));
[resAI, srcA, nameA] = collect_folders(aiDirs, 'AI', T);

fprintf('\n=================================================================\n');
fprintf('                 BATCH TEXTURE CONTRAST REPORT                    \n');
fprintf('=================================================================\n');
fprintf(' Rule source: %s\n', T.source);
fprintf(' Pooled: %d real image(s) from %d folder(s), %d AI from %d folder(s)\n\n', ...
        numel(resReal), numel(realDirs), numel(resAI), numel(aiDirs));
print_batch_stats('REAL Folders', resReal, T);
print_per_source('REAL', resReal, srcR, nameR, T);
fprintf('\n');
print_batch_stats('AI Folders  ', resAI, T);
print_per_source('AI', resAI, srcA, nameA, T);
fprintf('\n');
print_accuracy(resReal, resAI, T);
check_resolution(resReal, resAI);
fprintf('=================================================================\n');

draw_batch_figures(resReal, resAI, T);
end

function draw_batch_figures(resReal, resAI, T)
sR = [resReal.score]; sA = [resAI.score];
sR = sR(isfinite(sR)); sA = sA(isfinite(sA));

figure('Name','Texture Contrast: REAL vs AI','Position',[130 120 900 460]);
if isempty(sR) || isempty(sA)
    axis off; title('Not enough valid data to plot.');
    return;
end
hi = max([quantile_([sR(:);sA(:)],0.99), T.thr*1.5]);
lo = min([0, quantile_([sR(:);sA(:)],0.01)]);
if T.active
    % guarantee the cutoff is never clipped outside the plotted range
    pad = 0.05*max(hi-lo, eps);
    hi = max(hi, T.thr + pad);
    lo = min(lo, T.thr - pad);
end
ed = linspace(lo, hi, 30);
histogram(sR, ed, 'FaceColor','b','FaceAlpha',0.55,'EdgeColor','none','DisplayName','REAL'); hold on;
histogram(sA, ed, 'FaceColor','r','FaceAlpha',0.55,'EdgeColor','none','DisplayName','AI');
if T.active
    xline(T.thr, '--k', 'LineWidth', 2.2, 'DisplayName', 'cutoff');
end
a = auc_(sA, sR);
xlabel('Score = -contrast'); ylabel('images');
if T.active
    ttl = sprintf('Texture Contrast Score\nAUC %.3f   fires when %s cutoff', a, dirsym(T.dir));
else
    ttl = sprintf('Texture Contrast Score\nAUC %.3f   TRACE OFF (at chance)', a);
end
title(ttl);
legend('Location','best'); grid on;

[tp,tn,fp,fn] = compute_confusion(resReal, resAI);
realAcc = tn/max(tn+fp,1);
aiAcc   = tp/max(tp+fn,1);
if isfinite(realAcc)
    ax = gca;
    txt = sprintf(['REAL accuracy : %5.1f%%\n' ...
                    'AI accuracy   : %5.1f%%'], ...
                    100*realAcc, 100*aiAcc);
    text(ax, 0.02, 0.98, txt, 'Units','normalized', ...
        'VerticalAlignment','top', 'FontName','FixedWidth', 'FontSize',9, ...
        'BackgroundColor',[1 1 1 0.85], 'EdgeColor',[0.3 0.3 0.3]);
end
hold off;

% Figure 2: contrast score vs image index, per class
%   x = position of the image within its own class list (1..N), y = its
%   score. This shows how much the ratio jitters/drifts image-to-image
%   within REAL and within AI, side by side, rather than collapsing
%   everything into one histogram shape.
nR = numel(sR); nA = numel(sA);

figure('Name','Per-Image Texture Contrast: Real vs AI','Position',[130 90 1050 560]);
scatter(1:nR, sR, 22, 'b', 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName','REAL'); hold on;
scatter(1:nA, sA, 22, 'r', 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName','AI');
if T.active
    yline(T.thr, '--k', 'LineWidth', 1.6, 'DisplayName', 'cutoff');
end
xlabel('Image number (index within its own class)');
ylabel('Score = -contrast (energy ratio)');
title('Per-Image Texture Contrast: Real vs AI');
legend('Location','best'); grid on; hold off;

if isfinite(realAcc)
    ax = gca;
    txt = sprintf(['REAL accuracy : %5.1f%%\n' ...
                    'AI accuracy   : %5.1f%%'], ...
                    100*realAcc, 100*aiAcc);
    text(ax, 0.02, 0.98, txt, 'Units','normalized', ...
        'VerticalAlignment','top', 'FontName','FixedWidth', 'FontSize',9, ...
        'BackgroundColor',[1 1 1 0.85], 'EdgeColor',[0.3 0.3 0.3]);
end
end

% =========================================================================
% MODE 4: CALIBRATE  -- this is where the decision rule is actually made
% =========================================================================
function run_tune_threshold()
fprintf('\n---------------------------------------------------------------\n');
fprintf(' CALIBRATION\n');
fprintf(' The code sweeps every candidate cutoff in BOTH directions and\n');
fprintf(' keeps the best.  Direction is learned, not assumed: the trace\n');
fprintf(' can come out inverted, and forcing the assumed direction turns\n');
fprintf(' it into a source of false positives.\n');
fprintf('---------------------------------------------------------------\n');

realDirs = pick_folders('REAL');
if isempty(realDirs), disp('Cancelled.'); return; end
aiDirs = pick_folders('AI', realDirs{end});
if isempty(aiDirs), disp('Cancelled.'); return; end
if any(ismember(lower(realDirs), lower(aiDirs)))
    fprintf('\n [!] A folder appears in both lists.  Nothing to tune against.\n\n'); return;
end

nR = 0; for i=1:numel(realDirs), nR = nR + numel(list_images(realDirs{i})); end
nA = 0; for i=1:numel(aiDirs),   nA = nA + numel(list_images(aiDirs{i}));   end
if nR < 5 || nA < 5
    fprintf('\n [!] Need at least 5 images per class (got %d real, %d AI).\n\n', nR, nA);
    return;
end
if numel(aiDirs) == 1
    fprintf(['\n [i] Only one AI source.  A rule tuned on a single generator\n' ...
             '     usually collapses on the next one.  Pool several if you can.\n']);
end

T0 = decision_thresholds();
fprintf('\nMeasuring %d REAL folder(s) (%d images)...\n', numel(realDirs), nR);
[resR, srcR, nameR] = collect_folders(realDirs, 'REAL', T0);
fprintf('\nMeasuring %d AI folder(s) (%d images)...\n', numel(aiDirs), nA);
[resA, srcA, nameA] = collect_folders(aiDirs, 'AI', T0);
numRealFiles = numel(resR);  numAiFiles = numel(resA);

Rv = [resR.score];  Av = [resA.score];
Rv = Rv(isfinite(Rv));  Av = Av(isfinite(Av));

T = T0;
Aauc = auc_(Av, Rv);
[thr, dsign, bacc] = sweep_cut(Rv, Av);

fprintf('\n------------------------ CALIBRATION RESULT ------------------------\n');
fprintf(' %-16s %11s %6s %10s %8s  %s\n','trace','cutoff','dir','bal.acc','AUC','status');
fprintf(' ---------------------------------------------------------------------\n');
if abs(Aauc - 0.5) < 0.05
    T.active = false;  st = 'OFF (at chance)';
elseif Aauc < 0.5
    T.active = true;   st = 'ON (inverted: real scores higher)';
else
    T.active = true;   st = 'ON';
end
T.thr = thr;  T.dir = dsign;
fprintf(' %-16s %11.5f %6s %9.1f%% %8.3f  %s\n', T.name, thr, dirsym(dsign), 100*bacc, Aauc, st);
fprintf(' ---------------------------------------------------------------------\n');

if ~T.active
    fprintf('\n [!] The trace is at chance on this data.  It does not separate\n');
    fprintf('     these folders; report that, not a number.  The rule is saved\n');
    fprintf('     but will flag nothing (single trace, no vote to fall back on).\n');
end

fprintf('\n------------------------- THE VERDICT ---------------------------\n');
fprintf(' texture contrast trace : balanced accuracy %.1f%%,  AUC %.3f\n', 100*bacc, Aauc);
fprintf('-----------------------------------------------------------------\n');

T.source = sprintf('tuned on %d real (%d folders) + %d AI (%d folders), %s', ...
                   numRealFiles, numel(realDirs), numAiFiles, numel(aiDirs), ...
                   datestr(now,'yyyy-mm-dd'));
T.perf = struct('acc',NaN,'bacc',bacc,'auc',Aauc);
save('texture_contrast_thresholds.mat','T');
fprintf('\n Saved to texture_contrast_thresholds.mat and now in use.\n');
fprintf(' To hard-code, put these in decision_thresholds():\n');
fprintf('   T.thr    = %.6f;\n', T.thr);
fprintf('   T.dir    = %+d;\n', T.dir);
fprintf('   T.active = %s;\n', tf(T.active));

% evaluate accuracy at this cutoff, filled in with real verdicts
for i = 1:numel(resR), resR(i).verdict = tern(is_synthetic(resR(i),T),'SYNTHETIC','AUTHENTIC'); end
for i = 1:numel(resA), resA(i).verdict = tern(is_synthetic(resA(i),T),'SYNTHETIC','AUTHENTIC'); end
print_accuracy(resR, resA, T);

if numel(nameA) > 1
    fprintf('\n Per-generator recall with the tuned rule:\n');
    fprintf('  %-22s %6s %10s\n','AI source','N','caught');
    fprintf('  %s\n', repmat('-',1,40));
    worst = Inf; worstName = '';
    for i = 1:numel(nameA)
        m = (srcA == i);
        if ~any(m), continue; end
        ri = resA(m);
        caught = 0;
        for q = 1:numel(ri), caught = caught + is_synthetic(ri(q), T); end
        rate = caught/max(numel(ri),1);
        fprintf('  %-22s %6d %9.1f%%\n', trunc_(nameA{i},22), numel(ri), 100*rate);
        if rate < worst, worst = rate; worstName = nameA{i}; end
    end
    if worst < 0.5
        fprintf(['\n [!] Recall on "%s" is only %.0f%%.  The rule does not\n' ...
                 '     transfer to that generator.  Quote the per-source table.\n'], worstName, 100*worst);
    end
end
if numel(nameR) > 1
    fprintf('\n Per-source false-positive rate on REAL:\n');
    fprintf('  %-22s %6s %10s\n','REAL source','N','flagged');
    fprintf('  %s\n', repmat('-',1,40));
    for i = 1:numel(nameR)
        m = (srcR == i);
        if ~any(m), continue; end
        ri = resR(m);
        flagged = 0;
        for q = 1:numel(ri), flagged = flagged + is_synthetic(ri(q), T); end
        fprintf('  %-22s %6d %9.1f%%\n', trunc_(nameR{i},22), numel(ri), ...
                100*flagged/max(numel(ri),1));
    end
end

check_resolution(resR, resA);
if bacc < 0.60
    fprintf('\n [!] Balanced accuracy is under 60%%.  On this data the method\n');
    fprintf('     is close to guessing; report that rather than the number.\n');
end
fprintf('\n');

draw_batch_figures(resR, resA, T);
end

function s = tf(b)
if b, s = 'true '; else, s = 'false'; end
end

function [thr, dsign, bacc] = sweep_cut(cr, cf)
% Best single cutoff, trying BOTH directions.
thr = NaN; dsign = 1; bacc = NaN;
if numel(cr) < 2 || numel(cf) < 2, return; end
allv = [cr(:); cf(:)];
cand = unique(allv);
if numel(cand) > 500
    cand = unique(quantile_(allv, linspace(0,1,500)));
end
best = -Inf;  thr = median(allv);
for t = cand(:)'
    for ds = [1 -1]
        if ds > 0
            ba = 0.5*(mean(cf > t) + mean(cr <= t));
        else
            ba = 0.5*(mean(cf < t) + mean(cr >= t));
        end
        if ba > best, best = ba; thr = t; dsign = ds; end
    end
end
bacc = best;
end

% =========================================================================
% CONFOUND CHECK
% =========================================================================
function check_resolution(resR, resA)
% Texture-contrast can respond to resolution and JPEG quality as well as to
% synthesis.  If the two folders differ a lot in size, part of any
% separation may be image size, not the generator.
pr = [resR.npix];  pa = [resA.npix];
pr = pr(isfinite(pr));  pa = pa(isfinite(pa));
if isempty(pr) || isempty(pa), return; end
mr = median(pr);  ma = median(pa);
ratio = max(mr,ma)/max(min(mr,ma),1);
fprintf('\nConfound check (image size):\n');
fprintf('  median pixels: REAL %.2f MP, AI %.2f MP  (ratio %.2fx)\n', ...
        mr/1e6, ma/1e6, ratio);
if ratio > 1.6
    fprintf('  [!] The folders differ by more than 1.6x in size.  The contrast\n');
    fprintf('      trace can read resolution as well as synthesis, so part of\n');
    fprintf('      its separation may be image size, not the generator.\n');
else
    fprintf('  sizes are comparable; the trace is not obviously reading\n');
    fprintf('  resolution.\n');
end
end

% =========================================================================
% MULTI-FOLDER SELECTION
% =========================================================================
function dirs = pick_folders(label, startDir)
dirs = {};
if nargin < 2 || isempty(startDir), startDir = pwd; end
fprintf('\n Select %s folder(s).  Cancel when you have added them all.\n', label);
while true
    prompt = sprintf('%s folder #%d   (Cancel when done)', label, numel(dirs)+1);
    f = uigetdir(startDir, prompt);
    if isequal(f, 0), break; end
    if any(strcmpi(dirs, f))
        fprintf('   already added, skipping: %s\n', f);
        continue;
    end
    n = numel(list_images(f));
    if n == 0
        fprintf('   [!] no images found in %s -- not added\n', f);
        continue;
    end
    dirs{end+1} = f; %#ok<AGROW>
    startDir = f;
    fprintf('   added (%d images): %s\n', n, f);
end
if isempty(dirs)
    fprintf(' No %s folder chosen.\n', label);
end
end

function files = list_images(folder)
exts = {'*.jpg','*.jpeg','*.png','*.tif','*.tiff','*.bmp','*.webp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(folder, exts{i}))]; %#ok<AGROW>
end
if ~isempty(files), files = files(~[files.isdir]); end
end

% -------------------------------------------------------------------------
function [res, srcIdx, names] = collect_folders(dirs, label, T)
res = [];  srcIdx = [];  names = cell(1,numel(dirs));
for i = 1:numel(dirs)
    files = list_images(dirs{i});
    [~, nm] = fileparts(dirs{i});
    if isempty(nm), nm = dirs{i}; end
    names{i} = nm;
    fprintf('\n%s source %d/%d: %s  (%d images)\n', ...
            label, i, numel(dirs), nm, numel(files));
    r = process_file_list(dirs{i}, files, T);
    res    = [res; r];                       %#ok<AGROW>
    srcIdx = [srcIdx; i*ones(numel(r),1)];   %#ok<AGROW>
end
end

% -------------------------------------------------------------------------
function print_per_source(label, res, srcIdx, names, T) %#ok<INUSD>
if numel(names) < 2, return; end
fprintf('\n  %s, per source:\n', label);
fprintf('  %-22s %6s %10s %10s\n','source','N','flagged','medScore');
fprintf('  %s\n', repmat('-',1,52));
for i = 1:numel(names)
    m = (srcIdx == i);
    if ~any(m), continue; end
    ri = res(m);
    nS = sum(strcmp({ri.verdict},'SYNTHETIC'));
    sc = [ri.score]; sc = sc(isfinite(sc));
    fprintf('  %-22s %6d %9.1f%% %10.4f\n', ...
            trunc_(names{i},22), numel(ri), 100*nS/max(numel(ri),1), medsafe(sc));
end
end

function m = medsafe(v)
if isempty(v), m = NaN; else, m = median(v); end
end

function s = trunc_(s, n)
if numel(s) > n, s = [s(1:n-3) '...']; end
end

% =========================================================================
% BATCH PLUMBING
% =========================================================================
function res = process_file_list(baseDir, files, T)
n = numel(files);
res = repmat(struct('name','','ep',NaN,'er',NaN,'contrast',NaN,'score',NaN, ...
                    'npix',NaN,'verdict',''), n, 1);
t0 = tic;
for i = 1:n
    try
        img = im2double(imread(fullfile(baseDir, files(i).name)));
        out = compute_texture_features(img);
        res(i).name = files(i).name;
        res(i).npix = size(img,1)*size(img,2);
        if ~out.ok, res(i).verdict = 'TOO SMALL'; continue; end
        res(i).ep       = out.ep;
        res(i).er       = out.er;
        res(i).contrast = out.contrast;
        res(i).score    = out.score;
        if is_synthetic(out, T)
            res(i).verdict = 'SYNTHETIC';
        else
            res(i).verdict = 'AUTHENTIC';
        end
    catch
        res(i).name = files(i).name;
        res(i).verdict = 'ERROR';
    end
    if mod(i,20)==0
        fprintf('  Processed %d/%d images (%.1f s elapsed)\n', i, n, toc(t0));
    end
end
end

function print_batch_stats(label, res, T)
n = numel(res);
nAuth  = sum(strcmp({res.verdict},'AUTHENTIC'));
nSynth = sum(strcmp({res.verdict},'SYNTHETIC'));
nErr   = sum(strcmp({res.verdict},'ERROR'));
nSmall = sum(strcmp({res.verdict},'TOO SMALL'));
fprintf('%s (N = %d):\n', label, n);
fprintf('  * Consistent Pipeline Signature : %4d (%5.1f%%)\n', nAuth, 100*nAuth/n);
fprintf('  * Split Synthesis Signature     : %4d (%5.1f%%)\n', nSynth, 100*nSynth/n);
v = [res.score]; v = v(isfinite(v));
if ~isempty(v)
    fprintf('  * Mean score                    : %8.4f  (cutoff %.4f)\n', mean(v), T.thr);
end
if nSmall>0, fprintf('  * Too Small For 32x32 Patches   : %4d (%5.1f%%)\n', nSmall, 100*nSmall/n); end
if nErr>0,   fprintf('  * Image Read Errors             : %4d (%5.1f%%)\n', nErr, 100*nErr/n); end
fprintf('\n');
end

function print_accuracy(resReal, resAI, T) %#ok<INUSD>
[tp,tn,fp,fn,acc,bacc,prec,rec,f1] = compute_confusion(resReal, resAI);
tot = tp+tn+fp+fn;
if tot == 0, return; end

realAcc = tn/max(tn+fp,1);   % fraction of the REAL folder correctly called AUTHENTIC
aiAcc   = tp/max(tp+fn,1);   % fraction of the AI folder correctly called SYNTHETIC

fprintf('Threshold Performance On These Folders:\n');
fprintf('  REAL folder accuracy : %5.1f%%   (%d/%d correctly called AUTHENTIC)\n', ...
        100*realAcc, tn, tn+fp);
fprintf('  AI folder accuracy   : %5.1f%%   (%d/%d correctly called SYNTHETIC)\n', ...
        100*aiAcc, tp, tp+fn);
fprintf('  ---------------------------------------------------------------\n');
fprintf('  Overall accuracy   : %5.1f%%\n', 100*acc);
fprintf('  Balanced accuracy  : %5.1f%%\n', 100*bacc);
fprintf('  Precision          : %5.3f\n', prec);
fprintf('  Recall             : %5.3f\n', rec);
fprintf('  F1                 : %5.3f\n', f1);
fprintf('  Confusion          : TP %d  TN %d  FP %d  FN %d\n', tp, tn, fp, fn);

scR = [resReal.score]; scA = [resAI.score];
a = auc_(scA, scR);
fprintf('\n  Trace AUC on these folders (0.5 = useless, <0.5 = inverted): %.3f\n', a);
end

function [tp,tn,fp,fn,acc,bacc,prec,rec,f1] = compute_confusion(resReal, resAI)
tn = sum(strcmp({resReal.verdict},'AUTHENTIC'));
fp = sum(strcmp({resReal.verdict},'SYNTHETIC'));
tp = sum(strcmp({resAI.verdict},  'SYNTHETIC'));
fn = sum(strcmp({resAI.verdict},  'AUTHENTIC'));
tot = tp+tn+fp+fn;
if tot == 0
    acc = NaN; bacc = NaN; prec = NaN; rec = NaN; f1 = NaN; return;
end
acc  = (tp+tn)/tot;
bacc = 0.5*(tp/max(tp+fn,1) + tn/max(tn+fp,1));
prec = tp/max(tp+fp,1);
rec  = tp/max(tp+fn,1);
f1   = 2*prec*rec/max(prec+rec,eps);
end


% =========================================================================
% CORE ENGINE: SMASH & RECONSTRUCTION + TEXTURE CONTRAST
%   (kept identical in logic to visualize_texture_contrast.m)
% =========================================================================
function out = compute_texture_features(img)
ps = 32;
np = 192;
SRM = [-1 2 -2 2 -1; 2 -6 8 -6 2; -2 8 -12 8 -2; 2 -6 8 -6 2; -1 2 -2 2 -1]/12;

out = struct('ok',false,'g',[],'poor',[],'rich',[], ...
             'ep',NaN,'er',NaN,'contrast',NaN,'score',NaN,'npix',NaN);

if size(img,3) == 3
    g = img(:,:,2);   % Extract Green channel only
else
    g = img;
end

[H,W] = size(g);
out.g = g;
out.npix = H*W;
if H <= ps || W <= ps
    return;   % too small for 32x32 patch sampling
end

rng(0);
P = cell(1,np);
d = zeros(1,np);
for k = 1:np
    i = randi(H-ps);
    j = randi(W-ps);
    q = g(i:i+ps-1, j:j+ps-1);
    P{k} = q;

    d(k) = sum(abs(diff(q,1,2)),'all') + sum(abs(diff(q,1,1)),'all') + ...
           sum(abs(q(2:end,2:end)-q(1:end-1,1:end-1)),'all') + ...
           sum(abs(q(2:end,1:end-1)-q(1:end-1,2:end)),'all');
end

[~,ord] = sort(d);
third = floor(np/3);
poor_patches = P(ord(1:third));
rich_patches = P(ord(end-third+1:end));

grid_size = ceil(sqrt(third));
poor_img = zeros(grid_size*ps, grid_size*ps);
rich_img = zeros(grid_size*ps, grid_size*ps);

for idx = 1:third
    row = floor((idx-1)/grid_size) * ps + 1;
    col = mod(idx-1, grid_size) * ps + 1;

    poor_img(row:row+ps-1, col:col+ps-1) = poor_patches{idx};
    rich_img(row:row+ps-1, col:col+ps-1) = rich_patches{idx};
end
out.poor = poor_img;
out.rich = rich_img;

E = @(C) mean(cellfun(@(q) mean(abs(conv2(q,SRM,'valid')),'all'), C));
out.ep = E(poor_patches);
out.er = E(rich_patches);
out.contrast = (out.er-out.ep)/(out.er+out.ep+eps);
out.score = -out.contrast;   % F.tex_contrast convention from aigi_features

out.ok = true;
end

% =========================================================================
% MATHEMATICAL SUBROUTINES
% =========================================================================
function A = auc_(pos, neg)
% Mann-Whitney AUC: probability a random pos ranks above a random neg.
pos = pos(isfinite(pos)); neg = neg(isfinite(neg));
n1 = numel(pos); n2 = numel(neg);
if n1 == 0 || n2 == 0, A = NaN; return; end
[v,i] = sort([pos(:); neg(:)]);
r = zeros(size(v));  r(i) = 1:numel(v);
u = unique(v);
for k = 1:numel(u)
    m = (v == u(k));
    if nnz(m) > 1, r(i(m)) = mean(r(i(m))); end
end
A = (sum(r(1:n1)) - n1*(n1+1)/2)/(n1*n2);
end

function q = quantile_(x, p)
x = x(:); x = x(isfinite(x));
if isempty(x), q = nan(size(p)); return; end
x = sort(x); n = numel(x);
pos = p(:)*n + 0.5;
lo = min(max(floor(pos),1),n);
hi = min(max(ceil(pos), 1),n);
wt = pos - floor(pos);
q  = reshape(x(lo).*(1-wt) + x(hi).*wt, size(p));
end
