function Test07_NyquistOnly
% TEST07_NYQUISTONLY
% AI vs REAL detector using ONLY the camera Nyquist (CFA) test.
%
% Method (green channel):
%   1. 3x3 Laplacian residual e
%   2. Energy map p = e.^2 minus its local 9x9 mean
%   3. Hann window + 2-D FFT, |F| = P
%   4. Score = 20 log10( peak at (pi,pi) / median of P )
%
% Decision:
%   score >= NYQ_HIGH  -> AUTHENTIC_CAMERA   (camera demosaicing pattern present)
%   score <  NYQ_HIGH  -> AI_GENERATED       (no camera pattern)
clc;
close all;

fprintf('=================================================================\n');
fprintf('        AI vs REAL NYQUIST (CFA) FORENSIC ANALYZER\n');
fprintf('=================================================================\n');

modeChoice = menu('Select Analysis Mode', ...
                  'Compare Single Pair (1 Real vs 1 AI)', ...
                  'Batch Folder Analysis (Real Folder vs AI Folder)', ...
                  'Exit');
switch modeChoice
    case 1
        run_single_pair();
    case 2
        run_batch_folders();
    otherwise
        disp('Exiting.');
end
end

% =========================================================================
% SETTINGS (single place to edit)
% =========================================================================
function S = detector_settings()
S.NYQ_HIGH   = 12.0;   % dB. Replace with the tuned value printed in batch mode
S.SPLIT_SEED = 1;      % random seed for the train/test split
end

% =========================================================================
% MODE 1: SINGLE PAIR
% =========================================================================
function run_single_pair()
S = detector_settings();
flt = {'*.jpg;*.jpeg;*.jfif;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'};
[realFile, realPath] = uigetfile(flt, 'Select ONE REAL image');
if isequal(realFile,0), return; end
[aiFile, aiPath] = uigetfile(flt, 'Select ONE AI image');
if isequal(aiFile,0), return; end

R_real = analyze_image(fullfile(realPath, realFile), S, true);
R_ai   = analyze_image(fullfile(aiPath, aiFile), S, true);

fprintf('\n----------------- DIAGNOSTIC RESULTS -----------------\n');
fprintf('REAL [%s]: Nyq = %6.2f dB | Verdict: %s\n', realFile, R_real.nyqScore, R_real.verdict);
fprintf('AI   [%s]: Nyq = %6.2f dB | Verdict: %s\n', aiFile,   R_ai.nyqScore,   R_ai.verdict);
fprintf('Threshold: %.2f dB\n', S.NYQ_HIGH);
fprintf('------------------------------------------------------\n');

% Spatial tile maps
figure('Name', 'Spatial Nyquist Maps (dB)', 'Position', [100 100 1100 450]);
subplot(1,2,1);
imagesc(R_real.spatialMap); axis image; colorbar; colormap jet;
title(sprintf('REAL: %s\nScore: %.2f dB | Verdict: %s', realFile, R_real.nyqScore, R_real.verdict), ...
    'Interpreter', 'none');
subplot(1,2,2);
imagesc(R_ai.spatialMap); axis image; colorbar; colormap jet;
title(sprintf('AI: %s\nScore: %.2f dB | Verdict: %s', aiFile, R_ai.nyqScore, R_ai.verdict), ...
    'Interpreter', 'none');

% Score comparison
figure('Name', 'Nyquist Score Comparison', 'Position', [150 150 500 420]);
bar([R_real.nyqScore, R_ai.nyqScore]); hold on;
yline(S.NYQ_HIGH, '--k', sprintf('thr = %g dB', S.NYQ_HIGH), 'LineWidth', 1.3);
set(gca, 'XTickLabel', {'REAL', 'AI'});
ylabel('Nyquist score (dB)'); title('Nyquist Grid Energy at (\pi,\pi)'); grid on;

% Energy-map spectra (DC at top-left), with the (pi,pi) probe marked
figure('Name', 'Energy-map spectra', 'Position', [200 120 1100 480]);
RR = {R_real, R_ai}; nm = {['REAL: ' realFile], ['AI: ' aiFile]};
for k = 1:2
    subplot(1,2,k);
    P = RR{k}.spectrum;
    [h, w] = size(P);
    L = log10(P + eps);
    imagesc(L); axis image; colorbar; colormap(gca, parula); hold on;
    q = pct(L(:), [1 99.9]); if q(2) > q(1), caxis([q(1) q(2)]); end
    plot(w/2 + 1, h/2 + 1, 'ro', 'MarkerSize', 16, 'LineWidth', 2);
    xlabel('horizontal frequency index'); ylabel('vertical frequency index');
    title(sprintf('%s\n(\\pi,\\pi) probe circled | %.2f dB', strrep(nm{k}, '_', '\_'), RR{k}.nyqScore));
end
end

% =========================================================================
% MODE 2: BATCH FOLDERS
% =========================================================================
function run_batch_folders()
S = detector_settings();
realDir = uigetdir(pwd, 'Select Folder Containing REAL Images');
if isequal(realDir,0), return; end
aiDir = uigetdir(pwd, 'Select Folder Containing AI Images');
if isequal(aiDir,0), return; end

realFiles = list_images(realDir);
aiFiles   = list_images(aiDir);
fprintf('REAL folder: %s  -> %d images found\n', realDir, numel(realFiles));
fprintf('AI folder  : %s  -> %d images found\n', aiDir,   numel(aiFiles));
if isempty(realFiles) || isempty(aiFiles)
    error('No supported images found in the folder(s) showing 0 above.');
end

fprintf('\nAnalyzing REAL folder...\n');
resReal = process_file_list(realFiles, S);
fprintf('\nAnalyzing AI folder...\n');
resAI = process_file_list(aiFiles, S);

assignin('base', 'resReal', resReal);
assignin('base', 'resAI',   resAI);

fprintf('\n=================================================================\n');
fprintf('                       BATCH SUMMARY REPORT                      \n');
fprintf('=================================================================\n');
safe_run(@() report_accuracy(resReal, resAI, S), 'accuracy report');
safe_run(@() report_tuned(resReal, resAI, S),    'tuned threshold');

safe_run(@() plot_histogram(resReal, resAI, S),        'histogram');
safe_run(@() plot_roc(resReal, resAI, S),              'ROC curve');
safe_run(@() plot_threshold_sweep(resReal, resAI, S),  'threshold sweep');
safe_run(@() plot_strip(resReal, resAI, S),            'per-image scores');
safe_run(@() plot_verdict_bars(resReal, resAI, S),     'verdict bars');

try
    writetable(struct2table(resReal), fullfile(pwd, 'nyquist_only_REAL.csv'));
    writetable(struct2table(resAI),   fullfile(pwd, 'nyquist_only_AI.csv'));
    fprintf('Per-image results saved to %s\n', pwd);
catch ME
    fprintf('Could not save CSV files: %s\n', ME.message);
end
end

function res = process_file_list(files, S)
n = numel(files);
res = repmat(struct('name','', 'nyqScore',NaN, 'verdict',''), n, 1);
t0 = tic;
for i = 1:n
    filePath = fullfile(files(i).folder, files(i).name);
    res(i).name = files(i).name;
    try
        out = analyze_image(filePath, S, false);
        res(i).nyqScore = out.nyqScore;
        res(i).verdict  = out.verdict;
    catch ME
        warning('%s: %s', files(i).name, ME.message);
        res(i).verdict = 'ERROR';
    end
    if mod(i, 20) == 0
        fprintf('  Processed %d/%d images (%.1f s elapsed)\n', i, n, toc(t0));
    end
end
end

% =========================================================================
% REPORTS
% =========================================================================
function report_accuracy(resReal, resAI, S)
a = [resReal.nyqScore]; b = [resAI.nyqScore];
nErrR = sum(isnan(a)); nErrA = sum(isnan(b));
a = a(~isnan(a)); b = b(~isnan(b));

TP = sum(a >= S.NYQ_HIGH); FN = sum(a < S.NYQ_HIGH);
FP = sum(b >= S.NYQ_HIGH); TN = sum(b < S.NYQ_HIGH);
tpr = TP / max(TP+FN, 1); tnr = TN / max(TN+FP, 1);
[~, ~, bestThr, bestAcc] = roc_curve(a, b);

fprintf('Images analysed: REAL %d, AI %d', numel(a), numel(b));
if nErrR + nErrA > 0, fprintf(' (read errors: REAL %d, AI %d)', nErrR, nErrA); end
fprintf('\n\n');
fprintf('Nyquist score  : median REAL %.2f dB | median AI %.2f dB\n', median(a), median(b));
fprintf('AUC            : %.3f\n\n', auc_pairwise(a, b));

fprintf('---------- ACCURACY AT CURRENT THRESHOLD (%.2f dB) ----------\n', S.NYQ_HIGH);
fprintf('               called CAMERA   called AI\n');
fprintf('  REAL          %6d        %6d\n', TP, FN);
fprintf('  AI            %6d        %6d\n', FP, TN);
fprintf('REAL correct %5.1f%% | AI correct %5.1f%% | accuracy %5.1f%% | balanced %5.1f%%\n', ...
    100*tpr, 100*tnr, 100*(TP+TN)/max(TP+FN+FP+TN,1), 50*(tpr+tnr));
fprintf('Best threshold on ALL images: %.2f dB -> balanced %.1f%% (optimistic)\n', bestThr, 100*bestAcc);
fprintf('--------------------------------------------------------------\n\n');
end

function report_tuned(resReal, resAI, S)
% Choose the threshold on one random half, measure accuracy on the other half
a = [resReal.nyqScore]; b = [resAI.nyqScore];
a = a(~isnan(a)); b = b(~isnan(b));
rng(S.SPLIT_SEED);
ia = randperm(numel(a)); ib = randperm(numel(b));
trA = a(ia(1:floor(end/2))); teA = a(ia(floor(end/2)+1:end));
trB = b(ib(1:floor(end/2))); teB = b(ib(floor(end/2)+1:end));

[~, ~, thr, trainAcc] = roc_curve(trA, trB);
cR = mean(teA >= thr); cA = mean(teB < thr);

fprintf('---------- TUNED ON 50%%, TESTED ON THE OTHER 50%% ----------\n');
fprintf('Train: %d REAL + %d AI | Test: %d REAL + %d AI\n', ...
    numel(trA), numel(trB), numel(teA), numel(teB));
fprintf('Tuned threshold : %.2f dB (train balanced %.1f%%)\n', thr, 100*trainAcc);
fprintf('Test result     : REAL correct %5.1f%% | AI correct %5.1f%% | balanced %5.1f%%\n', ...
    100*cR, 100*cA, 50*(cR+cA));
fprintf('Test AUC        : %.3f\n', auc_pairwise(teA, teB));
fprintf('=> Report the TEST numbers. Put %.2f into NYQ_HIGH in detector_settings.\n', thr);
fprintf('--------------------------------------------------------------\n\n');
end

% =========================================================================
% PLOTS
% =========================================================================
function plot_histogram(resReal, resAI, S)
a = [resReal.nyqScore]; b = [resAI.nyqScore];
a = a(~isnan(a)); b = b(~isnan(b));
q = pct([a(:); b(:)], [1 99]);
lo = min(q(1), S.NYQ_HIGH); hi = max(q(2), S.NYQ_HIGH);
edges = linspace(lo, hi, 36);
figure('Name', 'Nyquist score distribution', 'Position', [60 60 800 500]);
histogram(min(max(a, lo), hi), edges, 'Normalization', 'probability', ...
    'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', sprintf('REAL (n=%d)', numel(a)));
hold on;
histogram(min(max(b, lo), hi), edges, 'Normalization', 'probability', ...
    'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', sprintf('AI (n=%d)', numel(b)));
xline(S.NYQ_HIGH, 'k--', sprintf('thr = %g dB', S.NYQ_HIGH), 'LineWidth', 1.5, 'HandleVisibility', 'off');
title({'Nyquist score distribution', summary_text(a, b, S)});
xlabel('Nyquist score (dB)'); ylabel('fraction of images');
legend('Location', 'northeast'); grid on;
add_result_box(a, b, S);
end

function plot_roc(resReal, resAI, S)
a = [resReal.nyqScore]; b = [resAI.nyqScore];
a = a(~isnan(a)); b = b(~isnan(b));
[fpr, tpr] = roc_curve(a, b);
figure('Name', 'ROC curve', 'Position', [120 120 700 620]);
plot(fpr, tpr, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Nyquist (AUC %.3f)', auc_pairwise(a, b)));
hold on;
plot(mean(b >= S.NYQ_HIGH), mean(a >= S.NYQ_HIGH), 'kp', 'MarkerSize', 16, ...
    'MarkerFaceColor', 'y', 'DisplayName', sprintf('thr = %g dB', S.NYQ_HIGH));
plot([0 1], [0 1], 'k:', 'HandleVisibility', 'off');
xlabel('False positive rate (AI called REAL)');
ylabel('True positive rate (REAL called REAL)');
title({'ROC, REAL = positive class', summary_text(a, b, S)}, 'FontSize', 9);
legend('Location', 'southeast'); grid on; axis square;
add_result_box(a, b, S);
end

function plot_threshold_sweep(resReal, resAI, S)
a = [resReal.nyqScore]; b = [resAI.nyqScore];
a = a(~isnan(a)); b = b(~isnan(b));
q = pct([a(:); b(:)], [1 99]);
t = linspace(q(1), q(2), 300);
cR = arrayfun(@(x) mean(a >= x), t);
cA = arrayfun(@(x) mean(b < x), t);
bal = (cR + cA) / 2;
[bMax, iMax] = max(bal);
figure('Name', 'Accuracy vs threshold', 'Position', [160 160 800 480]);
plot(t, 100*cR, 'b-', 'LineWidth', 1.8, 'DisplayName', 'REAL correct'); hold on;
plot(t, 100*cA, 'r-', 'LineWidth', 1.8, 'DisplayName', 'AI correct');
plot(t, 100*bal, 'k-', 'LineWidth', 2.2, 'DisplayName', 'Balanced accuracy');
xline(S.NYQ_HIGH, 'k--', sprintf('current %g dB', S.NYQ_HIGH), 'LineWidth', 1.2, 'HandleVisibility', 'off');
plot(t(iMax), 100*bMax, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', ...
    'DisplayName', sprintf('best %.2f dB (%.1f%%)', t(iMax), 100*bMax));
xlabel('Threshold (dB)'); ylabel('% correct'); ylim([0 100]);
title({'Accuracy vs threshold', summary_text(a, b, S)});
legend('Location', 'southeast'); grid on;
add_result_box(a, b, S);
end

function plot_strip(resReal, resAI, S)
a = [resReal.nyqScore]; b = [resAI.nyqScore];
a = a(~isnan(a)); b = b(~isnan(b));
rng(0);
figure('Name', 'Per-image Nyquist scores', 'Position', [200 200 800 560]);
scatter(1 + 0.3*(rand(size(a)) - 0.5), a, 18, 'b', 'filled', 'MarkerFaceAlpha', 0.4, ...
    'DisplayName', 'REAL'); hold on;
scatter(2 + 0.3*(rand(size(b)) - 0.5), b, 18, 'r', 'filled', 'MarkerFaceAlpha', 0.4, ...
    'DisplayName', 'AI');
yline(S.NYQ_HIGH, 'k--', sprintf('thr = %g dB', S.NYQ_HIGH), 'LineWidth', 1.4, 'HandleVisibility', 'off');
set(gca, 'XTick', [1 2], 'XTickLabel', {'REAL', 'AI'}); xlim([0.5 2.5]);
ylabel('Nyquist score (dB)');
title({'Every image, with the decision threshold', summary_text(a, b, S)}, 'FontSize', 9);
legend('Location', 'northeast'); grid on;
add_result_box(a, b, S);
end

function plot_verdict_bars(resReal, resAI, S)
cats = {'AUTHENTIC_CAMERA', 'AI_GENERATED'};
M = zeros(2, numel(cats));
for k = 1:numel(cats)
    M(1, k) = 100 * mean(strcmp({resReal.verdict}, cats{k}));
    M(2, k) = 100 * mean(strcmp({resAI.verdict},   cats{k}));
end
a = [resReal.nyqScore]; b = [resAI.nyqScore];
a = a(~isnan(a)); b = b(~isnan(b));
figure('Name', 'Verdict distribution', 'Position', [240 240 900 480]);
hb = bar(M, 'stacked');
set(gca, 'XTickLabel', {'REAL folder', 'AI folder'});
ylabel('% of images'); ylim([0 100]);
legend(hb, strrep(cats, '_', ' '), 'Location', 'eastoutside');
title({'Verdicts per folder', summary_text(a, b, S)}, 'FontSize', 9); grid on;
% percentage labels inside each bar segment
base = zeros(2, 1);
for k = 1:numel(cats)
    for r = 1:2
        if M(r, k) >= 4
            text(r, base(r) + M(r, k)/2, sprintf('%.1f%%', M(r, k)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'w');
        end
    end
    base = base + M(:, k);
end
end

function [txt, cR, cA] = summary_text(a, b, S)
% One-line result shown on every batch figure
cR = mean(a >= S.NYQ_HIGH); cA = mean(b < S.NYQ_HIGH);
txt = sprintf('AUC = %.3f  |  REAL correct %.1f%%  |  AI correct %.1f%%  |  balanced %.1f%%', ...
    auc_pairwise(a, b), 100*cR, 100*cA, 50*(cR + cA));
end

function add_result_box(a, b, S)
% Text box inside the axes with AUC and per-class accuracy
[~, cR, cA] = summary_text(a, b, S);
str = {sprintf('AUC = %.3f', auc_pairwise(a, b)), ...
       sprintf('Threshold = %.2f dB', S.NYQ_HIGH), ...
       sprintf('REAL correct = %.1f%%', 100*cR), ...
       sprintf('AI correct   = %.1f%%', 100*cA), ...
       sprintf('Balanced     = %.1f%%', 50*(cR + cA))};
text(0.02, 0.97, str, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontName', 'Consolas', 'FontSize', 10, 'BackgroundColor', [1 1 1], ...
    'EdgeColor', [0.3 0.3 0.3], 'Margin', 5);
end

% =========================================================================
% CORE ENGINE
% =========================================================================
function R = analyze_image(filePath, S, keepMaps)
[imgRaw, map] = imread(filePath);
if ~isempty(map), imgRaw = ind2rgb(imgRaw, map); end
rgb = im2double(imgRaw);
if ndims(rgb) == 2, rgb = repmat(rgb, [1 1 3]); end
if size(rgb, 3) > 3, rgb = rgb(:, :, 1:3); end
g = rgb(:, :, 2);   % green channel has the densest CFA sampling

[score, spatialMap, P] = compute_nyquist_periodicity(g, keepMaps);

R.nyqScore = score;
if score >= S.NYQ_HIGH
    R.verdict = 'AUTHENTIC_CAMERA';
else
    R.verdict = 'AI_GENERATED';
end
if keepMaps
    R.spatialMap = spatialMap;
    R.spectrum   = P;
end
end

function [score, scoreMap, P] = compute_nyquist_periodicity(g, doMap)
% Camera CFA fingerprint: squared Laplacian residual, peak at (pi,pi)
LAP = [0 1 0; 1 -4 1; 0 1 0];
e = conv2(g, LAP, 'same');
p = e.^2;
p = p - conv2(p, ones(9)/81, 'same');
p = p(9:end-8, 9:end-8);

[h, w] = size(p);
h = 2*floor(h/2); w = 2*floor(w/2);
p = p(1:h, 1:w);

P = abs(fft2(p .* hann2(h, w)));
iy = h/2 + 1; ix = w/2 + 1;
peakVal = max(max(P(iy-1:iy+1, ix-1:ix+1)));
score = 20*log10((peakVal + eps)/(median(P(:)) + eps));

scoreMap = [];
if ~doMap, return; end
bs = 128;
ny = floor(h/bs); nx = floor(w/bs);
scoreMap = nan(ny, nx);
for r = 1:ny
    for c = 1:nx
        blk = p((r-1)*bs+1:r*bs, (c-1)*bs+1:c*bs);
        B = abs(fft2(blk .* hann2(bs, bs)));
        pk = max(max(B(bs/2:bs/2+2, bs/2:bs/2+2)));
        scoreMap(r, c) = 20*log10((pk + eps)/(median(B(:)) + eps));
    end
end
end

% =========================================================================
% STATISTICS HELPERS (no toolbox needed)
% =========================================================================
function auc = auc_pairwise(pos, neg)
% P(REAL score > AI score), ties count 0.5 (rank-based, memory-safe)
pos = double(pos(~isnan(pos))); neg = double(neg(~isnan(neg)));
n1 = numel(pos); n0 = numel(neg);
if n1 == 0 || n0 == 0, auc = NaN; return; end
r = tie_rank([pos(:); neg(:)]);
auc = (sum(r(1:n1)) - n1*(n1+1)/2) / (n1*n0);
end

function r = tie_rank(x)
x = x(:); n = numel(x);
[xs, idx] = sort(x);
r = zeros(n, 1);
i = 1;
while i <= n
    j = i;
    while j < n && xs(j+1) == xs(i), j = j + 1; end
    r(idx(i:j)) = (i + j) / 2;
    i = j + 1;
end
end

function [fpr, tpr, bestThr, bestAcc] = roc_curve(pos, neg)
% ROC for the rule "REAL if score >= thr"; best = max balanced accuracy
pos = double(pos(~isnan(pos))); neg = double(neg(~isnan(neg)));
n1 = numel(pos); n0 = numel(neg);
if n1 == 0 || n0 == 0
    fpr = [0; 1]; tpr = [0; 1]; bestThr = NaN; bestAcc = NaN; return;
end
v   = [pos(:); neg(:)];
lab = [ones(n1, 1); zeros(n0, 1)];
[v, o] = sort(v, 'descend'); lab = lab(o);
tp = cumsum(lab); fp = cumsum(1 - lab);
keep = [v(1:end-1) ~= v(2:end); true];
tpr  = [0; tp(keep) / n1];
fpr  = [0; fp(keep) / n0];
thrv = [inf; v(keep)];
bal = (tpr + 1 - fpr) / 2;
[bestAcc, i] = max(bal);
bestThr = thrv(i);
end

function q = pct(x, p)
x = sort(double(x(~isnan(x(:))))); n = numel(x);
if n == 0, q = nan(size(p)); return; end
k = min(max(round(p/100*(n-1)) + 1, 1), n);
q = reshape(x(k), size(p));
end

function safe_run(fn, label)
try
    fn();
catch ME
    fprintf(2, '\n!! Step "%s" failed: %s\n', label, ME.message);
    for s = 1:numel(ME.stack)
        fprintf(2, '     in %s (line %d)\n', ME.stack(s).name, ME.stack(s).line);
    end
end
end

% =========================================================================
% UTILITIES
% =========================================================================
function W = hann2(h, w)
W = (0.5*(1 - cos(2*pi*(0:h-1)'/max(h-1,1)))) * ...
    (0.5*(1 - cos(2*pi*(0:w-1)/max(w-1,1))));
end

function files = list_images(folder)
% Images in the folder and all subfolders
exts = {'*.jpg','*.jpeg','*.jfif','*.png','*.tif','*.tiff','*.bmp','*.webp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(folder, '**', exts{i}))]; %#ok<AGROW>
end
if isempty(files), return; end
files = files(~[files.isdir]);
full = fullfile({files.folder}, {files.name});
[~, keep] = unique(lower(full), 'stable');
files = files(keep);
end
