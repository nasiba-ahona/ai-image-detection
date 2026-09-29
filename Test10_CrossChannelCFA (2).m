function aigi_crosschannel_detector
% AIGI_CROSSCHANNEL_DETECTOR
% Forensic analyzer evaluating high-frequency cross-channel correlations 
% and 2x2 Bayer CFA phase imbalances. 
% Differentiates physical demosaicing algorithms from generative synthesis.

clc; close all;

fprintf('=================================================================\n');
fprintf('     TEST 10: CROSS-CHANNEL & CFA PHASE IMBALANCE ANALYZER\n');
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
        return;
end
end

% =========================================================================
% MODE 1: SINGLE PAIR RUNNER
% =========================================================================
function run_single_pair()
[realFile, realPath] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'}, 'Select ONE REAL image');
if isequal(realFile,0), disp('Cancelled.'); return; end
[aiFile, aiPath] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp','Images'}, 'Select ONE AI image');
if isequal(aiFile,0), disp('Cancelled.'); return; end

fprintf('\nProcessing REAL image: %s...\n', realFile);
realImg = im2double(imread(fullfile(realPath, realFile)));
R = compute_cross_channel_features(realImg);

fprintf('Processing AI image:   %s...\n', aiFile);
aiImg = im2double(imread(fullfile(aiPath, aiFile)));
A = compute_cross_channel_features(aiImg);

% Diagnostic Metrics Table
fprintf('\n----------------------- FORENSIC METRICS -----------------------\n');
fprintf('%-24s | %-14s | %-14s\n', 'Metric', 'REAL Image', 'AI Image');
fprintf('----------------------------------------------------------------\n');
fprintf('%-24s | %-14.4f | %-14.4f\n', '1. Red-Green Corr', R.corrRG, A.corrRG);
fprintf('%-24s | %-14.4f | %-14.4f\n', '2. Green-Blue Corr', R.corrGB, A.corrGB);
fprintf('%-24s | %-14.4f | %-14.4f\n', '3. Red-Blue Corr', R.corrRB, A.corrRB);
fprintf('%-24s | %-14.3f | %-14.3f\n', '4. Green Energy Ratio', R.Gratio, A.Gratio);
fprintf('%-24s | %-14.4f | %-14.4f\n', '5. Green Phase Imbalance', R.phaseG, A.phaseG);
fprintf('%-24s | %-14.4f | %-14.4f\n', '6. R/B Phase Imbalance', R.phaseRB, A.phaseRB);
fprintf('----------------------------------------------------------------\n');

% Automated Diagnostic Interpretation
fprintf('\nForensic Assessment:\n');

% Real cameras have high cross-correlation due to demosaicing overlap
% and high phase imbalance due to the Bayer mosaic pattern.
r_is_ai = (R.corrRG < 0.25) && (R.phaseG < 0.04);
a_is_ai = (A.corrRG < 0.25) && (A.phaseG < 0.04);

fprintf('REAL Image: ');
if r_is_ai
    fprintf('SUSPICIOUS. Lacks demosaicing correlation and phase asymmetry.\n');
else
    fprintf('AUTHENTIC. High cross-channel correlation and physical Bayer asymmetry present.\n');
end

fprintf('AI Image  : ');
if a_is_ai
    fprintf('SYNTHETIC SIGNATURE. Channels are generated independently without a Bayer grid.\n');
else
    fprintf('INCONCLUSIVE. Channels exhibit correlated noise (Possible algorithmic artifact or compression).\n');
end
fprintf('================================================================\n\n');

% Visualizations
% Figure 1: High-Frequency Residual Maps (Showing the structural noise)
figure('Name', 'High-Frequency Green Channel Residuals', 'Position', [100 200 1000 450]);

subplot(1,2,1);
imagesc(abs(R.resG)); axis image; colormap gray; colorbar;
title(sprintf('REAL: Green HF Residuals\nPhase Imbalance: %.4f', R.phaseG), 'Interpreter', 'none');

subplot(1,2,2);
imagesc(abs(A.resG)); axis image; colormap gray; colorbar;
title(sprintf('AI: Green HF Residuals\nPhase Imbalance: %.4f', A.phaseG), 'Interpreter', 'none');

% Figure 2: Metric Comparison Bars
figure('Name', 'Cross-Channel & Phase Metrics', 'Position', [150 150 800 400]);

subplot(1,2,1);
bar([R.corrRG, R.corrGB, R.corrRB; A.corrRG, A.corrGB, A.corrRB]');
set(gca, 'XTickLabel', {'R-G', 'G-B', 'R-B'});
ylabel('Pearson Correlation (r)');
title('Cross-Channel Noise Correlation');
legend('REAL', 'AI'); grid on;

subplot(1,2,2);
bar([R.phaseG, R.phaseRB; A.phaseG, A.phaseRB]');
set(gca, 'XTickLabel', {'Green Phase', 'Red/Blue Phase'});
ylabel('Phase Imbalance Metric');
title('Bayer CFA Phase Asymmetry');
legend('REAL', 'AI'); grid on;
end

% =========================================================================
% MODE 2: BATCH FOLDER RUNNER
% =========================================================================
function run_batch_folders()
realDir = uigetdir(pwd, 'Select Folder Containing REAL Images');
if isequal(realDir,0), disp('Cancelled.'); return; end
aiDir = uigetdir(pwd, 'Select Folder Containing AI Images');
if isequal(aiDir,0), disp('Cancelled.'); return; end

realFiles = list_images(realDir);
aiFiles   = list_images(aiDir);

if isempty(realFiles) || isempty(aiFiles)
    error('One or both directories contain no supported images.');
end

fprintf('\nAnalyzing REAL folder (%d images)...\n', numel(realFiles));
resReal = process_file_list(realDir, realFiles);

fprintf('\nAnalyzing AI folder (%d images)...\n', numel(aiFiles));
resAI = process_file_list(aiDir, aiFiles);

% Print Summary Report
fprintf('\n=================================================================\n');
fprintf('                   BATCH CROSS-CHANNEL REPORT                    \n');
fprintf('=================================================================\n');
print_batch_stats('REAL Folder', resReal);
print_batch_stats('AI Folder  ', resAI);
fprintf('=================================================================\n');

% Figure 3: Scatter Plot in Feature Hyperspace 
figure('Name', 'Feature Hyperspace: CFA Physics vs Generative Independence', 'Position', [100 100 850 550]);
scatter([resReal.corrRG], [resReal.phaseG], 40, 'b', 'filled', 'DisplayName', 'REAL Images');
hold on;
scatter([resAI.corrRG], [resAI.phaseG], 40, 'r', 'filled', 'DisplayName', 'AI Images');

% Physical Decision Guardrails
xline(0.25, '--k', 'RG Correlation Threshold', 'LineWidth', 1.3, 'LabelOrientation', 'horizontal');
yline(0.04, '--k', 'Green Phase Imbalance Threshold', 'LineWidth', 1.3);

xlabel('Red-Green High Frequency Correlation (r)');
ylabel('Green Channel 2x2 Phase Imbalance');
title('Hyperspace Separation: Demosaicing (Real) vs Independent Synthesis (AI)');
legend('Location', 'best');
grid on; hold off;
end

function res = process_file_list(baseDir, files)
n = numel(files);
res = repmat(struct('name','','corrRG',0,'phaseG',0,'verdict',''), n, 1);
t0 = tic;
for i = 1:n
    try
        filePath = fullfile(baseDir, files(i).name);
        img = im2double(imread(filePath));
        out = compute_cross_channel_features(img);
        
        res(i).name   = files(i).name;
        res(i).corrRG = out.corrRG;
        res(i).phaseG = out.phaseG;
        
        % Basic rule: Real cameras have coupled channels and asymmetric phases
        if (out.corrRG < 0.25) && (out.phaseG < 0.04)
            res(i).verdict = 'SYNTHETIC';
        else
            res(i).verdict = 'AUTHENTIC';
        end
    catch
        res(i).name    = files(i).name;
        res(i).verdict = 'ERROR';
    end
    
    if mod(i, 20) == 0
        fprintf('  Processed %d/%d images (%.1f s elapsed)\n', i, n, toc(t0));
    end
end
end

function print_batch_stats(label, res)
n = numel(res);
nAuth  = sum(strcmp({res.verdict}, 'AUTHENTIC'));
nSynth = sum(strcmp({res.verdict}, 'SYNTHETIC'));
nErr   = sum(strcmp({res.verdict}, 'ERROR'));

fprintf('%s (N = %d):\n', label, n);
fprintf('  * Demosaiced Camera Signature : %4d (%5.1f%%)\n', nAuth, 100*nAuth/n);
fprintf('  * Independent Gen-AI Channels : %4d (%5.1f%%)\n', nSynth, 100*nSynth/n);
if nErr > 0
    fprintf('  * Image Read Errors           : %4d (%5.1f%%)\n', nErr, 100*nErr/n);
end
fprintf('\n');
end

% =========================================================================
% CORE FORENSIC ENGINE: CROSS-CHANNEL & CFA PHASE
% =========================================================================
function out = compute_cross_channel_features(rgb)
if ndims(rgb) == 2, rgb = repmat(rgb, [1 1 3]); end
if size(rgb,3) > 3, rgb = rgb(:,:,1:3); end

% 1. Extract High-Frequency Residuals (Noise & Texture)
res = zeros(size(rgb)); 
for k = 1:3
    res(:,:,k) = rgb(:,:,k) - medfilt2(rgb(:,:,k), [3 3], 'symmetric'); 
end
rR = res(:,:,1); 
rG = res(:,:,2); 
rB = res(:,:,3); 

out.resG = rG; % Store for visualization

% 2. Cross-Channel Correlation
out.corrRG = safecorr(rR, rG);
out.corrGB = safecorr(rG, rB);
out.corrRB = safecorr(rR, rB);

% 3. Energy Ratio (Green vs Red/Blue)
eR = mean(rR(:).^2); 
eG = mean(rG(:).^2); 
eB = mean(rB(:).^2); 
out.Gratio = 10 * log10((eG + eps) / (0.5 * (eR + eB) + eps));

% 4. CFA Phase Imbalance (Quincunx Lattice Asymmetry)
K = [0 .25 0; .25 0 .25; 0 .25 0]; 
out.phaseG  = phase_imbalance(rgb(:,:,2), K); 
out.phaseRB = 0.5 * (phase_imbalance(rgb(:,:,1), K) + phase_imbalance(rgb(:,:,3), K));
end

% =========================================================================
% MATHEMATICAL SUBROUTINES
% =========================================================================
function c = safecorr(a, b)
% Safely computes Pearson correlation, capping max samples for speed
a = a(:); b = b(:); 
if numel(a) > 200000
    s = round(linspace(1, numel(a), 200000)); 
    a = a(s); b = b(s); 
end
if std(a) < eps || std(b) < eps
    c = NaN; return; 
end
R = corrcoef(a, b); 
c = R(1,2);
end

function v = phase_imbalance(x, K)
% Measures interpolation asymmetry across the four 2x2 Bayer sub-phases
e = abs(x - conv2(x, K, 'same')); 
e = e(3:end-2, 3:end-2); 
[h, w] = size(e); 
h = 2 * floor(h/2); 
w = 2 * floor(w/2); 
e = e(1:h, 1:w);

% Mean interpolation error for each of the 4 pixel phases
p = [mean(mean(e(1:2:end, 1:2:end))), ...
     mean(mean(e(1:2:end, 2:2:end))), ...
     mean(mean(e(2:2:end, 1:2:end))), ...
     mean(mean(e(2:2:end, 2:2:end)))];

v = (max(p) - min(p)) / (max(p) + min(p) + eps);
end

function files = list_images(folder)
exts = {'*.jpg','*.jpeg','*.png','*.tif','*.tiff','*.bmp','*.webp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(folder, exts{i}))]; %#ok<AGROW>
end
if ~isempty(files), files = files(~[files.isdir]); end
end
