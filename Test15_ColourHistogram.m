function aigi_color_detector
% AIGI_COLOR_DETECTOR
% Forensic analyzer evaluating image histograms, color space utilization, 
% clipping behaviors, and quantization (comb) artifacts.
% Differentiates physical camera ISPs from generative neural rendering.

clc; close all;

fprintf('=================================================================\n');
fprintf('     TEST 15: COLOR HISTOGRAM & QUANTIZATION ANALYZER\n');
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
R = compute_color_features(realImg);

fprintf('Processing AI image:   %s...\n', aiFile);
aiImg = im2double(imread(fullfile(aiPath, aiFile)));
A = compute_color_features(aiImg);

% Diagnostic Metrics Table
fprintf('\n----------------------- FORENSIC METRICS -----------------------\n');
fprintf('%-24s | %-14s | %-14s\n', 'Metric', 'REAL Image', 'AI Image');
fprintf('----------------------------------------------------------------\n');
fprintf('%-24s | %-13.2f%%| %-13.2f%%\n', '1. Clipped Pixels (0/255)', R.co_clip*100, A.co_clip*100);
fprintf('%-24s | %-13.2f%%| %-13.2f%%\n', '2. Histogram Bin Usage', R.co_full*100, A.co_full*100);
fprintf('%-24s | %-14.4f | %-14.4f\n', '3. Quantization Comb FFT', R.co_comb, A.co_comb);
fprintf('%-24s | %-14.4f | %-14.4f\n', '4. Mean Saturation', R.co_satmean, A.co_satmean);
fprintf('%-24s | %-14.4f | %-14.4f\n', '5. Saturation Std Dev', R.co_satstd, A.co_satstd);
fprintf('----------------------------------------------------------------\n');

% Automated Diagnostic Interpretation
fprintf('\nForensic Assessment:\n');
% AI images often over-utilize the 8-bit color space (smooth histograms)
% while real camera ISPs leave quantization gaps (comb artifacts) due to JPEG/gamma.
fprintf('REAL Image: ');
if R.co_comb > 0.015
    fprintf('AUTHENTIC. High quantization comb detected (Consistent with camera ISP/JPEG saving).\n');
else
    fprintf('INCONCLUSIVE. Smooth histogram lacks typical camera quantization gaps.\n');
end

fprintf('AI Image  : ');
if A.co_full > 0.95 && A.co_comb < 0.01
    fprintf('SYNTHETIC SIGNATURE. "Too perfect" histogram. Over-utilizes color space without ISP quantization.\n');
else
    fprintf('INCONCLUSIVE. Histogram shows digital processing artifacts.\n');
end
fprintf('================================================================\n\n');

% Visualizations
% Figure 1: Luma Histograms (Reveals Quantization Gaps)
figure('Name', 'Luma Histogram & Quantization Comb', 'Position', [100 200 900 450]);
bins = 0:255;

subplot(1,2,1);
bar(bins, R.hy, 'FaceColor', [0.2 0.6 1.0], 'EdgeColor', 'none');
title(sprintf('REAL: Grayscale Histogram\nComb FFT Spike: %.4f', R.co_comb));
xlabel('Pixel Intensity (0-255)'); ylabel('Probability'); xlim([0 255]); grid on;

subplot(1,2,2);
bar(bins, A.hy, 'FaceColor', [1.0 0.4 0.2], 'EdgeColor', 'none');
title(sprintf('AI: Grayscale Histogram\nComb FFT Spike: %.4f', A.co_comb));
xlabel('Pixel Intensity (0-255)'); ylabel('Probability'); xlim([0 255]); grid on;

% Figure 2: Saturation Heatmaps
figure('Name', 'Spatial Saturation Distribution', 'Position', [150 150 900 450]);
subplot(1,2,1);
imagesc(R.sat_map); axis image; colormap hot; colorbar; clim([0 1]);
title(sprintf('REAL: Saturation Heatmap\nMean Saturation: %.3f', R.co_satmean));

subplot(1,2,2);
imagesc(A.sat_map); axis image; colormap hot; colorbar; clim([0 1]);
title(sprintf('AI: Saturation Heatmap\nMean Saturation: %.3f', A.co_satmean));
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

% Figure 3: Scatter Plot in Feature Hyperspace 
figure('Name', 'Feature Hyperspace: Histogram Utilization vs Saturation', 'Position', [100 100 850 550]);
scatter([resReal.co_satmean], [resReal.co_full]*100, 40, 'b', 'filled', 'DisplayName', 'REAL Images');
hold on;
scatter([resAI.co_satmean], [resAI.co_full]*100, 40, 'r', 'filled', 'DisplayName', 'AI Images');

% Physical Decision Guardrails
yline(95, '--k', '95% Bin Utilization (Smoothness Ceiling)', 'LineWidth', 1.3);

xlabel('Mean Image Saturation');
ylabel('Histogram Bin Utilization (%) - Avoidance of Quantization');
title('Hyperspace Separation: Camera ISP Limitations vs AI Color Rendering');
legend('Location', 'best');
grid on; hold off;
end

function res = process_file_list(baseDir, files)
n = numel(files);
res = repmat(struct('name','','co_full',0,'co_satmean',0,'co_comb',0), n, 1);
t0 = tic;
for i = 1:n
    try
        filePath = fullfile(baseDir, files(i).name);
        img = im2double(imread(filePath));
        out = compute_color_features(img);
        
        res(i).name       = files(i).name;
        res(i).co_full    = out.co_full;
        res(i).co_satmean = out.co_satmean;
        res(i).co_comb    = out.co_comb;
    catch
        res(i).name = files(i).name;
    end
    if mod(i, 20) == 0
        fprintf('  Processed %d/%d images (%.1f s elapsed)\n', i, n, toc(t0));
    end
end
end

% =========================================================================
% CORE FORENSIC ENGINE: COLOR & HISTOGRAM
% =========================================================================
function out = compute_color_features(rgb)
if ndims(rgb) == 2, rgb = repmat(rgb, [1 1 3]); end
if size(rgb,3) > 3, rgb = rgb(:,:,1:3); end

% Convert to 8-bit space for discrete histogram analysis
u = im2uint8(rgb); 

% 1. Clipping Mask (Pixels hitting exactly 0 or 255)
out.co_clip = mean(any(u == 0 | u == 255, 3), 'all');

% 2. Histogram Fullness (Bin Utilization)
full = 0;
for k = 1:3
    hk = histcounts(double(u(:,:,k)), -0.5:1:255.5); 
    full = full + mean(hk > 0); 
end
out.co_full = full / 3;

% 3. Quantization Comb Artifact (FFT of Grayscale Histogram)
y = double(im2uint8(rgb2gray(rgb))); 
hy = histcounts(y, -0.5:1:255.5); 
hy = hy / (sum(hy) + eps); 
out.hy = hy; % Stored for visualization

% Perform 1D FFT on the histogram to find periodic gaps (comb teeth)
Hf = abs(fft(hy - mean(hy))); 
out.co_comb = max(Hf(3:129)) / (sum(hy) + eps);

% 4. HSV Saturation Statistics
mx = max(rgb, [], 3); 
mn = min(rgb, [], 3); 
sat = (mx - mn) ./ (mx + eps); 

out.sat_map    = sat; % Stored for visualization
out.co_satmean = mean(sat(:));
out.co_satstd  = std(sat(:));
end

function files = list_images(folder)
exts = {'*.jpg','*.jpeg','*.png','*.tif','*.tiff','*.bmp','*.webp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(folder, exts{i}))]; %#ok<AGROW>
end
if ~isempty(files), files = files(~[files.isdir]); end
end
