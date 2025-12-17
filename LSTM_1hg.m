%% LSTM Water Quality Forecasting - Production Ready + Rolling Backward Evaluation
clear all; close all; clc;

%% Configuration
LOOKBACK_WINDOW = 6;     
FORECAST_HORIZON = 12;   
HIDDEN_UNITS = 64;       
MAX_EPOCHS = 200;        
TEST_SIZE = 12;          % 12 months test for final model
MAX_SPLITS = 3;          % up to 3 rolling-backward splits

fprintf('=== SEASONALITY-AWARE LSTM FORECASTING (with rolling-backward eval) ===\n');

%% Load Data
try
    data = readtable('6yr_data.xlsx');
    dates = datetime(data.Month);
    month_indices = month(dates); % Extract month numbers (1-12)
    fprintf('Data Loaded: %d months (%s to %s)\n', length(dates), datestr(dates(1)), datestr(dates(end)));
catch
    error('Error: ensure "6yr_data.xlsx" is in the folder and has a "Month" column.');
end

siteParams = {
    'Gandhi_Ghat', 'TDS', 'Gandhi_Ghat_TDS';
    'Gandhi_Ghat', 'Conductivity', 'Gandhi_Ghat_Conductivity';
    'Gandhi_Ghat', 'Chlorides', 'Gandhi_Ghat_Chlorides';
    'Gulabi_Ghat', 'TDS', 'Gulabi_Ghat_TDS';
    'Gulabi_Ghat', 'Conductivity', 'Gulabi_Ghat_Conductivity';
    'Gulabi_Ghat', 'Chlorides', 'Gulabi_Ghat_Chlorides'
};

% Storage for Summary CSV
summaryStats = {}; 

%% Main Loop
for i = 1:size(siteParams, 1)
    site = siteParams{i,1};
    param = siteParams{i,2};
    colName = siteParams{i,3};
    
    fprintf('\n[%d/%d] Processing %s...\n', i, size(siteParams,1), colName);
    
    raw_vals = data.(colName);
    if any(isnan(raw_vals)), raw_vals = fillmissing(raw_vals, 'linear'); end
    
    % --- 1. Preprocessing ---
    mu = mean(raw_vals);
    sig = std(raw_vals);
    valsNorm = (raw_vals - mu) / sig;
      
    monthsNorm = 2*pi*(month_indices/12);     % 0..2pi (seasonality angle)
    
    % --- 2. Create ALL sequences once ---
    total_samples = length(valsNorm) - LOOKBACK_WINDOW;
    if total_samples <= TEST_SIZE + 1
        warning('Not enough samples for %s. Skipping.', colName);
        continue;
    end
    
    XList = cell(total_samples,1);
    YList = zeros(total_samples,1);
    
    for t = 1 : total_samples
        seq_v = valsNorm(t : t + LOOKBACK_WINDOW - 1);
        seq_m = monthsNorm(t : t + LOOKBACK_WINDOW - 1);
        XList{t,1} = [seq_v'; seq_m']; % 2 features x seqLen
        YList(t,1) = valsNorm(t + LOOKBACK_WINDOW);
    end
    
    %% --------- A) ROLLING-BACKWARD EVALUATION (NO PLOTS) ---------
    all_R2_Test   = [];
    all_RMSE_Test = [];
    
    BLOCK = TEST_SIZE;
    maxPossibleSplits = floor(total_samples / BLOCK) - 1;  % ensure training set non-empty
    nSplits = min(MAX_SPLITS, maxPossibleSplits);
    
    for s = 1:nSplits
        % Test block is a 12-month chunk from the end, shifting backward
        test_end   = total_samples - (s-1)*BLOCK;
        test_start = test_end - BLOCK + 1;
        if test_start <= 1
            continue;
        end
        
        XTrain_s = XList(1:test_start-1);
        YTrain_s = YList(1:test_start-1);
        
        XTest_s  = XList(test_start:test_end);
        YTest_s  = YList(test_start:test_end);
        
        % LSTM Architecture
        layers = [
            sequenceInputLayer(2)
            lstmLayer(HIDDEN_UNITS, 'OutputMode','last')
            dropoutLayer(0.2)
            fullyConnectedLayer(1)
            regressionLayer];
        
        options = trainingOptions('adam', ...
            'MaxEpochs', MAX_EPOCHS, ...
            'GradientThreshold', 1, ...
            'InitialLearnRate', 0.005, ...
            'Shuffle','never', ...
            'Verbose', 0, 'Plots', 'none');
        
        net_s = trainNetwork(XTrain_s, YTrain_s, layers, options);
        
        % Predict & denormalize
        YPred_Test_s = predict(net_s, XTest_s);
        YTest_Real_s      = YTest_s      * sig + mu;
        YPred_Test_Real_s = YPred_Test_s * sig + mu;
        
        calcR2   = @(y,y_hat) 1 - sum((y - y_hat).^2) / sum((y - mean(y)).^2);
        calcRMSE = @(y,y_hat) sqrt(mean((y - y_hat).^2));
        
        R2_Test_s   = calcR2(YTest_Real_s, YPred_Test_Real_s);
        RMSE_Test_s = calcRMSE(YTest_Real_s, YPred_Test_Real_s);
        
        all_R2_Test(end+1)   = R2_Test_s;
        all_RMSE_Test(end+1) = RMSE_Test_s;
        
        approxYear = year(dates(LOOKBACK_WINDOW + test_start));
        fprintf(' Split %d: Test period approx year %d, R2=%.3f, RMSE=%.3f\n', ...
            s, approxYear, R2_Test_s, RMSE_Test_s);
    end
    
    if ~isempty(all_R2_Test)
        avg_R2_Test   = mean(all_R2_Test);
        avg_RMSE_Test = mean(all_RMSE_Test);
        fprintf(' Rolling-backward average over %d splits: R2=%.3f, RMSE=%.3f\n', ...
            numel(all_R2_Test), avg_R2_Test, avg_RMSE_Test);
    else
        fprintf(' Rolling-backward evaluation not possible (too few splits).\n');
    end
    
    %% --------- B) FINAL MODEL (LAST 12 MONTHS TEST) + PLOTS ---------
    % Standard chronological split: last TEST_SIZE samples = test (≈ 2024)
    train_len = total_samples - TEST_SIZE;
    
    XTrain = XList(1:train_len);
    YTrain = YList(1:train_len);
    XTest  = XList(train_len+1:end);
    YTest  = YList(train_len+1:end);
    
    % Re-create layers & train final model
    layers = [
        sequenceInputLayer(2)                   
        lstmLayer(HIDDEN_UNITS, 'OutputMode','last')
        dropoutLayer(0.2)                       
        fullyConnectedLayer(1)
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', MAX_EPOCHS, ...
        'GradientThreshold', 1, ...
        'InitialLearnRate', 0.005, ...
        'Shuffle','never', ...
        'Verbose', 0, 'Plots', 'none');
    
    net = trainNetwork(XTrain, YTrain, layers, options);
    
    % --- Evaluation for final model ---
    YPred_Train = predict(net, XTrain);
    YPred_Test  = predict(net, XTest);
    
    % Denormalize for metrics
    YTrain_Real      = YTrain      * sig + mu;
    YPred_Train_Real = YPred_Train * sig + mu;
    YTest_Real       = YTest       * sig + mu;
    YPred_Test_Real  = YPred_Test  * sig + mu;
    
    % Metrics (final split)
    calcR2   = @(y,y_hat) 1 - sum((y - y_hat).^2) / sum((y - mean(y)).^2);
    calcRMSE = @(y,y_hat) sqrt(mean((y - y_hat).^2));
    
    stats.R2_Train   = calcR2(YTrain_Real, YPred_Train_Real);
    stats.RMSE_Train = calcRMSE(YTrain_Real, YPred_Train_Real);
    stats.R2_Test    = calcR2(YTest_Real,  YPred_Test_Real);
    stats.RMSE_Test  = calcRMSE(YTest_Real, YPred_Test_Real);
    
    % Combine for "Overall Fit" (reconstructing whole timeline)
    Y_All_Real = [YTrain_Real; YTest_Real];            % length = total_samples
    Y_Pred_All = [YPred_Train_Real; YPred_Test_Real];
    stats.R2_Overall   = calcR2(Y_All_Real, Y_Pred_All);
    stats.RMSE_Overall = calcRMSE(Y_All_Real, Y_Pred_All);
    
    %% --- 6. Forecasting 2025 with final model ---
    curr_vals   = valsNorm(end-LOOKBACK_WINDOW+1 : end)';    % last 6 months
    curr_months = monthsNorm(end-LOOKBACK_WINDOW+1 : end)';  % last 6 months
    future_month_indices = (1:FORECAST_HORIZON)' / 12;       % simple month feature
    forecasts_norm = [];
    
    for step = 1:FORECAST_HORIZON
        input_feat = [curr_vals; curr_months];   % 2 x LOOKBACK_WINDOW
        next_val   = predict(net, input_feat);
        forecasts_norm = [forecasts_norm; next_val];
        curr_vals   = [curr_vals(2:end),   next_val];
        curr_months = [curr_months(2:end), future_month_indices(step)];
    end
    final_forecast = (forecasts_norm * sig) + mu;
    
    % A. Save Forecast CSV
    future_dates = (dates(end) + calmonths(1:FORECAST_HORIZON))';
    t_forecast = table(future_dates, final_forecast, ...
        'VariableNames', {'Date', [colName '_Forecast']});
    writetable(t_forecast, sprintf('%s_Forecast_2025.csv', colName));
    
    % B. Store Summary Stats (from final split)
    summaryStats{end+1, 1} = site;
    summaryStats{end,   2} = param;
    summaryStats{end,   3} = stats.R2_Train;
    summaryStats{end,   4} = stats.R2_Test;
    summaryStats{end,   5} = stats.R2_Overall;
    summaryStats{end,   6} = stats.RMSE_Train;
    summaryStats{end,   7} = stats.RMSE_Test;
    summaryStats{end,   8} = stats.RMSE_Overall;
    summaryStats{end,   9} = mean(final_forecast);
    
    %% --- 8. VISUALIZATION (Dashboard) for final model ---
    figure('Position', [100, 100, 1200, 800], 'Color', 'w');
    
    % Subplot 1: Historical Fit (now vectors have matching size)
    subplot(2,2,1);
    plot(dates(LOOKBACK_WINDOW+1:end), Y_All_Real, 'b-', 'LineWidth', 1.5); hold on;
    plot(dates(LOOKBACK_WINDOW+1:end), Y_Pred_All, 'r--', 'LineWidth', 1.5);
    xline(dates(end-TEST_SIZE), 'k-', 'Test Split'); % Show where training ended
    title('Historical Data vs Model Fit');
    legend('Actual', 'LSTM Fitted', 'Train/Test Split', 'Location','best');
    grid on; ylabel(param);
    
    % Subplot 2: 2025 Forecast
    subplot(2,2,2);
    plot(future_dates, final_forecast, 'g-o', 'LineWidth', 2, 'MarkerFaceColor','g');
    title(['2025 Forecast: ' param]);
    xtickangle(45); grid on; ylabel(param);
    
    % Subplot 3: Scatter (Train vs Test) with 1:1 line
    subplot(2,2,3);
    scatter(YTrain_Real, YPred_Train_Real, 30, 'b', 'filled', 'DisplayName', 'Train'); hold on;
    scatter(YTest_Real,  YPred_Test_Real,  50, 'r', 'filled', 'DisplayName', 'Test');
    allVals = [YTrain_Real; YTest_Real; YPred_Train_Real; YPred_Test_Real];
    minv = min(allVals);
    maxv = max(allVals);
    plot([minv maxv], [minv maxv], 'k--', 'LineWidth',1.5, 'DisplayName','1:1 line');
    hold off;
    title(sprintf('Accuracy: R^2 Overall = %.3f', stats.R2_Overall));
    xlabel('Actual'); ylabel('Predicted'); legend('Location','best'); grid on;
    
    % Subplot 4: Residuals
    subplot(2,2,4);
    residuals = Y_All_Real - Y_Pred_All;
    histogram(residuals, 10, 'FaceColor', [0.7 0.7 0.7]);
    title(['Residuals (Error Distribution) RMSE=' num2str(stats.RMSE_Overall, '%.3f')]);
    xlabel('Error Magnitude'); grid on;
    
    sgtitle([colName ' - LSTM Deep Learning Analysis']);
    saveas(gcf, sprintf('%s_LSTM_Analysis.png', colName));
    close(gcf);
end

%% Save Final Summary
T_Summary = cell2table(summaryStats, ...
    'VariableNames', {'Site', 'Parameter', 'R2_Train', 'R2_Test', 'R2_Overall', ...
    'RMSE_Train', 'RMSE_Test', 'RMSE_Overall', 'Avg_Forecast_2025'});

writetable(T_Summary, 'LSTM_Performance_Summary.csv');
disp(T_Summary);
fprintf('\nAll processing complete. Files saved.\n');
