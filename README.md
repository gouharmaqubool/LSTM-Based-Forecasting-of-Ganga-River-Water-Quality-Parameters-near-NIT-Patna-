# LSTM-Based Forecasting of Ganga River Water Quality Parameters near NIT Patna

## Overview
This project uses Long Short-Term Memory (LSTM) neural networks to forecast water quality parameters near the NIT Patna region, specifically at the Gandhi Ghat and Gulabi Ghat sites along the Ganga River. The model incorporates features such as seasonality awareness, auto-validation, and reporting with visualization.

## Features
- **Seasonality Awareness**: Incorporates seasonal factors (e.g., monthly variations) for more accurate forecasting.
- **Multi-site, Multi-parameter Processing**: Processes various water quality parameters for multiple sites.
- **Performance Metrics**: Calculates R², RMSE for training, testing, and overall fit.
- **Visualization Outputs**: Generates insightful plots for historical fit, future forecasts, residuals, and accuracy scatterplots.
- **Automated Reporting**: Saves forecasts, summaries, and performance metrics in well-structured CSV files.

## Files Generated
- **Forecast CSV Files**: Monthly forecasts for each parameter and site.
- **Performance Summary**: A summary CSV detailing R² and RMSE metrics for all processed parameters.
- **Visualization PNGs**: Dashboard-style plots for each parameter showing historical fit, forecasts, and residual analysis.

## Installation
1. Ensure you have MATLAB installed.
2. Place the dataset file (`6yr_data.xlsx`) in the same directory as the script.

## Input Requirements
- **Dataset Filename**: `6yr_data.xlsx`
- **Columns Expected**: 
  - For site `Gandhi_Ghat`:
    - `Gandhi_Ghat_pH`
    - `Gandhi_Ghat_Conductivity`
    - `Gandhi_Ghat_Chlorides`
  - For site `Gulabi_Ghat`:
    - `Gulabi_Ghat_pH`
    - `Gulabi_Ghat_Conductivity`
    - `Gulabi_Ghat_Chlorides`
  - `Month`: Dates in `YYYY-MM` format.

## How It Works
1. **Preprocessing**: 
    - Normalizes the dataset by removing seasonality effects and scaling values.
    - Handles missing data using linear interpolation.
2. **Sequence Generation**: 
    - Creates input sequences for a lookback window of 6 months.
3. **LSTM Network**:
    - Uses 2-channel input (parameter values and months).
    - Predicts water quality parameters for a horizon of 12 months.
4. **Model Training**: 
    - Fits the network using the Adam optimizer with 80 hidden units and a dropout layer.
5. **Metrics Evaluation**: 
    - Calculates R² and RMSE for training and test predictions.
6. **Visualization and Reporting**:
    - Saves CSVs, including a summary file.
    - Generates plots for historical trends, forecasts, residuals, and scatterplots.

## Configuration Parameters
- **Lookback Window**: 6 months
- **Forecast Horizon**: 12 months
- **Hidden Units**: 80
- **Max Epochs**: 250
- **Test Size**: Last 12 months used for testing

## Visualization Outputs
1. **Historical Fit**:
   - Compares actual vs predicted values for the entire timeline.
   - Highlights the training/test split point.
2. **Future Forecast Plot**:
   - Shows 12-month forecasts for each parameter in 2025.
3. **Residuals Analysis**:
   - Histogram of errors to evaluate the distribution of residuals.
4. **Accuracy Scatterplot**:
   - Displays predicted vs actual values for training and test sets.

## Example Output Summary
| Site         | Parameter     | R²_Train | R²_Test | R²_Overall | RMSE_Train | RMSE_Test | RMSE_Overall | Avg_Forecast_2025 |
|--------------|---------------|----------|---------|------------|------------|-----------|--------------|------------------|
| Gandhi Ghat  | pH            | 0.92     | 0.86    | 0.90       | 0.15       | 0.20      | 0.17         | 7.3              |
| Gulabi Ghat  | Conductivity  | 0.88     | 0.83    | 0.86       | 0.30       | 0.40      | 0.35         | 245.6            |

## Usage Instructions
1. Run the script `LSTM_1.m` from MATLAB.
2. Ensure that the dataset `6yr_data.xlsx` is available in the working directory.
3. Wait for the script to output forecasted data and visualizations automatically.

## Error Handling
- If the data file is missing or incorrectly formatted, the script will throw an error message:
  ```
  Error: ensure "6yr_data.xlsx" is in the folder.
  ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- MATLAB documentation for `trainNetwork` and `trainingOptions`.
- Research on improved LSTM applications for time series forecasting.

## Future Work
- Incorporate additional site parameters.
- Support for alternate datasets and formats (e.g., CSV or API input).
- Introduce hyperparameter optimization for better network tuning.
