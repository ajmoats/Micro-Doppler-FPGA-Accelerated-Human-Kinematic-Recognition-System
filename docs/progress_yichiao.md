# MIES Project Progress Log — YiChiao

## Project
Micro-Doppler FPGA-Accelerated Human Kinematic Recognition System

## Current Goal
First reproduce a working software baseline in Python, then compare settings and decide what can be improved or simplified before thinking about FPGA acceleration.

## What I finished today

### 1. Environment setup
- Created a new conda environment: `mies_yichiao`
- Installed the main packages needed for the PyTorch test version
- Fixed the NumPy compatibility issue by downgrading NumPy to a 1.x version

### 2. Safe test setup
To avoid modifying the original files, I created separate test files:
- `LSTM/data_loading_yichiao.py`
- `LSTM/JhummaLstm_yichiao.py`
- `LSTM/test_run_yichiao.py`

This lets me test the pipeline without touching the original repo files.

### 3. Baseline smoke test ran successfully
I successfully ran a first training smoke test with:
- sensor: `US40`
- folds: `[0]`
- epochs: `1`
- batch size: `50`
- max length: `404`

## Current result

### Dataset summary
- train samples: `1648`
- valid samples: `412`
- number of classes: `16`

### Sequence length
- train min / mean / max: `22 / 117.86 / 400`
- valid min / mean / max: `22 / 117.90 / 404`

### Training result
- train loss: `2.2709`
- train accuracy: `0.2518`

### Validation result
- valid loss: `1.6507`
- valid accuracy: `0.4515`

## Interpretation
This means the baseline pipeline is now running successfully.

Even with only 1 epoch, the validation accuracy already reached about `45.15%`, which is much higher than random guessing for 16 classes. So the model is learning meaningful information from the data.

## Important notes
The original LSTM code and the current data loader are not fully matched:
- `JhummaLstm.py` is older Keras/TensorFlow-style code and expects the older loader format. :contentReference[oaicite:0]{index=0}
- The current `data_loading.py` is already updated for a 16-class setup and returns single-sensor data, labels, and batch IDs. :contentReference[oaicite:1]{index=1}

Because of this, I used my own `yichiao` version for the first working baseline test.

## Next steps
1. Run the same setting with more epochs
   - try `5` epochs first
   - then try `10` epochs

2. Compare different sensor settings
   - `US25`
   - `US33`
   - `US40`
   - `all`

3. After choosing a better setting, run full 5-fold evaluation

4. Based on the baseline results, decide what to improve next
   - model settings
   - preprocessing
   - sensor combination
   - possible FPGA-friendly simplification later

## Status
- Environment setup: done
- First smoke test: done
- Software baseline reproduction: started successfully
- Parameter comparison: not started yet
- FPGA-related work: not started yet


## Update: 5-epoch baseline run

### Setting
- sensor: `US40`
- folds: `[0]`
- epochs: `5`
- batch size: `50`
- max length: `404`
- dropout: `0.5`
- learning rate: `1e-3`

### Results by epoch
- Epoch 1: train acc = `0.2518`, valid acc = `0.4515`
- Epoch 2: train acc = `0.4654`, valid acc = `0.5461`
- Epoch 3: train acc = `0.5783`, valid acc = `0.6044`
- Epoch 4: train acc = `0.6317`, valid acc = `0.6893`
- Epoch 5: train acc = `0.6784`, valid acc = `0.6505`

### Current observation
The model is clearly learning useful information from the dataset.  
Validation accuracy improved steadily from epoch 1 to epoch 4, reaching a best value of `68.93%`, but then dropped slightly at epoch 5 to `65.05%`.

This suggests that:
- the baseline pipeline is working well,
- the model performance is already much better than random guessing,
- and slight overfitting may begin after around 4 epochs under the current setting.

### Current best result
- best validation accuracy: `68.93%`
- achieved at: `epoch 4`

### Next step
The next step is to compare sensor settings:
- `US25`
- `US33`
- `US40`
- `all`

using the same basic configuration, then decide which input setting is the strongest baseline.


## Sensor comparison

| Sensor | Epochs | Best Valid Acc | Best Epoch | Final Valid Acc | Notes |
|-------|--------|----------------|------------|-----------------|-------|
| US25  | 5      | 78.40%         | 5          | 78.40%          | best single sensor |
| US33  | 5      | 75.73%         | 5          | 75.73%          | second best single sensor |
| US40  | 5      | 68.93%         | 4          | 65.05%          | slight overfitting after epoch 4 |
| all   | 5      | 97.82%         | 4 or 5     | 97.82%          | strongest overall result |



## 5-fold result with all sensors

### Setting
- sensor: `all` (`US25 + US33 + US40`)
- model: `1-layer LSTM`
- hidden size: `400`
- epochs: `5`
- folds: `5`
- batch size: `50`
- max length: `404`
- dropout: `0.5`
- learning rate: `1e-3`

### Final validation accuracy by fold
- Fold 0: `97.82%`
- Fold 1: `96.84%`
- Fold 2: `97.82%`
- Fold 3: `96.84%`
- Fold 4: `96.84%`

### Summary
- Mean final validation accuracy: `97.23%`
- Range: `96.84%` to `97.82%`

### Observation
Using all three sensors together gives much better performance than using only a single sensor.  
The 5-fold results are also very consistent, which suggests that the current baseline is stable and reliable.

### Current conclusion
The current best software baseline is:
- all sensors
- 1-layer LSTM with 400 hidden units
- 5 epochs
- 5-fold cross-validation

This will be used as the main baseline before trying further improvements or FPGA-related simplifications.