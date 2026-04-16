# MIES Project Progress Log — YiChiao

## Project
Micro-Doppler FPGA-Accelerated Human Kinematic Recognition System

## Current Objective
The current goal is to first reproduce a stable software baseline in Python, then compare model settings and sensor inputs before deciding what parts are worth improving or simplifying for possible FPGA-related work later.

---

## 1. Environment and Setup

### Environment
- Created a new conda environment: `mies_yichiao`
- Installed the main packages needed for the PyTorch test version
- Fixed the NumPy compatibility issue by downgrading NumPy to a 1.x version

### Safe test setup
To avoid modifying the original repository files, I created separate test versions:
- `LSTM/data_loading_yichiao.py`
- `LSTM/JhummaLstm_yichiao.py`
- `LSTM/test_run_yichiao.py`

This allows me to test the training pipeline safely without changing the original code.

---

## 2. Why a Separate Test Version Was Needed

The original LSTM training code and the current data loader are not fully matched.

- The original `JhummaLstm.py` is based on older Keras/TensorFlow-style code and expects an older loader format.
- The current `data_loading.py` is already updated for a 16-class setup and returns single-sensor data, labels, and batch IDs.

Because of this mismatch, I created a separate PyTorch-based `yichiao` version in order to get a working baseline first.

---

## 3. Initial Smoke Test

### Setting
- sensor: `US40`
- folds: `[0]`
- epochs: `1`
- batch size: `50`
- max length: `404`

### Dataset summary
- train samples: `1648`
- valid samples: `412`
- number of classes: `16`

### Sequence length
- train min / mean / max: `22 / 117.86 / 400`
- valid min / mean / max: `22 / 117.90 / 404`

### Result
- train loss: `2.2709`
- train accuracy: `0.2518`
- valid loss: `1.6507`
- valid accuracy: `0.4515`

### Interpretation
This confirmed that the pipeline was working correctly.

Even with only 1 epoch, the validation accuracy already reached `45.15%`, which is much higher than random guessing for a 16-class problem. This showed that the model was learning meaningful information from the data.

---

## 4. 5-Epoch Baseline Run on US40

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

### Observation
The model improved steadily from epoch 1 to epoch 4, reaching its best validation accuracy of `68.93%` at epoch 4. At epoch 5, the validation accuracy dropped slightly to `65.05%`, which may suggest mild overfitting under this setting.

### Current best result for US40
- best validation accuracy: `68.93%`
- achieved at: `epoch 4`

---

## 5. Sensor Comparison

To determine which input setting provides the strongest baseline, I compared the following options under the same configuration:
- `US25`
- `US33`
- `US40`
- `all`

### Common setting
- epochs: `5`
- folds: `[0]`
- hidden size: `400`
- batch size: `50`
- max length: `404`
- dropout: `0.5`
- learning rate: `1e-3`

### Results

| Sensor | Epochs | Best Valid Acc | Best Epoch | Final Valid Acc | Notes |
|--------|--------|----------------|------------|-----------------|-------|
| US25   | 5      | 78.40%         | 5          | 78.40%          | Best single sensor |
| US33   | 5      | 75.73%         | 5          | 75.73%          | Second best single sensor |
| US40   | 5      | 68.93%         | 4          | 65.05%          | Slight overfitting after epoch 4 |
| all    | 5      | 97.82%         | 4 or 5     | 97.82%          | Strongest overall result |

### Interpretation
Using all three sensors together performs much better than using any single sensor alone. This strongly suggests that combining the three sensor inputs provides much richer and more discriminative information for classification.

---

## 6. 5-Fold Baseline with All Sensors

Since `all` gave the best result in the sensor comparison, I used it for a more formal 5-fold evaluation.

### Setting
- sensor: `all` (`US25 + US33 + US40`)
- model: `1-layer LSTM`
- hidden size: `400`
- epochs: `5`
- folds: `[0, 1, 2, 3, 4]`
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
- mean final validation accuracy: `97.23%`
- range: `96.84%` to `97.82%`

### Interpretation
The 5-fold results are very consistent, which suggests that the current baseline is both stable and reliable. At this point, the strongest software baseline is clearly the all-sensor configuration.

---

## 7. Current Baseline Conclusion

The current best software baseline is:

- input: all sensors (`US25 + US33 + US40`)
- model: 1-layer LSTM
- hidden size: `400`
- epochs: `5`
- evaluation: 5-fold cross-validation
- mean final validation accuracy: `97.23%`

This is now the main baseline that can be used for future comparison.

---

## 8. Next Steps

Possible next steps include:

1. Save and organize the current baseline results more formally
2. Add confusion matrix or per-class analysis
3. Try smaller or simpler models to see whether similar performance can be achieved with lower complexity
4. Compare whether fewer epochs or reduced input settings still give strong accuracy
5. Start thinking about which parts of the pipeline may be more FPGA-friendly later

---

## 9. Current Status

- Environment setup: done
- Safe test version (`yichiao` files): done
- Initial smoke test: done
- 5-epoch single-sensor baseline: done
- Sensor comparison: done
- 5-fold all-sensor baseline: done
- Model simplification experiments: not started yet
- FPGA-related work: not started yet