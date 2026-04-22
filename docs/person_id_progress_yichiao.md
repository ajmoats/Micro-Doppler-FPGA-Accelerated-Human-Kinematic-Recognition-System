# Person ID Progress Log — YiChiao

## 0. Project Goal

The project direction has shifted from action recognition to **person identification** using offline ultrasonic recordings.

The current idea is:

- **Live demo side:** demonstrate how ultrasonic data is collected
- **Model side:** use recorded trial data for offline person identification

The machine learning task is now:

- **Input:** one ultrasonic action segment
- **Output:** which participant the segment belongs to

So the project is no longer focused on real-time prediction.  
Instead, it is:

1. showing how the ultrasonic system acquires data
2. using offline recordings to train and evaluate a person-identification model

---

## 1. Overall Workflow

This is the order I followed.

### Step 1
Inspect one participant `.mat` file in MATLAB to understand the structure.

### Step 2
Create a new person-identification data loader so that:
- the label becomes **person ID**
- the code scans all `*_1.mat` files
- empty samples are skipped automatically

### Step 3
Create a PyTorch LSTM training file for person identification.

### Step 4
Run a first baseline using:
- `US40`
- all actions `1~21`
- fold `0`
- `5` epochs

### Step 5
Improve the baseline by changing the sensor input to:
- `all` (`US25 + US33 + US40`)

### Step 6
Run the full 5-fold baseline using all sensors.

### Step 7
Run a supplementary single-action experiment using windowing.

---

## 2. Data Understanding

### Source data
Each `data_rot_xx_1.mat` file corresponds to one participant performing one full trial of actions.

From MATLAB inspection, each file contains:

- `lblMaster`: `23 x 1` cell
- `us25_data`: `23 x 1` cell
- `us33_data`: `23 x 1` cell
- `us40_data`: `23 x 1` cell

This means:

- each cell entry already corresponds to **one action**
- I do **not** need to manually segment actions from a continuous recording
- each action can be used directly as one sample candidate

### Labeling strategy
The original action-recognition task used:

- label = action

The new person-identification task uses:

- label = person

### Data policy
For fairness, the current main baseline uses only version `1` files (`*_1`) because not every participant has a `_2` file.

### Actions used
For the current baseline:
- use actions `1~21`
- exclude the last two freestyle actions

---

## 3. New Files I Created

To avoid modifying the earlier action-recognition code, I created a separate person-ID pipeline.

### `LSTM/data_loading_person_yichiao.py`
Purpose:
- scan all `data_rot_*_1.mat` files
- read `us25_data`, `us33_data`, `us40_data`, and `lblMaster`
- convert each action segment into one sample
- assign the label as **person ID**
- support:
  - `US25`
  - `US33`
  - `US40`
  - `all`
- skip empty samples automatically
- later, also support windowing for the single-action experiment

### `LSTM/PersonLstm_yichiao.py`
Purpose:
- define the PyTorch LSTM model for person identification
- load data from `data_loading_person_yichiao.py`
- run training and validation
- support fold-based evaluation
- support optional windowing settings (`window_len`, `stride`)

### `LSTM/test_run_person_yichiao.py`
Purpose:
- first reproducible person-ID baseline
- uses:
  - `US40`
  - actions `1~21`
  - version `1`
  - fold `0`
  - `5` epochs

### `LSTM/test_run_person_all_yichiao.py`
Purpose:
- improved person-ID baseline using:
  - `all` sensors
  - actions `1~21`
  - version `1`
  - fold `0`
  - `5` epochs

### `LSTM/test_run_person_all_5fold_yichiao.py`
Purpose:
- main person-ID baseline
- runs:
  - `all` sensors
  - actions `1~21`
  - version `1`
  - folds `[0,1,2,3,4]`
  - `5` epochs

### `LSTM/test_run_person_single_action_all_yichiao.py`
Purpose:
- supplementary experiment
- uses only **one fixed action**
- applies **windowing**
- uses `all` sensors
- checks whether fixing the action makes person identification easier

---

## 4. Environment Setup

I ran the experiments in a conda environment called `mies_yichiao`.

### Recommended setup

```bash
conda create -n mies_yichiao python=3.11
conda activate mies_yichiao
conda install numpy=1.26 scipy matplotlib scikit-image ipykernel pytorch -c pytorch
```

### Why NumPy was pinned

I previously ran into a NumPy 2.x compatibility issue with PyTorch, so `numpy=1.26` was used to avoid that problem.

### Before running anything

Make sure you are at the **repo root**, not inside `LSTM/`.

Example:

```bash
cd Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System
conda activate mies_yichiao
```

## 5. Person-ID Dataset Construction

### Files used

The person-ID task uses all `*_1.mat` files in the `data/` folder.

### Empty samples found and skipped

The loader automatically skipped the following empty samples:

- `data_rot_bb_1.mat`: action 1, action 2
- `data_rot_mc_1.mat`: action 4, action 5, action 21

### Usable sample count

Expected maximum:
- 10 participants
- 21 actions each
- total = `210`

Actual usable samples:
- total = `205`

So the dataset is slightly incomplete, but still mostly balanced.

---

## 6. Reproducible Run Order

If someone wants to reproduce the current person-ID results, this is the order to follow.

### Step A — Run the first `US40` baseline

Command:

```bash
python LSTM/test_run_person_yichiao.py
```

What this does:
- uses `US40` only
- uses actions `1~21`
- uses version `1`
- uses fold `0`
- trains for `5` epochs

Expected result:
- a weak but working baseline
- best valid accuracy around **20.83%**

This step is mainly used to confirm:
- the person-ID loader works
- the model can run successfully
- single-sensor person ID is hard

### Step B — Run the improved `all`-sensor baseline (fold 0 only)

Command:

```bash
python LSTM/test_run_person_all_yichiao.py
```

What this does:
- uses `all` sensors (`US25 + US33 + US40`)
- uses actions `1~21`
- uses version `1`
- uses fold `0`
- trains for `5` epochs

Expected result:
- much stronger than `US40`
- best valid accuracy around **64.58%**

This step is used to show:
- multi-sensor fusion gives a major improvement

### Step C — Run the main 5-fold baseline with all sensors

Command:

```bash
python LSTM/test_run_person_all_5fold_yichiao.py
```

What this does:
- uses `all` sensors
- uses actions `1~21`
- uses version `1`
- runs full `5` folds
- trains for `5` epochs each fold

Expected result:
- final valid accuracies roughly:
  - Fold 0: `64.58%`
  - Fold 1: `72.50%`
  - Fold 2: `65.00%`
  - Fold 3: `71.79%`
  - Fold 4: `57.89%`
- mean final validation accuracy = **66.35%**

This is the current main baseline result for the person-ID project.

### Step D — Run the supplementary single-action experiment

Command:

```bash
python LSTM/test_run_person_single_action_all_yichiao.py
```

What this does:
- uses `all` sensors
- uses only one fixed action
- applies windowing
- uses:
  - `window_len = 200`
  - `stride = 100`
- runs 5-fold evaluation

Expected result:
- final valid accuracies roughly:
  - Fold 0: `40.00%`
  - Fold 1: `58.33%`
  - Fold 2: `63.64%`
  - Fold 3: `66.67%`
  - Fold 4: `50.00%`
- mean final validation accuracy = **55.73%**

This experiment is supplementary only, not the main result.

---

## 7. First Baseline Result: US40 Only

### Setting
- task: person identification
- sensor: `US40`
- version: `1`
- actions: `1~21`
- model: `1-layer LSTM`
- hidden size: `400`
- fold: `0`
- epochs: `5`
- batch size: `16`

### Dataset summary

#### Train fold 0
- num_samples: `157`
- num_classes: `10`
- feature dimension: `328`
- sequence length min / mean / max:
  - `248 / 1098.34 / 4004`

#### Valid fold 0
- num_samples: `48`
- num_classes: `10`
- feature dimension: `328`
- sequence length min / mean / max:
  - `263 / 1138.25 / 3590`

### Results by epoch
- Epoch 1: train acc = `0.1146`, valid acc = `0.1875`
- Epoch 2: train acc = `0.3121`, valid acc = `0.1875`
- Epoch 3: train acc = `0.4076`, valid acc = `0.1667`
- Epoch 4: train acc = `0.6051`, valid acc = `0.1875`
- Epoch 5: train acc = `0.7006`, valid acc = `0.2083`

### Best result
- best valid accuracy: `20.83%`
- best epoch: `5`

### Interpretation

This confirmed that:
- the person-ID pipeline works
- the model learns something
- but `US40` alone is too weak for strong person classification

---

## 8. Improved Baseline: All Sensors, Fold 0

### Setting
- task: person identification
- sensor: `all` (`US25 + US33 + US40`)
- version: `1`
- actions: `1~21`
- model: `1-layer LSTM`
- hidden size: `400`
- fold: `0`
- epochs: `5`
- batch size: `16`

### Dataset summary

#### Train fold 0
- num_samples: `157`
- num_classes: `10`
- feature dimension: `983`
- sequence length min / mean / max:
  - `248 / 1098.34 / 4004`

#### Valid fold 0
- num_samples: `48`
- num_classes: `10`
- feature dimension: `983`
- sequence length min / mean / max:
  - `263 / 1138.25 / 3590`

### Results by epoch
- Epoch 1: train acc = `0.1847`, valid acc = `0.2917`
- Epoch 2: train acc = `0.7516`, valid acc = `0.5000`
- Epoch 3: train acc = `0.9108`, valid acc = `0.5625`
- Epoch 4: train acc = `0.9682`, valid acc = `0.6042`
- Epoch 5: train acc = `0.9618`, valid acc = `0.6458`

### Best result
- best valid accuracy: `64.58%`
- best epoch: `5`

### Interpretation

Compared with `US40`:
- `US40` only: `20.83%`
- `all` sensors: `64.58%`

This showed that multi-sensor fusion is very important for person identification.

---

## 9. Main Baseline: 5-Fold All-Sensor Person ID

### Setting
- task: person identification
- sensor: `all`
- version: `1`
- actions: `1~21`
- model: `1-layer LSTM`
- hidden size: `400`
- epochs: `5`
- folds: `[0, 1, 2, 3, 4]`
- batch size: `16`

### Final validation accuracy by fold
- Fold 0: `64.58%`
- Fold 1: `72.50%`
- Fold 2: `65.00%`
- Fold 3: `71.79%`
- Fold 4: `57.89%`

### Summary
- mean final validation accuracy: `66.35%`
- range: `57.89%` to `72.50%`

### Interpretation

This is the first full 5-fold baseline for the person-identification task using all three sensors together.

Compared with the earlier `US40` baseline:
- `US40` only: best valid accuracy = `20.83%`
- `all` sensors: mean final valid accuracy = `66.35%`

This is the current main result for the person-ID project.

---

## 10. Supplementary Experiment: Single-Action Person ID with Windowing

After building the main baseline, I also tested a more controlled experiment:

- use only one action
- use all sensors
- split each full action segment into overlapping windows
- check whether fixing the action helps person identification

### File used
- `LSTM/test_run_person_single_action_all_yichiao.py`

### Setting
- task: person identification
- sensor: `all`
- version: `1`
- action_indices: `[9]`
- action: one fixed action
- model: `1-layer LSTM`
- hidden size: `400`
- epochs: `5`
- folds: `[0, 1, 2, 3, 4]`
- window length: `200`
- stride: `100`
- batch size: `32`
- max length: `200`

### Final validation accuracy by fold
- Fold 0: `40.00%`
- Fold 1: `58.33%`
- Fold 2: `63.64%`
- Fold 3: `66.67%`
- Fold 4: `50.00%`

### Summary
- mean final validation accuracy: `55.73%`
- range: `40.00%` to `66.67%`

### Interpretation

This result shows that single-action person identification is still possible, because the accuracy is clearly above random guessing.

However, it did not outperform the main baseline:
- main baseline (`all` sensors, actions `1~21`): `66.35%`
- single-action windowed baseline: `55.73%`

So at the moment, the single-action experiment is a supplementary exploratory result, not the main result.

### Important limitation

This experiment uses overlapping windows cut from the same original action segment.  
That means the train and validation sets may still contain windows derived from the same source trial, so this evaluation is not as clean as a fully source-separated split.

---

## 11. What We Have Done So Far

Up to this point, the work completed is:

1. inspected the participant `.mat` files in MATLAB
2. confirmed that each action is already stored as a separate cell
3. changed the task from action recognition to person identification
4. created a new person-ID data loader
5. added automatic empty-sample handling
6. built a PyTorch LSTM person-ID training pipeline
7. ran a first `US40` baseline
8. ran an improved `all`-sensor baseline
9. ran the full 5-fold all-sensor baseline
10. ran a supplementary single-action windowed experiment

So the project is no longer just re-running an older model.  
It now has a clear new question:

> Can ultrasonic motion signatures be used to identify different people, and how much does multi-sensor fusion help?

---

## 12. Main Takeaways

1. The project has successfully shifted from action recognition to person identification.
2. The new person-ID data loader works correctly.
3. Empty or missing samples can now be handled automatically.
4. Person identification is much harder than action recognition.
5. Single-sensor performance is weak.
6. Multi-sensor fusion gives a major performance improvement.
7. The current main baseline is meaningful, but overfitting is still present.
8. The current single-action windowed experiment does not outperform the full all-action baseline.

---

## 13. Next Steps

### Priority 1
Clean up and document the current baseline pipeline for the team.

### Priority 2
If needed, test additional sensor settings for person ID:
- `US25`
- `US33`
- `US40`
- `all`

### Priority 3
If enough time remains, test whether version `2` files can be used for:
- cross-session evaluation
- robustness testing
- supplementary experiments

### Priority 4
If we continue the single-action direction, improve the evaluation split so that windows from the same original source segment do not leak across train and validation.

---

## 14. Current Status
- baseline action-recognition pipeline: done
- person-ID loader: done
- empty sample handling: done
- first person-ID smoke test: done
- first 5-epoch US40 person-ID baseline: done
- all-sensor person-ID baseline (fold 0): done
- all-sensor 5-fold person-ID baseline: done
- single-action person-ID supplementary experiment: done
- cleaner source-separated single-action evaluation: not started yet
- cross-session testing with version 2: not started yet

---

## 15. Cross-session testing w/ Version 2
The testing accuracy dropped significantly when identifying the same person between different sessions. The results were as follows for 1 fold:
Epoch 1 | train_acc=0.1463 valid_acc=0.1310
Epoch 2 | train_acc=0.4195 valid_acc=0.2857
Epoch 3 | train_acc=0.5512 valid_acc=0.3214
Epoch 4 | train_acc=0.7024 valid_acc=0.3333
Epoch 5 | train_acc=0.7756 valid_acc=0.3214

--- Per-Person Accuracy (Cross-Session) ---
dm        :  42.86% (9/21)
gg        :   9.52% (2/21)
ks        :  28.57% (6/21)
tm        :  47.62% (10/21)

with the following settings:
        "sensor_data": "all",
        "action_indices": list(range(21)), # All 21 Actions
        "lstm_layers": [400],
        "nepochs": 5,                   # 5 Epochs
        "bsize": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
