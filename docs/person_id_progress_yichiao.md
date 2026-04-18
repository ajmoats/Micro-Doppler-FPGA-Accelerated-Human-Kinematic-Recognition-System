# Person ID Progress Log — YiChiao

## Current Task
The project direction has now shifted from action recognition to **person identification** using offline ultrasonic recordings.

The new goal is:

- input: ultrasonic motion data from one action segment
- output: which participant this sample belongs to

For the main baseline, I currently use:
- only version `*_1` files
- actions `1~21`
- label = person ID
- first test on `US40`

---

## Dataset Construction

### Data source
Each `data_rot_xx_1.mat` file corresponds to one participant performing a full trial of actions.

From MATLAB inspection:
- `lblMaster`: `23 x 1` cell
- `us25_data`: `23 x 1` cell
- `us33_data`: `23 x 1` cell
- `us40_data`: `23 x 1` cell

This means each action is already stored as one separate cell entry, so I do **not** need to manually segment actions.

### Current labeling strategy
- old task: label = action
- new task: label = person

### Current data policy
For fairness, the current main baseline uses only version `1` files (`*_1`), since not every participant has a `_2` file.

---

## First Person-ID Loader Result

I built a new person-identification loader and successfully ran the first training pipeline.

### Empty samples found and skipped
The loader found and skipped a few empty samples:

- `data_rot_bb_1.mat`: action 1, action 2
- `data_rot_mc_1.mat`: action 4, action 5, action 21

This reduced the total number of usable samples slightly.

### Usable sample count
Expected maximum:
- 10 participants
- 21 actions each
- total = 210 samples

Actual usable samples:
- total = `205`

So the current dataset is slightly incomplete but still mostly balanced.

---

## First Person-ID Baseline (US40 only)

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

### Samples per person
Train set:
- bb: 15
- dm: 16
- gg: 16
- gt: 16
- jm: 16
- jz: 16
- ks: 16
- mc: 14
- tf: 16
- tm: 16

Validation set:
- bb: 4
- dm: 5
- gg: 5
- gt: 5
- jm: 5
- jz: 5
- ks: 5
- mc: 4
- tf: 5
- tm: 5

---

## Training Result

### Results by epoch
- Epoch 1: train acc = `0.1146`, valid acc = `0.1875`
- Epoch 2: train acc = `0.3121`, valid acc = `0.1875`
- Epoch 3: train acc = `0.4076`, valid acc = `0.1667`
- Epoch 4: train acc = `0.6051`, valid acc = `0.1875`
- Epoch 5: train acc = `0.7006`, valid acc = `0.2083`

### Best result so far
- best valid accuracy: `20.83%`
- best epoch: `5`

---

## Interpretation

This result shows that the **person identification pipeline is working**, but the current baseline is still weak.

Important observations:

1. The model is learning something:
   - random guessing for 10 classes is about `10%`
   - current best validation accuracy is `20.83%`

2. However, the model is not generalizing well:
   - train accuracy rises to `70.06%`
   - validation accuracy stays around `16% ~ 21%`

3. This suggests likely overfitting and indicates that **person identification is much harder than action recognition**, especially when using only one sensor (`US40`) and mixing many different actions together.

This is still a meaningful result, because it confirms:
- the new task is technically feasible,
- the dataset has been successfully rebuilt for person ID,
- and there is room for real improvement.

---

## Current Conclusion

At this stage:

- the person-ID data loader works
- empty samples are handled correctly
- the first person-identification baseline has been successfully trained
- but `US40` alone is not strong enough for good person classification performance

So the next goal is no longer “can the pipeline run?”  
The next goal is: **how to improve person identification performance.**

---

## Next Steps

### Priority 1
Run the same person-ID experiment with:
- `sensor_data = "all"`

Reason:
- multi-sensor fusion was much stronger in the earlier action-recognition baseline
- it may also help person identification

### Priority 2
If needed, run a simpler baseline:
- single action only
- person identification within one controlled action
- example: `Walk in Place`

### Priority 3
After finding a stronger configuration:
- run full 5-fold evaluation
- compare `US25`, `US33`, `US40`, and `all`

---

## Current Status

- baseline action-recognition pipeline: done
- person-ID loader: done
- empty sample handling: done
- first person-ID smoke test: done
- first 5-epoch US40 person-ID baseline: done
- all-sensor person-ID baseline: not started yet
- single-action person-ID baseline: not started yet


## Update: Person-ID baseline with all sensors

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

### Empty samples skipped
The same empty samples were skipped as before:
- `data_rot_bb_1.mat`: action 1, action 2
- `data_rot_mc_1.mat`: action 4, action 5, action 21

### Results by epoch
- Epoch 1: train acc = `0.1847`, valid acc = `0.2917`
- Epoch 2: train acc = `0.7516`, valid acc = `0.5000`
- Epoch 3: train acc = `0.9108`, valid acc = `0.5625`
- Epoch 4: train acc = `0.9682`, valid acc = `0.6042`
- Epoch 5: train acc = `0.9618`, valid acc = `0.6458`

### Best result so far
- best valid accuracy: `64.58%`
- best epoch: `5`

### Interpretation
This result is much stronger than the earlier `US40`-only baseline.

Comparison:
- `US40` only: best valid accuracy = `20.83%`
- `all` sensors: best valid accuracy = `64.58%`

This suggests that multi-sensor fusion is very important for person identification. Even though the model still shows signs of overfitting, combining all three sensors gives a major improvement over using only one sensor.

### Current conclusion
The current best person-identification baseline is:
- version `1` only
- actions `1~21`
- all sensors combined
- 1-layer LSTM
- 5 epochs
- fold 0 best valid accuracy = `64.58%`

This is now the strongest person-ID baseline so far, and the next step is to run full 5-fold evaluation.