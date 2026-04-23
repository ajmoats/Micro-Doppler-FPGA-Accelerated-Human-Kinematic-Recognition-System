# Completed Work Summary

## Project Direction

The work completed so far establishes a software-first baseline for the Micro-Doppler FPGA-Accelerated Human Kinematic Recognition System and then extends that baseline into a person-identification workflow using offline ultrasonic recordings.

The current repo now reflects three main tracks of completed work:

1. a reproducible action-recognition baseline
2. a separate person-identification pipeline
3. supporting tooling for evaluation, documentation, and live collection demo setup

## 1. Action-Recognition Baseline

Completed work in the action-recognition pipeline includes:

- setting up a working Python/PyTorch environment for reproducible experiments
- creating `yichiao` baseline files so the newer experiments could run without directly depending on the older legacy training path
- validating the pipeline with a smoke test on `US40`
- comparing single-sensor and all-sensor inputs
- running a 5-fold baseline with all sensors combined

Current baseline result:

- task: action recognition
- input: `US25 + US33 + US40`
- model: 1-layer LSTM
- evaluation: 5-fold cross-validation
- aggregate accuracy: `97.96%` (`results/action_recognition/baseline_all_5fold/aggregate/summary.json`)

This establishes the strongest completed action-recognition baseline in the repo so far.

## 2. Person-Identification Pipeline

The project was then extended from action classification to person identification using the recorded ultrasonic trial files.

Completed work in this track includes:

- inspecting the `.mat` participant files and confirming each action is already stored as an individual segment
- creating a separate loader for person-ID data
- switching labels from action IDs to participant IDs
- skipping empty samples automatically during dataset construction
- building a dedicated PyTorch LSTM training path for person identification
- running smoke-test and baseline experiments for person ID
- running full 5-fold all-sensor evaluation
- updating the cross-session runner to train and test only on the overlapping version-2 participant subset: `dm`, `gg`, `ks`, and `tm`
- remapping that cross-session configuration to a 4-class label space so train/test labels stay aligned

Completed dataset assumptions:

- use `*_1.mat` files for consistency across participants
- use actions `1` through `21`
- exclude the last two freestyle actions
- for the current cross-session runner, restrict both sessions to the version-2 overlap set: `dm`, `gg`, `ks`, `tm`

Current completed person-ID results reflected in `results/`:

- `validation_person_smoke`: fold-0 smoke-test output for `US40`
- `Split_Person_US40`: 5-fold person-ID run using `US40` only with aggregate accuracy `13.66%`
- `Split_Person_LSTM_ALL`: 5-fold person-ID run using all sensors with aggregate accuracy `34.15%`

These saved runs show that person identification is feasible offline, and that combining all three sensors improves performance relative to `US40` alone in the current tracked results.

Cross-session status:

- `LSTM/PersonLstm_Cross.py` now filters both training and test splits to `dm`, `gg`, `ks`, and `tm`
- the verified loader output for that configuration is `84` training samples from version `1` and `84` test samples from version `2`
- full cross-session training was not completed in this update, so no new saved accuracy result has been added to `results/`

## 3. Evaluation and Reporting Improvements

Supporting work has also been completed to make experiments easier to validate and review:

- result folders are now populated under `results/action_recognition/` and `results/person_identification/`
- confusion matrices, normalized confusion matrices, history plots, and JSON summaries are being saved
- shared reporting helpers were added in `LSTM/eval_utils_yichiao.py`
- aggregate and per-fold outputs are present for:
  - `action_recognition/baseline_all_5fold`
  - `action_recognition/validation_action_smoke`
  - `person_identification/validation_person_smoke`
  - `person_identification/Split_Person_US40`
  - `person_identification/Split_Person_LSTM_ALL`

These changes make the experiments more reproducible and easier to compare across runs.

## 4. Repository Organization Updates

The current branch/worktree also reflects cleanup and organization work:

- legacy files were preserved under `LSTM/Original_Code/`
- runnable experiment entry points were reorganized under `LSTM/test/`
- older top-level test scripts in `LSTM/` were removed from the active layout
- documentation was added for the live ultrasonic collection demo in `docs/live_collection_demo.md`
- a helper script for the collection workflow exists at `ultrasonic_files/ultrasound_demo/live_collection_demo.py`

This improves separation between legacy reference code and the active experiment pipeline.

## 5. Main Takeaways

- A stable software baseline for action recognition has been completed.
- Multi-sensor fusion is the strongest completed input configuration.
- The repo now supports both action recognition and person identification workflows.
- Person identification has been demonstrated offline with saved smoke-test, `US40`, and all-sensor result sets.
- Result artifacts and plotting utilities are in place to support review and future comparisons.
- The codebase has been cleaned up so active experiments, legacy code, and demo tooling are more clearly separated.

## 6. Completed Status

Completed:

- environment setup for reproducible experiments
- action-recognition smoke testing
- action-recognition sensor comparison
- action-recognition 5-fold all-sensor baseline
- person-ID data loading pipeline
- person-ID baseline training pipeline
- person-ID smoke testing
- person-ID 5-fold `US40` baseline saved under `results/person_identification/Split_Person_US40`
- person-ID 5-fold all-sensor baseline saved under `results/person_identification/Split_Person_LSTM_ALL`
- cross-session person-ID loader updated to the shared version-1/version-2 participant subset (`dm`, `gg`, `ks`, `tm`) with verified `84`/`84` sample splits
- saved evaluation artifacts and summary reports
- live collection demo helper and documentation
- repo organization for legacy vs. active experiment code

Not yet completed:

- source-separated single-action evaluation cleanup
- broader overfitting mitigation experiments 
- saved cross-session evaluation results for the `dm`/`gg`/`ks`/`tm` subset **in progress**
- FPGA-oriented model simplification or deployment work
