# Live Collection Demo Workflow

This repo already includes the original ultrasound demo executables and MATLAB script under `ultrasonic_files/ultrasound_demo/`.

Use the helper script for a dry-run checklist:

```bash
python ultrasonic_files/ultrasound_demo/live_collection_demo.py --com 5
```

To launch the Windows executables after verifying the hardware setup:

```bash
python ultrasonic_files/ultrasound_demo/live_collection_demo.py --com 5 --gain 1 --run
```

What it does:

1. Verifies that `dacq_spga.exe`, `dacq2pipe.exe`, and the MATLAB script are present.
2. Prints the exact acquisition commands for the selected COM port.
3. Optionally launches the two Windows executables.
4. Tells the operator how to run the MATLAB spectrogram script.

Notes:

- This is a demo/collection helper, not live model inference.
- The MATLAB script remains the source of truth for the live spectrogram display.
- If `matlab` is not on `PATH`, open MATLAB manually in `ultrasonic_files/ultrasound_demo/` and run `script_us.m`.
