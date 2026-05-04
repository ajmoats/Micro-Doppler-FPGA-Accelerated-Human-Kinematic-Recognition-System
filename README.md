# Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System

### Context and Background
Traditional optical identification systems often degrade in low-light, smoke, or occluded environments. To address these limitations, this project proposes a micro-Doppler radar-based human motion recognition system accelerated on FPGA hardware. By analyzing motion-induced frequency shifts from gait and limb movement, the system classifies human kinematic patterns in a privacy-preserving and low-latency manner without relying on facial recognition.

### System Overview
Our research utilizes a **tri-sensor ultrasonic array** (US25, US33, and US40) to capture motion from multiple angles. The processing pipeline consists of:
* **Data Acquisition:** I/Q demodulation and spectrogram generation.
* **Feature Extraction:** Concatenating individual sensor streams into a **983-dimensional fused feature vector**.
* **Classification:** A 1-layer LSTM network with 400 hidden units and 0.5 dropout.

### Key Research Findings
The software baseline established significant performance gains through multi-sensor fusion:
* **Action Recognition:** Achieved a stable **97.23% accuracy** across 21 body actions.
* **Person Identification:** Fusion improved performance from a 20.83% (US40 baseline) to **66.35% accuracy** in single-session settings.
* **Cross-Session Performance:** Dropped to 33.33%, highlighting challenges in generalization across different recording environments.

## Instructions
Be sure to drag the "data_rot_chunks10.mat" data file into the same folder.

## The Team
![Team Group Photo](20260408_154805.jpeg)

*Left to Right: Alexis Moats, Gabrielle Chavez, YiChiao Wang*
