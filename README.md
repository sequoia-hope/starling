# Author's Note

Heads up this entire project was vibe coded with ChatGPT and later Claude. I saw [Benn Jordan's video](https://www.youtube.com/watch?v=hCQCP-5g5bo) and thought I could apply my knowledge of digital modulation to build a basic research tool.

I have not thoroughly reviewed this code. It was a fun idea and I present this to the community as a jumping off point for further exploration.

Everything that follows is AI generated.

# Starling Digital Communication Testbench

A digital signal processing testbench for testing ASK/FSK modulation schemes through biological channels, specifically starling vocal reproduction.

## Inspiration

This project was inspired by Benn Jordan's YouTube video demonstrating that starlings can reproduce audio with sufficient fidelity to recover encoded bitmap data. The video showed that when bitmaps were encoded as audio and played to starlings, the birds could mimic the sounds with enough precision that parts of the original bitmap could be reconstructed from their vocalizations.

This discovery raised an intriguing question: **Can starlings act as a biological transmission channel for digital data?**

## Project Overview

This testbench allows researchers to explore whether traditional digital modulation schemes (ASK and FSK) can successfully transmit data through starling vocal reproduction. The process involves:

1. **Encoding**: Convert binary data into audio using ASK or FSK modulation
2. **Transmission**: Play the encoded audio to starlings
3. **Biological Channel**: Starlings learn and reproduce the audio
4. **Reception**: Record the starling's reproduction
5. **Decoding**: Analyze the recorded audio to recover the original data
6. **Analysis**: Measure bit error rates and channel characteristics

## Features

### Modulation Schemes
- **ASK (Amplitude Shift Keying)**: Encodes data by varying signal amplitude
- **FSK (Frequency Shift Keying)**: Encodes data by switching between frequencies (more robust for biological channels)

### Channel Simulation
- **Additive noise**: Simulates environmental and recording noise
- **Frequency shifting**: Models pitch changes in starling reproduction
- **Split frequency shifting**: Applies different shifts to reference vs. noisy signals

### Analysis Tools
- **Bit Error Rate (BER)** calculation
- **FFT-based demodulation** for improved accuracy
- **Spectrogram visualization** with detected bits overlay
- **Real-time audio playback** for testing with live starlings

### Visualization
- Spectrograms with color-coded bit detection (green = '1', red = '0')
- Waterfall plots showing frequency content over time
- Debug output for per-bit analysis

## Installation

### Requirements
```bash
pip install numpy matplotlib scipy sounddevice
```

### Optional Dependencies
- `sounddevice`: For real-time audio playback (install with `pip install sounddevice`)

## Usage

### Quick Start

Generate a test signal using the default starling-optimized parameters:
```bash
python starling_testbench.py
```

This creates `fsk.wav` with a 64-bit test pattern using FSK modulation optimized for starling vocal range (1500-2500 Hz).

### Basic Examples

**Generate FSK test signal:**
```bash
python starling_testbench.py --mode fsk --outfile starling_test.wav
```

**Generate ASK test signal:**
```bash
python starling_testbench.py --mode ask --outfile starling_ask.wav
```

**Decode recorded starling audio:**
```bash
python starling_testbench.py --decode-from starling_recording.wav --plot
```

**Test with frequency shift (simulating pitch change):**
```bash
python starling_testbench.py --freq-shift 100 --debug
```

### Preset-Based Testing

**Use optimized presets for different scenarios:**
```bash
# Standard starling experiments
python starling_testbench.py --preset starling-default

# ASK modulation testing
python starling_testbench.py --preset ask-optimized

# High noise robustness testing
python starling_testbench.py --preset high-noise --plot

# Fast data rate experiments
python starling_testbench.py --preset fast-rate

# Long-distance/low-frequency testing
python starling_testbench.py --preset low-freq

# Standardized research protocol
python starling_testbench.py --preset research-protocol --debug
```

### Research Workflows

**Complete experimental protocol:**
```bash
# 1. Generate training signals for starlings
python starling_testbench.py --preset starling-default --outfile training.wav --play

# 2. Generate test signals with different parameters
python starling_testbench.py --preset high-noise --outfile test_noisy.wav
python starling_testbench.py --preset fast-rate --outfile test_fast.wav

# 3. After recording starling reproductions, analyze them
python starling_testbench.py --decode-from starling_reproduction.wav --preset starling-default --plot --debug
```

**Comparative modulation study:**
```bash
# Compare ASK vs FSK performance
python starling_testbench.py --preset starling-default --outfile fsk_test.wav
python starling_testbench.py --preset ask-optimized --outfile ask_test.wav

# Decode both and compare BER results
python starling_testbench.py --decode-from recorded_fsk.wav --preset starling-default
python starling_testbench.py --decode-from recorded_ask.wav --preset ask-optimized
```

**Parameter optimization study:**
```bash
# Test frequency range sensitivity
python starling_testbench.py --preset low-freq --debug    # 800-1200 Hz
python starling_testbench.py --preset starling-default    # 1500-2500 Hz
python starling_testbench.py --fsk-freq0 2000 --fsk-freq1 3000  # 2000-3000 Hz

# Test bit rate vs accuracy trade-off
python starling_testbench.py --preset fast-rate --debug   # 40ms bits
python starling_testbench.py --preset starling-default    # 80ms bits  
python starling_testbench.py --bitdur 0.15 --debug        # 150ms bits
```

### Advanced Usage

**Custom bit pattern:**
```bash
python starling_testbench.py --bits "110011001100" --bitdur 0.1
```

**Low noise testing:**
```bash
python starling_testbench.py --snr 30 --no-noise
```

**Real-time playback for live starling testing:**
```bash
python starling_testbench.py --play
```

**Full analysis with visualization:**
```bash
python starling_testbench.py --plot --debug --outfile experiment1.wav
```

## Command Line Options

### Core Parameters
- `--mode`: Modulation type (`ask` or `fsk`)
- `--bits`: Custom bit pattern (string of 0s and 1s)
- `--bitdur`: Duration per bit in seconds (default: 0.08)
- `--fs`: Sample rate in Hz (default: 44100)

### FSK Parameters
- `--fsk-freq0`: Frequency for '0' bits (default: 1500 Hz)
- `--fsk-freq1`: Frequency for '1' bits (default: 2500 Hz)

### ASK Parameters
- `--ask-freq`: Carrier frequency (default: 2200 Hz)
- `--amp0`: Amplitude for '0' bits (default: 0.2)
- `--threshold`: Demodulation threshold (default: 0.3)

### Channel Effects
- `--snr`: Signal-to-noise ratio in dB (default: 20)
- `--noise` / `--no-noise`: Enable/disable noise
- `--freq-shift`: Frequency shift in Hz
- `--shift-reference`: Apply freq shift only to reference signal

### I/O and Analysis
- `--outfile`: Output WAV filename
- `--decode-from`: Decode existing WAV file
- `--play`: Enable audio playback
- `--plot`: Show spectrogram visualization
- `--debug`: Print detailed per-bit analysis

## Presets

The testbench includes 6 optimized presets for different experimental scenarios:

### starling-default
Standard configuration for starling vocal reproduction experiments:
- **Mode**: FSK (more robust than ASK)
- **Frequencies**: 1500-2500 Hz (within starling vocal range)
- **Bit duration**: 80ms (allows time for biological reproduction)
- **Test pattern**: 64-bit sequence with good frequency balance

### ask-optimized
Optimized for ASK modulation experiments:
- **Mode**: ASK with enhanced amplitude contrast
- **Frequency**: 2000 Hz carrier (lower for better starling reproduction)
- **Bit duration**: 100ms (longer for amplitude detection)
- **SNR**: 25dB (higher SNR needed for ASK)

### high-noise
Designed for testing noise robustness:
- **Mode**: FSK with wider frequency separation (1200-2800 Hz)
- **Bit duration**: 120ms (longer for noise immunity)
- **SNR**: 10dB (challenging noise conditions)
- **Pattern**: Block sequences for error pattern analysis

### fast-rate
High-speed data transmission testing:
- **Mode**: FSK with optimized timing
- **Frequencies**: 1600-2400 Hz (800 Hz separation)
- **Bit duration**: 40ms (25 bits/second)
- **SNR**: 30dB (clean signal for speed)

### low-freq
Lower frequency range for distance/obstacle testing:
- **Mode**: FSK at 800-1200 Hz
- **Bit duration**: 100ms (optimized for low-freq propagation)
- **Use case**: Testing through barriers or over distance

### research-protocol
Standardized configuration for comparative studies:
- **Mode**: FSK with structured test patterns
- **Includes**: Synchronization preamble
- **Pattern**: Designed for statistical analysis
- **Use case**: Reproducible research across different studies

## Experimental Workflow

### Phase 1: Baseline Testing
1. Generate test signals with known parameters
2. Add controlled noise and frequency shifts
3. Verify demodulation accuracy with simulated channel effects

### Phase 2: Starling Training
1. Play generated test signals to starlings repeatedly
2. Allow starlings to learn and reproduce the patterns
3. Record starling reproductions in controlled acoustic environment

### Phase 3: Analysis
1. Decode recorded starling audio using the testbench
2. Calculate bit error rates and analyze error patterns
3. Visualize results using spectrogram overlays
4. Optimize parameters based on starling reproduction characteristics

### Phase 4: Optimization
1. Adjust frequencies to match starling vocal preferences
2. Modify bit timing for optimal biological reproduction
3. Test different modulation schemes and parameters
4. Develop error correction strategies if needed

## Technical Details

### Frequency Selection
The default frequencies (1500-2500 Hz) are chosen to match the natural vocal range of European starlings, which typically ranges from 1-4 kHz with peak sensitivity around 2-3 kHz.

### Bit Timing
The 80ms bit duration provides sufficient time for starlings to reproduce frequency transitions while maintaining reasonable data rates (12.5 bits/second).

### FFT-Based Demodulation
Uses windowed FFT analysis for robust frequency detection, with Hamming windowing to reduce spectral leakage and improve frequency resolution.

### Error Analysis
Bit error rate calculations help quantify the "channel capacity" of starling vocal reproduction and identify optimal operating parameters.

## Research Applications

This testbench enables research into:
- **Bio-inspired communication systems**
- **Animal cognition and vocal learning**
- **Acoustic channel characterization**
- **Biomimetic signal processing**
- **Cross-species information transfer**

## Files Generated

- `fsk.wav` / `ask.wav`: Generated test signals
- Spectrograms: Visual analysis of signal characteristics
- Console output: BER measurements and debug information

## Testing & Validation

### Quick Verification
```bash
# Install dependencies
pip install -r requirements.txt

# Generate and test FSK signal
python starling_testbench.py --mode fsk --plot --debug

# Expected: BER ~0.0000 with clean signal, spectrograms showing 1500/2500 Hz switching
```

### Common Test Commands
```bash
# Basic functionality
python starling_testbench.py                    # Default FSK test
python starling_testbench.py --mode ask         # ASK modulation
python starling_testbench.py --play             # Audio playback

# Analysis
python starling_testbench.py --plot --debug     # Visualization + detailed output
python starling_testbench.py --snr 15           # Test noise robustness

# Decode recorded starling audio
python starling_testbench.py --decode-from recording.wav --plot
```

### Validation Checklist
- [ ] **Dependencies install**: `pip install -r requirements.txt` succeeds
- [ ] **Clean signal decode**: BER = 0.0000% with `--no-noise`
- [ ] **Noisy signal robustness**: BER < 1% with `--snr 20`
- [ ] **Spectrograms display**: Clear frequency transitions in `--plot` mode
- [ ] **Audio playback works**: `--play` produces audible tones (if speakers available)

## Troubleshooting

**No audio playback**: Install sounddevice (`pip install sounddevice`)

**Poor demodulation**: Try adjusting `--threshold` for ASK or `--freq-shift` to compensate for pitch changes

**High bit error rates**: Increase SNR, adjust frequencies to starling vocal range, or increase bit duration

## Contributing

This project welcomes contributions from researchers interested in bio-acoustic communication, digital signal processing, and animal cognition.

## Acknowledgments

- **Benn Jordan**: For the inspiring YouTube video that sparked this research direction
- **European Starling research community**: For foundational work on starling vocal learning
- **Digital signal processing community**: For the mathematical foundations of FSK/ASK modulation

## License

This project is released under an open source license to encourage scientific collaboration and reproducible research.
