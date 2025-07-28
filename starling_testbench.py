#!/usr/bin/env python3
"""
Starling Digital Communication Testbench

A digital signal processing testbench for testing ASK/FSK modulation schemes
through biological channels, specifically starling vocal reproduction.

Inspired by Benn Jordan's YouTube video demonstrating that starlings can
reproduce audio with sufficient fidelity to recover encoded bitmap data.

This testbench allows researchers to:
- Generate ASK/FSK modulated bitstreams as audio
- Simulate channel effects (noise, frequency shifts)
- Test data transmission through starling vocal reproduction
- Measure bit error rates and visualize results

Author: Starling Communication Research Project
Version: 12 (adds split frequency shifting for reference/noisy paths)
"""

__version__ = "1.0.0"

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io.wavfile import write, read
import scipy.signal as signal
import os

try:
    import sounddevice as sd
except ImportError:
    sd = None

# === Constants ===
# Audio processing constants
SAMPLE_RATE_DEFAULT = 44100
BIT_DEPTH = 16
FFT_SIZE = 1024
FFT_OVERLAP = 512
MAX_AMPLITUDE = 32767
MIN_SNR = -50

# === Configuration Presets ===
# Optimized parameters for different experimental scenarios
PRESETS = {
    "starling-default": {
        "mode": "fsk",           # FSK modulation (more robust than ASK)
        "bitdur": 0.08,          # 80ms per bit (allows starling reproduction time)
        "fsk_freq0": 1500,       # '0' bit frequency (within starling vocal range)
        "fsk_freq1": 2500,       # '1' bit frequency (within starling vocal range)
        "ask_freq": 2200,        # ASK carrier frequency
        "amp0": 0.2,             # ASK '0' bit amplitude (20% of '1' bit)
        "threshold": 0.3,        # ASK demodulation threshold
        "bits": "1011001110001111" * 4,  # Test pattern (64 bits)
        "snr": 20,               # Signal-to-noise ratio in dB
        "noise": True            # Enable noise simulation
    },
    
    "ask-optimized": {
        "mode": "ask",           # ASK modulation for amplitude-sensitive channels
        "bitdur": 0.1,           # Longer bits for better ASK detection
        "ask_freq": 2000,        # Lower carrier frequency (better for starlings)
        "amp0": 0.1,             # Higher contrast between 0/1 amplitudes
        "threshold": 0.25,       # Lower threshold for better sensitivity
        "bits": "1010" * 16,     # Simple alternating pattern (64 bits)
        "snr": 25,               # Higher SNR needed for ASK
        "noise": True
    },
    
    "high-noise": {
        "mode": "fsk",           # FSK better for noisy conditions
        "bitdur": 0.12,          # Longer bits for noise robustness
        "fsk_freq0": 1200,       # Wider frequency separation
        "fsk_freq1": 2800,       # 1600 Hz separation for better detection
        "bits": "11110000" * 8,  # Block pattern for error analysis
        "snr": 10,               # Challenging noise level
        "noise": True
    },
    
    "fast-rate": {
        "mode": "fsk",           # FSK for speed
        "bitdur": 0.04,          # 40ms bits (25 bits/second)
        "fsk_freq0": 1600,       # Tighter frequency range
        "fsk_freq1": 2400,       # 800 Hz separation
        "bits": "101101001011" * 8,  # Varied pattern (96 bits)
        "snr": 30,               # High SNR for fast decode
        "noise": True
    },
    
    "low-freq": {
        "mode": "fsk",           # Lower frequencies for distance/obstacles
        "bitdur": 0.1,           # Longer bits for low freq propagation
        "fsk_freq0": 800,        # Lower frequency range
        "fsk_freq1": 1200,       # Still within most bird ranges
        "bits": "110011" * 12,   # Simple pattern (72 bits)
        "snr": 15,               # Moderate noise
        "noise": True
    },
    
    "research-protocol": {
        "mode": "fsk",           # Standard for comparative studies
        "bitdur": 0.08,          # Standard timing
        "fsk_freq0": 1500,       # Standard frequencies
        "fsk_freq1": 2500,
        "bits": "11110000111100001111000011110000",  # Structured test pattern
        "snr": 20,               # Baseline noise level
        "noise": True,
        "preamble": "1010101010"  # Synchronization pattern
    }
}

# === Digital Modulation Functions ===

def ask_modulate(bits, carrier_freq, fs, bit_dur, amp_0=0.3):
    """
    Amplitude Shift Keying (ASK) modulation.
    
    Encodes binary data by varying the amplitude of a carrier wave.
    '1' bits use full amplitude, '0' bits use reduced amplitude.
    
    Args:
        bits: String of '0' and '1' characters to encode
        carrier_freq: Carrier frequency in Hz
        fs: Sample rate in Hz
        bit_dur: Duration of each bit in seconds
        amp_0: Amplitude for '0' bits (fraction of '1' amplitude)
    
    Returns:
        numpy array of modulated audio samples
    """
    t = np.linspace(0, bit_dur, int(fs * bit_dur), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    return np.concatenate([(1.0 if b == '1' else amp_0) * carrier for b in bits]).astype(np.float32)

def fsk_modulate(bits, freq0, freq1, fs, bit_dur):
    """
    Frequency Shift Keying (FSK) modulation.
    
    Encodes binary data by switching between two frequencies.
    '0' bits use freq0, '1' bits use freq1.
    More robust than ASK for noisy channels like starling reproduction.
    
    Args:
        bits: String of '0' and '1' characters to encode
        freq0: Frequency for '0' bits in Hz
        freq1: Frequency for '1' bits in Hz
        fs: Sample rate in Hz
        bit_dur: Duration of each bit in seconds
    
    Returns:
        numpy array of modulated audio samples
    """
    t = np.linspace(0, bit_dur, int(fs * bit_dur), endpoint=False)
    return np.concatenate([
        np.sin(2 * np.pi * (freq1 if b == '1' else freq0) * t) for b in bits
    ]).astype(np.float32)

# === Digital Demodulation Functions ===

def fsk_demodulate_fft(sig, freq0, freq1, fs, bit_dur, debug=False):
    """
    FFT-based FSK demodulation with bit tracking.
    
    Demodulates FSK signal by analyzing frequency content of each bit period.
    Uses FFT to find dominant frequency and compares to expected freq0/freq1.
    
    Args:
        sig: Input audio signal
        freq0: Expected frequency for '0' bits
        freq1: Expected frequency for '1' bits
        fs: Sample rate in Hz
        bit_dur: Duration of each bit in seconds
        debug: Print per-bit analysis if True
    
    Returns:
        tuple: (decoded_bits_string, list_of_bit_centers)
        bit_centers format: [(sample_position, detected_freq, bit_value), ...]
    """
    spb = int(fs * bit_dur)
    bits = ''
    bit_centers = []
    for i in range(0, len(sig), spb):
        seg = sig[i:i + spb]
        if len(seg) < spb: continue
        windowed = seg * np.hamming(len(seg))
        freqs = np.fft.rfftfreq(len(windowed), 1/fs)
        spectrum = np.abs(np.fft.rfft(windowed))
        peak_freq = freqs[np.argmax(spectrum)]
        bit = '1' if abs(peak_freq - freq1) < abs(peak_freq - freq0) else '0'
        bits += bit
        bit_centers.append((i + spb // 2, peak_freq, bit))
        if debug:
            print(f"Bit {i//spb:03d}: peak={peak_freq:.1f} Hz → {bit}")
    return bits, bit_centers

def ask_demodulate(sig, carrier_freq, fs, bit_dur, threshold=0.5, debug=False):
    """
    ASK demodulation using energy detection.
    
    Demodulates ASK signal by measuring energy in carrier frequency band.
    Uses bandpass filter around carrier frequency and energy threshold.
    
    Args:
        sig: Input audio signal
        carrier_freq: ASK carrier frequency in Hz
        fs: Sample rate in Hz
        bit_dur: Duration of each bit in seconds
        threshold: Energy threshold for '1' vs '0' decision
        debug: Print per-bit analysis if True
    
    Returns:
        String of decoded bits
    """
    spb = int(fs * bit_dur)
    bits = ''
    sos = signal.butter(4, [carrier_freq - 100, carrier_freq + 100], btype='band', fs=fs, output='sos')
    sig = sig / np.max(np.abs(sig))
    for i in range(0, len(sig), spb):
        seg = sig[i:i + spb]
        if len(seg) < spb: continue
        energy = np.mean(np.abs(signal.sosfilt(sos, seg)))
        bit = '1' if energy > threshold else '0'
        bits += bit
        if debug:
            print(f"Bit {i // spb:03d}: energy={energy:.4f} → {bit}")
    return bits

# === Analysis and Utility Functions ===

def bit_error_rate(original, decoded):
    """
    Calculate bit error rate between original and decoded bitstreams.
    
    Args:
        original: Original bit string
        decoded: Decoded bit string
    
    Returns:
        tuple: (error_rate, total_errors)
    """
    trimmed = decoded[:len(original)]
    errors = sum(a != b for a, b in zip(original, trimmed))
    return errors / len(original), errors

def plot_spectrogram_with_overlay(sig, fs, bit_centers, bitdur):
    """
    Plot spectrogram with detected bits overlaid as colored markers.
    
    Visualizes frequency content over time with detected bit positions.
    Green markers = '1' bits, Red markers = '0' bits.
    Useful for analyzing starling reproduction accuracy.
    
    Args:
        sig: Audio signal
        fs: Sample rate
        bit_centers: List of (sample_pos, freq, bit_value) tuples
        bitdur: Bit duration in seconds
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    Pxx, freqs, bins, im = ax.specgram(sig, Fs=fs, NFFT=1024, noverlap=512, cmap='inferno')
    ax.set_title("FSK Waterfall with Detected Bits")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    for sample, freq, bit in bit_centers:
        ax.plot(sample / fs, freq, 'go' if bit == '1' else 'ro', markersize=5)
    plt.tight_layout()
    plt.colorbar(im, ax=ax, label="dB")
    plt.show()

def add_noise(signal, snr_db):
    """
    Add Gaussian white noise to signal at specified SNR.
    
    Simulates environmental noise, recording equipment noise,
    and acoustic channel effects.
    
    Args:
        signal: Input audio signal
        snr_db: Signal-to-noise ratio in decibels
    
    Returns:
        Noisy signal with same shape as input
    """
    if snr_db < MIN_SNR:
        print(f"⚠️  SNR={snr_db} dB too low; using pure noise")
        return np.random.normal(0, 1.0, signal.shape).astype(np.float32)
    rms = np.sqrt(np.mean(signal**2))
    noise_rms = rms / (10**(snr_db / 20))
    if not np.isfinite(noise_rms) or noise_rms > 1e3:
        print(f"⚠️  Noise RMS too high ({noise_rms}), clamping")
        noise_rms = 1.0
    noise = np.random.normal(0, noise_rms, signal.shape)
    return (signal + noise).astype(np.float32)

def apply_freq_shift(signal, fs, shift_hz):
    """
    Apply frequency shift to simulate pitch changes.
    
    Simulates the frequency shift that occurs when starlings
    reproduce audio at a different pitch than the original.
    
    Args:
        signal: Input audio signal
        fs: Sample rate
        shift_hz: Frequency shift in Hz (positive = higher pitch)
    
    Returns:
        Frequency-shifted signal
    """
    t = np.arange(len(signal)) / fs
    return (signal * np.exp(2j * np.pi * shift_hz * t)).real.astype(np.float32)

def write_wav(filename, data, fs):
    """
    Write audio data to WAV file.
    
    Converts float32 audio data to 16-bit integer WAV format.
    
    Args:
        filename: Output WAV filename
        data: Audio data as float32 array (-1.0 to 1.0)
        fs: Sample rate in Hz
    """
    write(filename, fs, (data * MAX_AMPLITUDE).astype(np.int16))

def main():
    """
    Main experiment controller.
    
    Handles command-line arguments, applies presets, generates test signals,
    simulates channel effects, performs demodulation, and calculates results.
    
    Supports two main modes:
    1. Signal generation: Create test audio files for starling playback
    2. Signal analysis: Decode audio files recorded from starling reproduction
    """
    parser = argparse.ArgumentParser(
        description="Starling Digital Communication Testbench",
        epilog="""Examples:
  Generate FSK test signal: python starling_testbench.py --mode fsk --outfile test.wav
  Decode recorded audio: python starling_testbench.py --decode-from recording.wav
  Test with frequency shift: python starling_testbench.py --freq-shift 100
  Visualize results: python starling_testbench.py --plot --debug
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", type=str)
    parser.add_argument("--mode", choices=["ask", "fsk"])
    parser.add_argument("--bits", type=str)
    parser.add_argument("--bitdur", type=float)
    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument("--snr", type=float)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--no-noise", dest="noise", action="store_false")
    parser.set_defaults(noise=True)
    parser.add_argument("--ask-freq", type=float)
    parser.add_argument("--amp0", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--fsk-freq0", type=float)
    parser.add_argument("--fsk-freq1", type=float)
    parser.add_argument("--freq-shift", type=float, default=0.0)
    parser.add_argument("--shift-reference", action="store_true")
    parser.add_argument("--outfile", type=str)
    parser.add_argument("--preamble", type=str, default="0101010101")
    parser.add_argument("--decode-from", type=str)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = PRESETS.get(args.preset or "starling-default", {}).copy()
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    fs = config.get("fs", 44100)
    bitdur = config.get("bitdur", 0.08)
    snr = config.get("snr", 30)
    bits = config.get("preamble", "") + config.get("bits", "10101010" * 4)

    if config.get("decode_from"):
        fs_wav, data = read(config["decode_from"])
        sig = data.astype(np.float32) / MAX_AMPLITUDE
        if config["mode"] == "ask":
            decoded = ask_demodulate(sig, config["ask_freq"], fs_wav, bitdur, config.get("threshold", 0.5), debug=config.get("debug"))
        else:
            decoded, centers = fsk_demodulate_fft(sig, config["fsk_freq0"], config["fsk_freq1"], fs_wav, bitdur, debug=config.get("debug"))
            if config.get("plot"):
                plot_spectrogram_with_overlay(sig, fs_wav, centers, bitdur)
        print("Decoded bits:\n", decoded)
        return

    if config["mode"] == "ask":
        sig = ask_modulate(bits, config["ask_freq"], fs, bitdur, amp_0=config.get("amp0", 0.2))
    else:
        sig = fsk_modulate(bits, config["fsk_freq0"], config["fsk_freq1"], fs, bitdur)

    sig_ref = sig.copy()
    if config.get("shift_reference"):
        sig_ref = apply_freq_shift(sig_ref, fs, config["freq_shift"])

    if config.get("play") and sd:
        print("Playing clean...")
        sd.play(sig_ref, fs)
        sd.wait()

    sig_noisy = apply_freq_shift(sig, fs, config["freq_shift"])
    noisy = add_noise(sig_noisy, snr) if config.get("noise", True) else sig_noisy

    outfile = config.get("outfile", f"{config['mode']}.wav")
    write_wav(outfile, noisy, fs)
    print(f"Saved noisy WAV to: {outfile}")

    if config.get("play") and sd:
        print("Playing noisy...")
        sd.play(noisy, fs)
        sd.wait()

    if config["mode"] == "ask":
        decoded = ask_demodulate(noisy, config["ask_freq"], fs, bitdur, config.get("threshold", 0.5), debug=config.get("debug"))
    else:
        decoded, centers = fsk_demodulate_fft(noisy, config["fsk_freq0"], config["fsk_freq1"], fs, bitdur, debug=config.get("debug"))
        if config.get("plot"):
            plot_spectrogram_with_overlay(noisy, fs, centers, bitdur)

    ber, errs = bit_error_rate(bits, decoded)
    print(f"\nBER: {ber:.4f} | Errors: {errs} / {len(bits)}")

if __name__ == "__main__":
    main()
