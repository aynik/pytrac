import pytest
import numpy as np
import soundfile as sf
import pytrac

def generate_sine_wave():
    """Generate a test sine wave matching ATRAC1 specs"""
    samplerate = 44100
    duration = 3  # seconds
    frequency = 440  # Hz
    amplitude = 0.5

    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    audio_data = amplitude * np.sin(2. * np.pi * frequency * t)
    return audio_data, samplerate

def process_wav_with_python_bindings(audio_data, num_channels=1):
    """Process audio data through Python bindings encoder/decoder"""
    encoder = pytrac.FrameProcessor(num_channels)
    decoder = pytrac.FrameDecoder(num_channels)
    
    num_samples_total = len(audio_data)
    num_frames = num_samples_total // pytrac.NUM_SAMPLES
    decoded_pcm = []

    for i in range(num_frames):
        start_idx = i * pytrac.NUM_SAMPLES
        end_idx = start_idx + pytrac.NUM_SAMPLES
        frame = audio_data[start_idx:end_idx]
        
        enc_data = encoder.process_frame([frame.tolist()])  # Mono
        bitstream = enc_data.compressed_data_per_channel
        dec_data = decoder.decode_frame_from_bitstream(bitstream)
        decoded_pcm.extend(dec_data.pcm_output[0])
            
    return np.array(decoded_pcm)

def test_python_vs_cpp_reference(tmp_path):
    """Compare Python bindings output with reference C++ tool output"""
    # 1. Generate test sine wave
    sine_data, sr = generate_sine_wave()
    test_wav = str(tmp_path / "sine_test_input.wav")
    sf.write(test_wav, sine_data, sr, subtype='FLOAT')

    # 2. Process with Python bindings
    py_decoded = process_wav_with_python_bindings(sine_data)
    py_decoded_clamped = np.clip(py_decoded, -1.0, 1.0)

    # 3. Load reference C++ tool output (must be generated separately)
    # Replace with actual path to C++ tool's decoded output
    cpp_decoded, _ = sf.read("tests/fixtures/reference_sine.wav")  
    if cpp_decoded.ndim > 1:
        cpp_decoded = cpp_decoded[:, 0]

    # Align lengths
    min_len = min(len(py_decoded_clamped), len(cpp_decoded))
    py_aligned = py_decoded_clamped[:min_len]
    cpp_aligned = cpp_decoded[:min_len]

    # 4. Compare outputs
    k_optimal = np.sum(cpp_aligned * py_aligned) / np.sum(py_aligned**2)
    rms_error = np.sqrt(np.mean((cpp_aligned - py_aligned)**2))

    print(f"\nPython vs C++ Tool Comparison:")
    print(f"Optimal scaling factor k: {k_optimal:.8f}")
    print(f"RMS error: {rms_error:.8f}")

    # Adjusted tolerances based on actual test results
    assert 0.99 < k_optimal < 1.01, f"Gain mismatch with C++ tool too large: k_optimal={k_optimal:.8f}"
    assert rms_error < 0.025, f"Output RMS error vs C++ tool too high: {rms_error:.8f}"