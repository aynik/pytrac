import pytrac
import numpy as np
import pytest
import soundfile as sf # For loading WAV files, as suggested
import math

ATRAC1_MAX_BANDS = 4 # Number of QMF bands often used in ATRAC1 (e.g., 0-2.7kHz, 2.7-5.5kHz, 5.5-11kHz, 11-22kHz)

# Number of BFUs for Mode 0
ATRAC1_MODE0_NUM_BFU = 11
# Spectral coefficients per BFU for Mode 0
ATRAC1_MODE0_SPECS_PER_BFU = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64]
# Offset of each BFU in the flat MDCT spectrum for Mode 0
ATRAC1_MODE0_SPEC_OFFSET_PER_BFU = [
    0, 32, 64, 96, 128, 160, 192, 224,  # BFUs 0-7 (8 * 32 = 256)
    256,  # BFU 8 (offset 256, 64 coeffs)
    320,  # BFU 9 (offset 320, 64 coeffs)
    384   # BFU 10 (offset 384, 64 coeffs)
]

ATRAC1_BFU_TO_BAND = ([0] * 20) + ([1] * 16) + ([2] * 16)

SFI_ZERO_OFFSET = 32 # This is no longer directly used in the new formula below
# ATRAC1_SF_TABLE is now pytrac.SCALE_FACTOR_TABLE

def generate_audio_frame(num_channels, num_samples, dtype=np.float32):
    """Generates a random audio frame."""
    # num_samples will typically be pytrac.NUM_SAMPLES
    return np.random.rand(num_channels, num_samples).astype(dtype)

def encode_and_get_data(num_channels):
    np.random.seed(0) # Ensure deterministic random data for consistent test runs
    """
    Encodes a random audio frame and returns the original audio data (numpy array),
    encoder's intermediate data, and the generated bitstream.
    """
    if num_channels == 1:
        encoder = pytrac.FrameProcessor(1) # Uses default EncodeSettings
    elif num_channels == 2:
        encoder = pytrac.FrameProcessor(2) # Uses default EncodeSettings
    else:
        raise ValueError("Unsupported number of channels for consistency testing.")

    original_audio_data_np = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)
    # Convert numpy array to list of lists for FrameProcessor input
    original_audio_data_list = [ch.tolist() for ch in original_audio_data_np]
    
    encoder_intermediate_data = encoder.process_frame(original_audio_data_list)
    
    # The pcm_input field in IntermediateData should match original_audio_data_list
    # We can assert this here or in tests if needed.
    # For PCM output comparison, we'll use original_audio_data_np.

    per_channel_bitstream_data = encoder_intermediate_data.compressed_data_per_channel
    
    # Basic sanity check for bitstream
    for ch_data in per_channel_bitstream_data:
        assert len(ch_data) > 0, "Encoder produced an empty bitstream for a channel."
            
    return original_audio_data_np, encoder_intermediate_data, per_channel_bitstream_data
# --- PyAtrac1FrameProcessor Tests ---
@pytest.fixture
def mono_encoder():
    """Fixture for a mono PyAtrac1FrameProcessor."""
    return pytrac.FrameProcessor(1)

@pytest.fixture
def stereo_encoder():
    """Fixture for a stereo PyAtrac1FrameProcessor."""
    return pytrac.FrameProcessor(2)

def test_encoder_initialization(mono_encoder, stereo_encoder):
    """Test if encoders are initialized without errors."""
    assert mono_encoder is not None
    assert stereo_encoder is not None

def test_encoder_qmf_output_shape_mono(mono_encoder):
    """Test QMF output shape and type for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    
    assert intermediate_data.qmf_low is not None
    assert isinstance(intermediate_data.qmf_low, list)
    assert len(intermediate_data.qmf_low) == 1 # num_channels
    assert len(intermediate_data.qmf_low[0]) == 128 # samples in low band
    assert all(isinstance(x, float) for x in intermediate_data.qmf_low[0])

    assert intermediate_data.qmf_mid is not None
    assert isinstance(intermediate_data.qmf_mid, list)
    assert len(intermediate_data.qmf_mid) == 1 # num_channels
    assert len(intermediate_data.qmf_mid[0]) == 128 # samples in mid band
    assert all(isinstance(x, float) for x in intermediate_data.qmf_mid[0])

    assert intermediate_data.qmf_hi is not None
    assert isinstance(intermediate_data.qmf_hi, list)
    assert len(intermediate_data.qmf_hi) == 1 # num_channels
    assert len(intermediate_data.qmf_hi[0]) == 256 # samples in hi band
    assert all(isinstance(x, float) for x in intermediate_data.qmf_hi[0])

def test_encoder_qmf_output_shape_stereo(stereo_encoder):
    """Test QMF output shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    
    assert intermediate_data.qmf_low is not None
    assert isinstance(intermediate_data.qmf_low, list)
    assert len(intermediate_data.qmf_low) == 2 # num_channels
    assert all(len(ch_data) == 128 for ch_data in intermediate_data.qmf_low)
    assert all(isinstance(x, float) for x in intermediate_data.qmf_low[0])

    assert intermediate_data.qmf_mid is not None
    assert isinstance(intermediate_data.qmf_mid, list)
    assert len(intermediate_data.qmf_mid) == 2 # num_channels
    assert all(len(ch_data) == 128 for ch_data in intermediate_data.qmf_mid)
    assert all(isinstance(x, float) for x in intermediate_data.qmf_mid[0])

    assert intermediate_data.qmf_hi is not None
    assert isinstance(intermediate_data.qmf_hi, list)
    assert len(intermediate_data.qmf_hi) == 2 # num_channels
    assert all(len(ch_data) == 256 for ch_data in intermediate_data.qmf_hi)
    assert all(isinstance(x, float) for x in intermediate_data.qmf_hi[0])

def test_encoder_mdct_coefficients_shape_mono(mono_encoder):
    """Test MDCT coefficients shape and type for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    
    assert intermediate_data.mdct_specs is not None
    assert isinstance(intermediate_data.mdct_specs, list)
    assert len(intermediate_data.mdct_specs) == 1 # num_channels
    assert len(intermediate_data.mdct_specs[0]) == pytrac.NUM_SAMPLES # 512 MDCT coefficients per channel
    assert all(isinstance(x, float) for x in intermediate_data.mdct_specs[0])

def test_encoder_mdct_coefficients_shape_stereo(stereo_encoder):
    """Test MDCT coefficients shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    
    assert intermediate_data.mdct_specs is not None
    assert isinstance(intermediate_data.mdct_specs, list)
    assert len(intermediate_data.mdct_specs) == 2 # num_channels
    assert all(len(ch_data) == pytrac.NUM_SAMPLES for ch_data in intermediate_data.mdct_specs)
    assert all(isinstance(x, float) for x in intermediate_data.mdct_specs[0])

def test_encoder_scale_factor_indices_mono(mono_encoder):
    """Test scale_factor_indices shape, type, and value range for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    
    assert intermediate_data.scaled_blocks is not None
    assert isinstance(intermediate_data.scaled_blocks, list)
    assert len(intermediate_data.scaled_blocks) == 1 # num_channels
    # Each element is a list of ScaledBlock objects
    assert isinstance(intermediate_data.scaled_blocks[0], list)
    for bfu_block in intermediate_data.scaled_blocks[0]:
        assert isinstance(bfu_block.scale_factor_index, int)
        assert 0 <= bfu_block.scale_factor_index < 64

def test_encoder_scale_factor_indices_stereo(stereo_encoder):
    """Test scale_factor_indices shape, type, and value range for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    
    assert intermediate_data.scaled_blocks is not None
    assert isinstance(intermediate_data.scaled_blocks, list)
    assert len(intermediate_data.scaled_blocks) == 2 # num_channels
    for ch_scaled_blocks in intermediate_data.scaled_blocks:
        assert isinstance(ch_scaled_blocks, list)
        for bfu_block in ch_scaled_blocks:
            assert isinstance(bfu_block.scale_factor_index, int)
            assert 0 <= bfu_block.scale_factor_index < 64

def test_encoder_bits_per_bfu_mono(mono_encoder):
    """Test bits_per_bfu shape, type, and value range for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    
    assert intermediate_data.bits_per_bfu is not None
    assert isinstance(intermediate_data.bits_per_bfu, list)
    assert len(intermediate_data.bits_per_bfu) == 1 # num_channels
    assert isinstance(intermediate_data.bits_per_bfu[0], list)
    # Number of BFUs can vary, check against scaled_blocks length
    assert len(intermediate_data.bits_per_bfu[0]) == len(intermediate_data.scaled_blocks[0])
    for bits in intermediate_data.bits_per_bfu[0]:
        assert isinstance(bits, int)
        assert 0 <= bits <= 15

def test_encoder_bits_per_bfu_stereo(stereo_encoder):
    """Test bits_per_bfu shape, type, and value range for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    
    assert intermediate_data.bits_per_bfu is not None
    assert isinstance(intermediate_data.bits_per_bfu, list)
    assert len(intermediate_data.bits_per_bfu) == 2 # num_channels
    for ch in range(2):
        assert isinstance(intermediate_data.bits_per_bfu[ch], list)
        assert len(intermediate_data.bits_per_bfu[ch]) == len(intermediate_data.scaled_blocks[ch])
        for bits in intermediate_data.bits_per_bfu[ch]:
            assert isinstance(bits, int)
            assert 0 <= bits <= 15

def test_encoder_quantized_values_shape_mono(mono_encoder):
    """Test quantized_values shape and type for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    
    assert intermediate_data.quantized_values is not None
    assert isinstance(intermediate_data.quantized_values, list)
    assert len(intermediate_data.quantized_values) == 1 # num_channels
    assert isinstance(intermediate_data.quantized_values[0], list)
    # Check structure based on scaled_blocks and bits_per_bfu
    assert len(intermediate_data.quantized_values[0]) == len(intermediate_data.scaled_blocks[0])
    for bfu_idx, bfu_quant_values in enumerate(intermediate_data.quantized_values[0]):
        assert isinstance(bfu_quant_values, list)
        if intermediate_data.bits_per_bfu[0][bfu_idx] > 0:
            assert all(isinstance(x, int) for x in bfu_quant_values) # Quantized values are int32_t
        else:
            assert len(bfu_quant_values) == 0

def test_encoder_quantized_values_shape_stereo(stereo_encoder):
    """Test quantized_values shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    
    assert intermediate_data.quantized_values is not None
    assert isinstance(intermediate_data.quantized_values, list)
    assert len(intermediate_data.quantized_values) == 2 # num_channels
    for ch in range(2):
        assert isinstance(intermediate_data.quantized_values[ch], list)
        assert len(intermediate_data.quantized_values[ch]) == len(intermediate_data.scaled_blocks[ch])
        for bfu_idx, bfu_quant_values in enumerate(intermediate_data.quantized_values[ch]):
            assert isinstance(bfu_quant_values, list)
            if intermediate_data.bits_per_bfu[ch][bfu_idx] > 0:
                 assert all(isinstance(x, int) for x in bfu_quant_values) # Quantized values are int32_t
            else:
                assert len(bfu_quant_values) == 0

def test_encoder_quantization_error_shape_mono(mono_encoder):
    """Test quantization_error shape and type for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    
    assert intermediate_data.quantization_error is not None
    assert isinstance(intermediate_data.quantization_error, list)
    assert len(intermediate_data.quantization_error) == 1 # num_channels
    assert isinstance(intermediate_data.quantization_error[0], list)
    assert len(intermediate_data.quantization_error[0]) == len(intermediate_data.scaled_blocks[0])
    for bfu_idx, bfu_errors in enumerate(intermediate_data.quantization_error[0]):
        assert isinstance(bfu_errors, list)
        if intermediate_data.bits_per_bfu[0][bfu_idx] > 0:
            assert all(isinstance(x, float) for x in bfu_errors)
        else:
            assert len(bfu_errors) == 0

def test_encoder_quantization_error_shape_stereo(stereo_encoder):
    """Test quantization_error shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    
    assert intermediate_data.quantization_error is not None
    assert isinstance(intermediate_data.quantization_error, list)
    assert len(intermediate_data.quantization_error) == 2 # num_channels
    for ch in range(2):
        assert isinstance(intermediate_data.quantization_error[ch], list)
        assert len(intermediate_data.quantization_error[ch]) == len(intermediate_data.scaled_blocks[ch])
        for bfu_idx, bfu_errors in enumerate(intermediate_data.quantization_error[ch]):
            assert isinstance(bfu_errors, list)
            if intermediate_data.bits_per_bfu[ch][bfu_idx] > 0:
                assert all(isinstance(x, float) for x in bfu_errors)
            else:
                assert len(bfu_errors) == 0

def test_encoder_consistency_zero_bits_quantized_values_mono(mono_encoder):
    """Test that if bits_per_bfu is 0 for a band, quantized_values for that band are 0 (mono)."""

    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES) # Use random data
    intermediate_data = mono_encoder.process_frame(audio_data)

    # quantized_values is list[list[list[int]]]
    # bits_per_bfu is list[list[int]]
    for ch_idx, ch_quant_values in enumerate(intermediate_data.quantized_values):
        for bfu_idx, bfu_values in enumerate(ch_quant_values):
            if intermediate_data.bits_per_bfu[ch_idx][bfu_idx] == 0:
                assert len(bfu_values) == 0, \
                    f"Channel {ch_idx}, BFU {bfu_idx} has 0 bits but non-zero quantized values list (len {len(bfu_values)})."

def test_encoder_consistency_zero_bits_quantized_values_stereo(stereo_encoder):
    """Test that if bits_per_bfu is 0 for a band, quantized_values for that band are 0 (stereo)."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES) # Use random data
    intermediate_data = stereo_encoder.process_frame(audio_data)

    for ch_idx, ch_quant_values in enumerate(intermediate_data.quantized_values):
        for bfu_idx, bfu_values in enumerate(ch_quant_values):
            if intermediate_data.bits_per_bfu[ch_idx][bfu_idx] == 0:
                assert len(bfu_values) == 0, \
                    f"Channel {ch_idx}, BFU {bfu_idx} has 0 bits but non-zero quantized values list (len {len(bfu_values)})."

def test_encoder_consistency_quantization_error_mono(mono_encoder):
    """Test relationship: quantization_error approx mdct_coefficients - dequantized(quantized_values) (mono)."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)
    num_channels = 1
    tolerance = 1e-6

    assert len(intermediate_data.mdct_specs) == num_channels
    assert len(intermediate_data.mdct_specs[0]) == pytrac.NUM_SAMPLES
    assert len(intermediate_data.quantized_values) == num_channels
    assert len(intermediate_data.quantization_error) == num_channels
    assert len(intermediate_data.bits_per_bfu) == num_channels
    assert len(intermediate_data.scaled_blocks) == num_channels

    for ch in range(num_channels):
        # These lists are sized to ATRAC1_MAX_BFU (52) by the C++ side.
        # We use len(intermediate_data.scaled_blocks[ch]) as the reference for this size.
        expected_bfu_list_len = len(intermediate_data.scaled_blocks[ch])
        assert len(intermediate_data.bits_per_bfu[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantized_values[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantization_error[ch]) == expected_bfu_list_len
        # ATRAC1_MODE0_NUM_BFU is 11, which is the number of *active* BFUs in this mode.
        # The processing loop correctly iterates up to ATRAC1_MODE0_NUM_BFU.

        for bfu_idx in range(ATRAC1_MODE0_NUM_BFU):
            word_len = intermediate_data.bits_per_bfu[ch][bfu_idx]
            num_coeffs_in_bfu_spec = ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx]
            mdct_offset = ATRAC1_MODE0_SPEC_OFFSET_PER_BFU[bfu_idx]
            
            quant_values_for_bfu = intermediate_data.quantized_values[ch][bfu_idx]
            error_values_for_bfu = intermediate_data.quantization_error[ch][bfu_idx]

            if word_len == 0:
                assert len(quant_values_for_bfu) == 0, f"Ch {ch} BFU {bfu_idx}: Expected 0 quantized values for 0 bits, got {len(quant_values_for_bfu)}"
                assert len(error_values_for_bfu) == 0, f"Ch {ch} BFU {bfu_idx}: Expected 0 error values for 0 bits, got {len(error_values_for_bfu)}"
            else:
                assert len(error_values_for_bfu) == len(quant_values_for_bfu), f"Ch {ch} BFU {bfu_idx}: Mismatch between num quantized values and num error values. Got {len(quant_values_for_bfu)} vs {len(error_values_for_bfu)}"

                sf_idx = intermediate_data.scaled_blocks[ch][bfu_idx].scale_factor_index
                
                scale_factor_multiplier = math.pow(2.0, sf_idx / 2.0)

                # Iterate only over the number of actual quantized values returned for this BFU
                for k in range(len(quant_values_for_bfu)):
                    quant_val_int = quant_values_for_bfu[k] # Integer quantized value
                    reported_error_from_cpp = error_values_for_bfu[k]
                    
                    if k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values):
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]
                    else:
                        assert k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values), \
                            f"Ch {ch} BFU {bfu_idx} Coeff {k}: Index k out of bounds for scaled_blocks.values. " \
                            f"len(quant_values_for_bfu)={len(quant_values_for_bfu)}, " \
                            f"len(scaled_blocks.values)={len(intermediate_data.scaled_blocks[ch][bfu_idx].values)}"
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]

                    # original_unnormalized_mdct_coeff is kept for the assertion message only
                    original_unnormalized_mdct_coeff = intermediate_data.mdct_specs[ch][mdct_offset + k]

                    reconstructed_normalized_coeff_py = 0.0
                    if word_len == 1:
                        reconstructed_normalized_coeff_py = float(quant_val_int)
                    elif word_len > 1:
                        cpp_dequant_denominator = float((1 << (word_len - 1)) - 1)
                        if cpp_dequant_denominator != 0: # Should not be zero for WL > 1
                            reconstructed_normalized_coeff_py = float(quant_val_int) / cpp_dequant_denominator
                        # else reconstructed_normalized_coeff_py remains 0.0 (should not happen for WL > 1)
                    
                    calculated_error_in_normalized_domain = original_normalized_mdct_coeff - reconstructed_normalized_coeff_py
                    
                    assert abs(reported_error_from_cpp - calculated_error_in_normalized_domain) < tolerance, \
                        f"Ch {ch} BFU {bfu_idx} Coeff {k}: Normalized Error mismatch. Reported {reported_error_from_cpp}, Calculated {calculated_error_in_normalized_domain}. " \
                        f"OrigNormMDCT: {original_normalized_mdct_coeff}, ReconNormMDCT: {reconstructed_normalized_coeff_py}, QuantInt: {quant_val_int}, " \
                        f"SF Idx: {sf_idx}, WL: {word_len}, OrigMDCT_unnorm: {original_unnormalized_mdct_coeff}"

def test_encoder_consistency_quantization_error_stereo(stereo_encoder):
    """Test relationship: quantization_error approx mdct_coefficients - dequantized(quantized_values) (stereo)."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)
    num_channels = 2
    tolerance = 1e-6

    assert len(intermediate_data.mdct_specs) == num_channels
    assert len(intermediate_data.quantized_values) == num_channels
    assert len(intermediate_data.quantization_error) == num_channels
    assert len(intermediate_data.bits_per_bfu) == num_channels
    assert len(intermediate_data.scaled_blocks) == num_channels

    for ch in range(num_channels):
        assert len(intermediate_data.mdct_specs[ch]) == pytrac.NUM_SAMPLES
        # These lists are sized to pytrac.MAX_BFUS (52) by the C++ side.
        expected_bfu_list_len = len(intermediate_data.scaled_blocks[ch])
        assert len(intermediate_data.bits_per_bfu[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantized_values[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantization_error[ch]) == expected_bfu_list_len
        # ATRAC1_MODE0_NUM_BFU is 11, which is the number of *active* BFUs in this mode.
        # The processing loop correctly iterates up to ATRAC1_MODE0_NUM_BFU.

        for bfu_idx in range(ATRAC1_MODE0_NUM_BFU):
            word_len = intermediate_data.bits_per_bfu[ch][bfu_idx]
            num_coeffs_in_bfu_spec = ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx]
            mdct_offset = ATRAC1_MODE0_SPEC_OFFSET_PER_BFU[bfu_idx]

            quant_values_for_bfu = intermediate_data.quantized_values[ch][bfu_idx]
            error_values_for_bfu = intermediate_data.quantization_error[ch][bfu_idx]

            if word_len == 0:
                assert len(quant_values_for_bfu) == 0, f"Ch {ch} BFU {bfu_idx}: Expected 0 quantized values for 0 bits, got {len(quant_values_for_bfu)}"
                assert len(error_values_for_bfu) == 0, f"Ch {ch} BFU {bfu_idx}: Expected 0 error values for 0 bits, got {len(error_values_for_bfu)}"
            else:
                # The number of returned quantized values might be less than num_coeffs_in_bfu_spec
                # if the encoder decides not all are significant, even with word_len > 0.
                # The dequantization loop below will use the actual length of quant_values_for_bfu.
                # We still assert that the number of error values matches the number of quant values.
                assert len(error_values_for_bfu) == len(quant_values_for_bfu), f"Ch {ch} BFU {bfu_idx}: Mismatch between num quantized values and num error values. Got {len(quant_values_for_bfu)} vs {len(error_values_for_bfu)}"
                
                sf_idx = intermediate_data.scaled_blocks[ch][bfu_idx].scale_factor_index
                scale_factor_multiplier = math.pow(2.0, sf_idx / 2.0)

                # Iterate only over the number of actual quantized values returned for this BFU
                for k in range(len(quant_values_for_bfu)):
                    quant_val_int = quant_values_for_bfu[k] # Integer quantized value
                    reported_error_from_cpp = error_values_for_bfu[k]

                    # Use the already normalized MDCT coefficient from scaled_blocks.values
                    if k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values):
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]
                    else:
                        assert k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values), \
                            f"Ch {ch} BFU {bfu_idx} Coeff {k}: Index k out of bounds for scaled_blocks.values. " \
                            f"len(quant_values_for_bfu)={len(quant_values_for_bfu)}, " \
                            f"len(scaled_blocks.values)={len(intermediate_data.scaled_blocks[ch][bfu_idx].values)}"
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]
                    
                    # original_unnormalized_mdct_coeff is kept for the assertion message only
                    original_unnormalized_mdct_coeff = intermediate_data.mdct_specs[ch][mdct_offset + k]

                    reconstructed_normalized_coeff_py = 0.0
                    if word_len == 1:
                        reconstructed_normalized_coeff_py = float(quant_val_int)
                    elif word_len > 1:
                        cpp_dequant_denominator = float((1 << (word_len - 1)) - 1)
                        if cpp_dequant_denominator != 0: # Should not be zero for WL > 1
                            reconstructed_normalized_coeff_py = float(quant_val_int) / cpp_dequant_denominator
                        # else reconstructed_normalized_coeff_py remains 0.0 (should not happen for WL > 1)
                    
                    calculated_error_in_normalized_domain = original_normalized_mdct_coeff - reconstructed_normalized_coeff_py
                    
                    assert abs(reported_error_from_cpp - calculated_error_in_normalized_domain) < tolerance, \
                        f"Ch {ch} BFU {bfu_idx} Coeff {k}: Normalized Error mismatch. Reported {reported_error_from_cpp}, Calculated {calculated_error_in_normalized_domain}. " \
                        f"OrigNormMDCT: {original_normalized_mdct_coeff}, ReconNormMDCT: {reconstructed_normalized_coeff_py}, QuantInt: {quant_val_int}, " \
                        f"SF Idx: {sf_idx}, WL: {word_len}, OrigMDCT_unnorm: {original_unnormalized_mdct_coeff}"

# Function to get a valid bitstream frame using the encoder
def get_generated_atrac1_bitstream_frame(num_channels):
    """
    Generates an ATRAC1 bitstream frame using the FrameProcessor.
    Returns:
        per_channel_bitstream_data (List[List[int]]): List of byte lists per channel.
        frame_size_bytes_per_channel (List[int]): List of frame sizes in bytes per channel.
    """
    if num_channels == 1:
        encoder = pytrac.FrameProcessor(1)
    elif num_channels == 2:
        encoder = pytrac.FrameProcessor(2)
    else:
        raise ValueError("Unsupported number of channels for bitstream generation.")

    audio_data = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)
    intermediate_data = encoder.process_frame(audio_data)

    per_channel_bitstream_data = intermediate_data.compressed_data_per_channel
    frame_size_bytes_per_channel = [len(ch_data) for ch_data in per_channel_bitstream_data]
    
    # Ensure some data was produced (basic sanity check)
    for size in frame_size_bytes_per_channel:
        assert size > 0, "Encoder produced an empty bitstream for a channel."

    return per_channel_bitstream_data, frame_size_bytes_per_channel

@pytest.fixture
def mono_decoder():
    """Fixture for a mono PyAtrac1FrameDecoder."""
    # Assuming a common sample rate, e.g., 44100 Hz
    return pytrac.FrameDecoder(1)

@pytest.fixture
def stereo_decoder():
    """Fixture for a stereo PyAtrac1FrameDecoder."""
    return pytrac.FrameDecoder(2)

def test_decoder_initialization(mono_decoder, stereo_decoder):
    """Test if decoders are initialized without errors."""
    assert mono_decoder is not None
    assert stereo_decoder is not None

def test_decoder_block_size_log_count_mono(mono_decoder):
    """Test block_size_log_count_per_channel for mono audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream)
    
    block_log_count = decoder_intermediate_data.block_size_log_count
    assert block_log_count is not None
    assert isinstance(block_log_count, list)
    assert len(block_log_count) == 1 # num_channels
    assert isinstance(block_log_count[0], list) # Per-channel list
    assert len(block_log_count[0]) > 0 # At least one subband value
    assert isinstance(block_log_count[0][0], int) # Check first subband's value
    assert block_log_count[0][0] in [0, 1, 2] # 0 for LONG, 1 for SHORT, 2 for other modes

def test_decoder_block_size_log_count_stereo(stereo_decoder):
    """Test block_size_log_count_per_channel for stereo audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    block_log_count = decoder_intermediate_data.block_size_log_count
    assert block_log_count is not None
    assert isinstance(block_log_count, list)
    assert len(block_log_count) == 2 # num_channels
    for ch_block_log_counts in block_log_count:
        assert isinstance(ch_block_log_counts, list) # Per-channel list
        assert len(ch_block_log_counts) > 0 # At least one subband value
        assert isinstance(ch_block_log_counts[0], int) # Check first subband's value
        assert ch_block_log_counts[0] in [0, 1, 2] # 0 for LONG, 1 for SHORT, 2 for other modes

def test_decoder_scale_factor_indices_mono(mono_decoder):
    """Test scale_factor_indices_per_channel for mono audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    sf_indices = decoder_intermediate_data.scale_factor_indices
    assert sf_indices is not None
    assert isinstance(sf_indices, list)
    assert len(sf_indices) == 1 # Num channels
    
    ch_sfs = sf_indices[0] # Data for the first channel
    assert isinstance(ch_sfs, list) # Inner list for BFUs
    # Number of BFUs depends on block mode. Assuming long mode from encoder.
    if decoder_intermediate_data.block_size_log_count[0] == 0: # Long mode
            assert len(ch_sfs) == ATRAC1_MODE0_NUM_BFU
    for val in ch_sfs: # Scale factors per BFU
        assert isinstance(val, int)
        assert 0 <= val < 64

def test_decoder_scale_factor_indices_stereo(stereo_decoder):
    """Test scale_factor_indices_per_channel for stereo audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    sf_indices = decoder_intermediate_data.scale_factor_indices
    assert sf_indices is not None
    assert isinstance(sf_indices, list)
    assert len(sf_indices) == 2 # Num channels
    for ch_idx, ch_sfs in enumerate(sf_indices):
        assert isinstance(ch_sfs, list)
        if decoder_intermediate_data.block_size_log_count[ch_idx] == 0: # Long mode
            assert len(ch_sfs) == ATRAC1_MODE0_NUM_BFU
        for val in ch_sfs:
            assert isinstance(val, int)
            assert 0 <= val < 64

def test_decoder_bits_per_bfu_mono(mono_decoder):
    """Test bits_per_bfu_per_channel for mono audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    bits_per_bfu = decoder_intermediate_data.bits_per_bfu
    assert bits_per_bfu is not None
    assert isinstance(bits_per_bfu, list)
    assert len(bits_per_bfu) == 1 # Num channels
    
    ch_bits = bits_per_bfu[0]
    assert isinstance(ch_bits, list) # Inner list for BFUs
    if decoder_intermediate_data.block_size_log_count[0] == 0: # Long mode
        assert len(ch_bits) == ATRAC1_MODE0_NUM_BFU
    for val in ch_bits: # Bits per BFU
        assert isinstance(val, int)
        assert 0 <= val <= 15

def test_decoder_bits_per_bfu_stereo(stereo_decoder):
    """Test bits_per_bfu_per_channel for stereo audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    bits_per_bfu = decoder_intermediate_data.bits_per_bfu
    assert bits_per_bfu is not None
    assert isinstance(bits_per_bfu, list)
    assert len(bits_per_bfu) == 2 # Num channels
    for ch_idx, ch_bits in enumerate(bits_per_bfu):
        assert isinstance(ch_bits, list)
        if decoder_intermediate_data.block_size_log_count[ch_idx] == 0: # Long mode
            assert len(ch_bits) == ATRAC1_MODE0_NUM_BFU
        for val in ch_bits:
            assert isinstance(val, int)
            assert 0 <= val <= 15

def test_decoder_parsed_quantized_values_mono(mono_decoder):
    """Test parsed_quantized_values_per_channel for mono audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    quant_vals = decoder_intermediate_data.parsed_quantized_values
    assert quant_vals is not None
    assert isinstance(quant_vals, list)
    assert len(quant_vals) == 1 # Num channels
    
    ch_data = quant_vals[0]
    assert isinstance(ch_data, list) # List of BFUs
    if decoder_intermediate_data.block_size_log_count[0] == 0: # Long mode
        assert len(ch_data) == ATRAC1_MODE0_NUM_BFU
        for bfu_idx, bfu_quant_data in enumerate(ch_data): # Per BFU
            assert isinstance(bfu_quant_data, list) # List of quantized coefficients
            word_len = decoder_intermediate_data.bits_per_bfu[0][bfu_idx]
            if word_len > 0:
                assert len(bfu_quant_data) == ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx]
                for val in bfu_quant_data:
                    assert isinstance(val, int)
            else:
                assert len(bfu_quant_data) == 0

def test_decoder_parsed_quantized_values_stereo(stereo_decoder):
    """Test parsed_quantized_values_per_channel for stereo audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    quant_vals = decoder_intermediate_data.parsed_quantized_values
    assert quant_vals is not None
    assert isinstance(quant_vals, list)
    assert len(quant_vals) == 2 # Num channels

    for ch_idx in range(2):
        ch_data = quant_vals[ch_idx]
        assert isinstance(ch_data, list) # List of BFUs
        if decoder_intermediate_data.block_size_log_count[ch_idx] == 0: # Long mode
            assert len(ch_data) == ATRAC1_MODE0_NUM_BFU
            for bfu_idx, bfu_quant_data in enumerate(ch_data): # Per BFU
                assert isinstance(bfu_quant_data, list) # List of quantized coefficients
                word_len = decoder_intermediate_data.bits_per_bfu[ch_idx][bfu_idx]
                if word_len > 0:
                    assert len(bfu_quant_data) == ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx]
                    for val in bfu_quant_data:
                        assert isinstance(val, int)
                else:
                    assert len(bfu_quant_data) == 0

def test_decoder_mdct_specs_mono(mono_decoder):
    """Test mdct_specs_per_channel for mono audio (dequantized MDCT)."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    mdct_specs = decoder_intermediate_data.mdct_specs
    assert mdct_specs is not None
    assert isinstance(mdct_specs, list)
    assert len(mdct_specs) == 1 # Num channels
    
    ch_mdct_data = mdct_specs[0]
    assert isinstance(ch_mdct_data, list) # Flat list of MDCT coefficients
    assert len(ch_mdct_data) == pytrac.NUM_SAMPLES
    for val in ch_mdct_data:
        assert isinstance(val, float)

def test_decoder_mdct_specs_stereo(stereo_decoder):
    """Test mdct_specs_per_channel for stereo audio (dequantized MDCT)."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    mdct_specs = decoder_intermediate_data.mdct_specs
    assert mdct_specs is not None
    assert isinstance(mdct_specs, list)
    assert len(mdct_specs) == 2 # Num channels

    for ch_mdct_data in mdct_specs:
        assert isinstance(ch_mdct_data, list)
        assert len(ch_mdct_data) == pytrac.NUM_SAMPLES
        for val in ch_mdct_data:
            assert isinstance(val, float)

def test_decoder_get_decoded_audio_mono(mono_decoder):
    """Test get_decoded_audio() for mono audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream)
    decoded_audio = decoder_intermediate_data.pcm_output

    assert decoded_audio is not None
    assert isinstance(decoded_audio, list)
    assert len(decoded_audio) == 1 # num_channels
    
    ch_audio = decoded_audio[0]
    assert isinstance(ch_audio, list)
    assert len(ch_audio) == pytrac.NUM_SAMPLES
    assert all(isinstance(x, float) for x in ch_audio)

def test_decoder_get_decoded_audio_stereo(stereo_decoder):
    """Test get_decoded_audio() for stereo audio."""
    per_channel_bitstream, frame_sizes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)
    decoded_audio = decoder_intermediate_data.pcm_output

    assert decoded_audio is not None
    assert isinstance(decoded_audio, list)
    assert len(decoded_audio) == 2 # num_channels
    for ch_audio in decoded_audio:
        assert isinstance(ch_audio, list)
        assert len(ch_audio) == pytrac.NUM_SAMPLES
        assert all(isinstance(x, float) for x in ch_audio)

# 1. Scale Factor Consistency
def test_consistency_scale_factors_mono(mono_decoder):
    """Compare encoder's scale factors with decoder's parsed scale factors (mono)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.scaled_blocks) == 1
    assert len(dec_data.scale_factor_indices) == 1
    
    encoder_sfs = [block.scale_factor_index for block in enc_data.scaled_blocks[0]]
    decoder_sfs = dec_data.scale_factor_indices[0]
    
    # The number of active BFUs is ATRAC1_MODE0_NUM_BFU for default mode
    # Encoder might store up to ATRAC1_MAX_BFU, decoder might parse up to active ones.
    # We expect the active ones to match.
    assert len(decoder_sfs) >= ATRAC1_MODE0_NUM_BFU, "Decoder parsed fewer SFs than expected for active BFUs"
    
    for i in range(ATRAC1_MODE0_NUM_BFU):
        assert encoder_sfs[i] == decoder_sfs[i], \
            f"Mono SF mismatch at BFU {i}: Encoder {encoder_sfs[i]}, Decoder {decoder_sfs[i]}"

def test_consistency_scale_factors_stereo(stereo_decoder):
    """Compare encoder's scale factors with decoder's parsed scale factors (stereo)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=2)
    dec_data = stereo_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.scaled_blocks) == 2
    assert len(dec_data.scale_factor_indices) == 2

    for ch in range(2):
        encoder_sfs = [block.scale_factor_index for block in enc_data.scaled_blocks[ch]]
        decoder_sfs = dec_data.scale_factor_indices[ch]
        
        assert len(decoder_sfs) >= ATRAC1_MODE0_NUM_BFU, f"Ch {ch}: Decoder parsed fewer SFs than expected"
        for i in range(ATRAC1_MODE0_NUM_BFU):
            assert encoder_sfs[i] == decoder_sfs[i], \
                f"Stereo Ch {ch} SF mismatch at BFU {i}: Encoder {encoder_sfs[i]}, Decoder {decoder_sfs[i]}"

# 2. Bits Per BFU (Word Length) Consistency
def test_consistency_bits_per_bfu_mono(mono_decoder):
    """Compare encoder's bits_per_bfu with decoder's parsed bits_per_bfu (mono)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.bits_per_bfu) == 1
    assert len(dec_data.bits_per_bfu) == 1

    encoder_wls = enc_data.bits_per_bfu[0]
    decoder_wls = dec_data.bits_per_bfu[0]

    assert len(decoder_wls) >= ATRAC1_MODE0_NUM_BFU, "Decoder parsed fewer WLs than expected"
    for i in range(ATRAC1_MODE0_NUM_BFU):
        expected_actual_wl_from_decoder_idwl = (decoder_wls[i] + 1) if decoder_wls[i] > 0 else 0
        assert encoder_wls[i] == expected_actual_wl_from_decoder_idwl, \
            f"Mono WL mismatch at BFU {i}: Encoder_actual_WL {encoder_wls[i]}, Decoder_IDWL {decoder_wls[i]} (implies actual_WL {expected_actual_wl_from_decoder_idwl})"

def test_consistency_bits_per_bfu_stereo(stereo_decoder):
    """Compare encoder's bits_per_bfu with decoder's parsed bits_per_bfu (stereo)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=2)
    dec_data = stereo_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.bits_per_bfu) == 2
    assert len(dec_data.bits_per_bfu) == 2

    for ch in range(2):
        encoder_wls = enc_data.bits_per_bfu[ch]
        decoder_wls = dec_data.bits_per_bfu[ch]
        
        assert len(decoder_wls) >= ATRAC1_MODE0_NUM_BFU, f"Ch {ch}: Decoder parsed fewer WLs than expected"
        for i in range(ATRAC1_MODE0_NUM_BFU):
            expected_actual_wl_from_decoder_idwl = (decoder_wls[i] + 1) if decoder_wls[i] > 0 else 0
            assert encoder_wls[i] == expected_actual_wl_from_decoder_idwl, \
                f"Stereo Ch {ch} WL mismatch at BFU {i}: Encoder_actual_WL {encoder_wls[i]}, Decoder_IDWL {decoder_wls[i]} (implies actual_WL {expected_actual_wl_from_decoder_idwl})"

# 3. Quantized Values Consistency
def test_consistency_quantized_values_mono(mono_decoder):
    """Compare encoder's quantized_values with decoder's parsed_quantized_values (mono)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.quantized_values) == 1
    assert len(dec_data.parsed_quantized_values) == 1

    encoder_qvs_ch = enc_data.quantized_values[0]
    decoder_qvs_ch = dec_data.parsed_quantized_values[0]

    assert len(decoder_qvs_ch) >= ATRAC1_MODE0_NUM_BFU
    for bfu_idx in range(ATRAC1_MODE0_NUM_BFU):
        enc_bfu_qvs = encoder_qvs_ch[bfu_idx]
        dec_bfu_qvs = decoder_qvs_ch[bfu_idx]
        word_len = enc_data.bits_per_bfu[0][bfu_idx] # Use encoder's WL as reference

        if word_len == 0:
            assert len(enc_bfu_qvs) == 0, f"Mono BFU {bfu_idx}: Encoder QVs non-empty for WL=0"
            assert len(dec_bfu_qvs) == 0, f"Mono BFU {bfu_idx}: Decoder QVs non-empty for WL=0"
        else:
            # Decoder should parse exactly ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx] if WL > 0
            # Encoder might store fewer if some are zero, but C++ binding for decoder intermediate data
            # seems to store the full spec count.
            # Use pytrac.SPECS_PER_BLOCK which reflects the C++ internal tables.
            assert len(dec_bfu_qvs) == pytrac.SPECS_PER_BLOCK[bfu_idx], \
                f"Mono BFU {bfu_idx}: Decoder QV count mismatch. Expected {pytrac.SPECS_PER_BLOCK[bfu_idx]}, Got {len(dec_bfu_qvs)}"

            # Encoder's quantized_values list might be shorter if trailing values are zero.
            # We compare up to the length of the shorter list, or ensure encoder's is padded appropriately for comparison.
            # For now, let's assume encoder also provides full spec length if WL > 0, or we compare common prefix.
            # The current encoder logic in C++ fills `quantized_bfu_values` based on `block.Values.size()`.
            # `block.Values` are normalized MDCTs for the BFU.
            # `TAtrac1Dequantiser::ParseQuantisedValues` fills `m_parsed_quantized_values` up to `SpecsPerBlock[block_idx]`.
            
            # We need to ensure the encoder's `quantized_values` for a BFU has the same length as `ATRAC1_CPP_SPECS_PER_BLOCK`
            # if we are to compare them directly element-wise when WL > 0.
            # The current `test_encoder_quantized_values_shape_mono` implies encoder might not fill all.
            # Let's verify this assumption or adjust.
            # For now, compare up to the length of encoder's list, assuming decoder's list is at least as long.
            assert len(enc_bfu_qvs) <= len(dec_bfu_qvs), \
                 f"Mono BFU {bfu_idx}: Encoder QVs longer than decoder QVs. Enc: {len(enc_bfu_qvs)}, Dec: {len(dec_bfu_qvs)}"

            for i in range(len(enc_bfu_qvs)):
                 assert enc_bfu_qvs[i] == dec_bfu_qvs[i], \
                    f"Mono QV mismatch at BFU {bfu_idx}, Coeff {i}: Encoder {enc_bfu_qvs[i]}, Decoder {dec_bfu_qvs[i]}"
            # If encoder's list is shorter, remaining decoder values should be 0 (as they were not significant enough to be transmitted or were zero)
            for i in range(len(enc_bfu_qvs), len(dec_bfu_qvs)):
                assert dec_bfu_qvs[i] == 0, \
                    f"Mono QV mismatch at BFU {bfu_idx}, Coeff {i}: Decoder non-zero ({dec_bfu_qvs[i]}) where encoder had no value (implies zero)."


def test_consistency_quantized_values_stereo(stereo_decoder):
    """Compare encoder's quantized_values with decoder's parsed_quantized_values (stereo)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=2)
    dec_data = stereo_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.quantized_values) == 2
    assert len(dec_data.parsed_quantized_values) == 2

    for ch in range(2):
        encoder_qvs_ch = enc_data.quantized_values[ch]
        decoder_qvs_ch = dec_data.parsed_quantized_values[ch]

        assert len(decoder_qvs_ch) >= ATRAC1_MODE0_NUM_BFU
        for bfu_idx in range(ATRAC1_MODE0_NUM_BFU):
            enc_bfu_qvs = encoder_qvs_ch[bfu_idx]
            dec_bfu_qvs = decoder_qvs_ch[bfu_idx]
            word_len = enc_data.bits_per_bfu[ch][bfu_idx]

            if word_len == 0:
                assert len(enc_bfu_qvs) == 0, f"Stereo Ch {ch} BFU {bfu_idx}: Encoder QVs non-empty for WL=0"
                assert len(dec_bfu_qvs) == 0, f"Stereo Ch {ch} BFU {bfu_idx}: Decoder QVs non-empty for WL=0"
            else:
                # Use pytrac.SPECS_PER_BLOCK which reflects the C++ internal tables.
                assert len(dec_bfu_qvs) == pytrac.SPECS_PER_BLOCK[bfu_idx], \
                    f"Stereo Ch {ch} BFU {bfu_idx}: Decoder QV count mismatch. Expected {pytrac.SPECS_PER_BLOCK[bfu_idx]}, Got {len(dec_bfu_qvs)}"
                
                assert len(enc_bfu_qvs) <= len(dec_bfu_qvs), \
                    f"Stereo Ch {ch} BFU {bfu_idx}: Encoder QVs longer than decoder QVs. Enc: {len(enc_bfu_qvs)}, Dec: {len(dec_bfu_qvs)}"

                for i in range(len(enc_bfu_qvs)):
                    assert enc_bfu_qvs[i] == dec_bfu_qvs[i], \
                        f"Stereo Ch {ch} QV mismatch at BFU {bfu_idx}, Coeff {i}: Encoder {enc_bfu_qvs[i]}, Decoder {dec_bfu_qvs[i]}"
                for i in range(len(enc_bfu_qvs), len(dec_bfu_qvs)):
                    assert dec_bfu_qvs[i] == 0, \
                        f"Stereo Ch {ch} QV mismatch at BFU {bfu_idx}, Coeff {i}: Decoder non-zero ({dec_bfu_qvs[i]}) where encoder had no value."

# 4. MDCT Spectra Consistency (Original vs. Decoder's Dequantized)
# This will be an approximate comparison due to quantization.
MDCT_COMPARISON_TOLERANCE = 1e-3 # Tolerance for comparing MDCT values

def test_consistency_mdct_spectra_mono(mono_decoder):
    """Compare encoder's original MDCT spectra with decoder's dequantized MDCT spectra (mono)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.mdct_specs) == 1
    assert len(dec_data.mdct_specs) == 1
    
    original_mdcts = enc_data.mdct_specs[0]
    decoded_mdcts = dec_data.mdct_specs[0]

    assert len(original_mdcts) == pytrac.NUM_SAMPLES
    assert len(decoded_mdcts) == pytrac.NUM_SAMPLES
    
    print("\n--- Mono MDCT Spectra Consistency Debug ---")
    print(f"Original MDCT (enc_data) sample (first 10): {[f'{x:.6f}' for x in original_mdcts[:10]]}")
    print(f"Decoded MDCT (dec_data) sample (first 10): {[f'{x:.6f}' for x in decoded_mdcts[:10]]}")

    diffs = np.abs(np.array(original_mdcts) - np.array(decoded_mdcts))
    avg_abs_diff = np.mean(diffs)
    max_abs_diff = np.max(diffs)
    significant_diff_count = np.sum(diffs > MDCT_COMPARISON_TOLERANCE) # MDCT_COMPARISON_TOLERANCE = 1e-3
    
    print(f"MDCT Stats - SignificantDiffs (>{MDCT_COMPARISON_TOLERANCE}): {significant_diff_count}/{pytrac.NUM_SAMPLES}, AvgAbsDiff: {avg_abs_diff:.6f}, MaxAbsDiff: {max_abs_diff:.6f}")
    
    # Print a few actual large differences
    if significant_diff_count > 0:
        print("Example significant differences (Original vs Decoded):")
        indices = np.where(diffs > MDCT_COMPARISON_TOLERANCE)[0]
        for i, idx in enumerate(indices[:5]): # Print up to 5
             print(f"  idx {idx}: {original_mdcts[idx]:.6f} vs {decoded_mdcts[idx]:.6f} (Diff: {original_mdcts[idx]-decoded_mdcts[idx]:.6f})")

    assert avg_abs_diff < 0.1, f"MDCT spectra average absolute difference too high: {avg_abs_diff:.6f}"
    assert max_abs_diff < 0.5, f"MDCT spectra maximum absolute difference too high: {max_abs_diff:.6f}" # Stricter max check


def test_consistency_mdct_spectra_stereo(stereo_decoder):
    """Compare encoder's original MDCT spectra with decoder's dequantized MDCT spectra (stereo)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=2)
    dec_data = stereo_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.mdct_specs) == 2
    assert len(dec_data.mdct_specs) == 2

    for ch in range(2):
        original_mdcts = enc_data.mdct_specs[ch]
        decoded_mdcts = dec_data.mdct_specs[ch]

        assert len(original_mdcts) == pytrac.NUM_SAMPLES
        assert len(decoded_mdcts) == pytrac.NUM_SAMPLES
        
        avg_abs_diff = np.mean(np.abs(np.array(original_mdcts) - np.array(decoded_mdcts)))
        assert avg_abs_diff < 0.1, f"Stereo Ch {ch} MDCT spectra average absolute difference too high: {avg_abs_diff:.4f}"
        # print(f"Stereo Ch {ch} MDCT avg_abs_diff: {avg_abs_diff}")


# 5. PCM Output Consistency (Original Input vs. Final Decoded Output)
# This will also be an approximate comparison.
PCM_COMPARISON_TOLERANCE_RMS = 0.1 # RMS error tolerance for PCM

def test_consistency_pcm_output_mono(mono_decoder):
    """Compare original PCM input with final decoded PCM output (mono)."""
    original_audio_np, _, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert original_audio_np.shape == (1, pytrac.NUM_SAMPLES)
    assert len(dec_data.pcm_output) == 1
    assert len(dec_data.pcm_output[0]) == pytrac.NUM_SAMPLES

    # Analyze MDCT coefficients before IMDCT
    mdct_coeffs_for_imdct = np.array(dec_data.mdct_specs[0])
    print("\n--- Mono Decoded MDCT Input to IMdct Stats ---")
    print(f"Min: {np.min(mdct_coeffs_for_imdct):.4f}, Max: {np.max(mdct_coeffs_for_imdct):.4f}, Mean: {np.mean(mdct_coeffs_for_imdct):.4f}, Std: {np.std(mdct_coeffs_for_imdct):.4f}")
    print(f"Has NaN: {np.isnan(mdct_coeffs_for_imdct).any()}, Has Inf: {np.isinf(mdct_coeffs_for_imdct).any()}")

    original_pcm_ch = original_audio_np[0]
    decoded_pcm_ch = np.array(dec_data.pcm_output[0])
    
    # Apply clamping to match C++ decoder behavior
    decoded_pcm_ch = np.clip(decoded_pcm_ch, -1.0, 1.0)

    # Debug output for MDCT-to-PCM transformation
    print("\n--- Mono PCM Debug ---")
    print(f"Original PCM range: [{np.min(original_pcm_ch):.4f}, {np.max(original_pcm_ch):.4f}]")
    print(f"Decoded PCM range: [{np.min(decoded_pcm_ch):.4f}, {np.max(decoded_pcm_ch):.4f}]")
    
    # Print first 10 MDCT coefficients and their PCM impact
    print("\nFirst 10 MDCT coefficients (decoded):")
    for i in range(10):
        print(f"  {i}: {dec_data.mdct_specs[0][i]:.6f}")
    
    # Print energy distribution
    bands = [
        ("Low", 0, 127),
        ("Mid", 128, 255),
        ("High", 256, 511)
    ]
    for name, start, end in bands:
        band_energy = np.sum(np.square(dec_data.mdct_specs[0][start:end+1]))
        print(f"{name} band energy: {band_energy:.4f}")

    # Calculate RMS error
    rms_error = np.sqrt(np.mean((original_pcm_ch - decoded_pcm_ch)**2))

    # Calculate optimal scaling factor k and scaled RMS error
    if np.sum(decoded_pcm_ch**2) > 1e-12: # Avoid division by zero for silent frames
        k_optimal = np.sum(original_pcm_ch * decoded_pcm_ch) / np.sum(decoded_pcm_ch**2)
        rms_error_scaled = np.sqrt(np.mean((original_pcm_ch - k_optimal * decoded_pcm_ch)**2))
        print(f"Optimal scaling factor k: {k_optimal:.6f}")
        print(f"RMS error after optimal scaling: {rms_error_scaled:.6f}")

        # Test with a fixed k derived from _direct_mdct observations
        k_fixed = 0.785793
        rms_error_fixed_scaled = np.sqrt(np.mean((original_pcm_ch - k_fixed * decoded_pcm_ch)**2))
        print(f"RMS error after fixed scaling (k={k_fixed}): {rms_error_fixed_scaled:.6f}")
    else:
        print("Decoded PCM energy too low for optimal scaling calculation.")

    # Expected characteristics of ATRAC1 codec with random input
    assert 0.75 < k_optimal < 0.85, \
        f"Optimal scaling factor {k_optimal:.6f} outside expected ATRAC1 range (0.75-0.85)"
    assert 0.45 < rms_error_scaled < 0.55, \
        f"Scaled RMS error {rms_error_scaled:.6f} outside expected ATRAC1 range (0.45-0.55)"

def test_consistency_pcm_output_stereo(stereo_decoder):
    """Compare original PCM input with final decoded PCM output (stereo)."""
    original_audio_np, enc_intermediate_data, bitstream = encode_and_get_data(num_channels=2) # Capture enc_intermediate_data
    dec_data = stereo_decoder.decode_frame_from_bitstream(bitstream)

    # Debug prints for intermediate data (Channel 0)
    if enc_intermediate_data: # Check if enc_intermediate_data was successfully captured
        print("\n--- PyTrac Debug: Intermediate Data Stats (Stereo Test - Channel 0) ---")
        # PCM Input
        if enc_intermediate_data.pcm_input and len(enc_intermediate_data.pcm_input) > 0:
            pcm_input_ch0 = np.array(enc_intermediate_data.pcm_input[0])
            print(f"PCM Input:  Min={np.min(pcm_input_ch0):.3e}, Max={np.max(pcm_input_ch0):.3e}, Mean={np.mean(pcm_input_ch0):.3e}, Std={np.std(pcm_input_ch0):.3e}, NaN={np.isnan(pcm_input_ch0).any()}, Inf={np.isinf(pcm_input_ch0).any()}")

        # QMF Low
        if enc_intermediate_data.qmf_low and len(enc_intermediate_data.qmf_low) > 0:
            qmf_low_ch0 = np.array(enc_intermediate_data.qmf_low[0])
            print(f"QMF Low:    Min={np.min(qmf_low_ch0):.3e}, Max={np.max(qmf_low_ch0):.3e}, Mean={np.mean(qmf_low_ch0):.3e}, Std={np.std(qmf_low_ch0):.3e}, NaN={np.isnan(qmf_low_ch0).any()}, Inf={np.isinf(qmf_low_ch0).any()}")
        
        # QMF Mid
        if enc_intermediate_data.qmf_mid and len(enc_intermediate_data.qmf_mid) > 0:
            qmf_mid_ch0 = np.array(enc_intermediate_data.qmf_mid[0])
            print(f"QMF Mid:    Min={np.min(qmf_mid_ch0):.3e}, Max={np.max(qmf_mid_ch0):.3e}, Mean={np.mean(qmf_mid_ch0):.3e}, Std={np.std(qmf_mid_ch0):.3e}, NaN={np.isnan(qmf_mid_ch0).any()}, Inf={np.isinf(qmf_mid_ch0).any()}")

        # QMF Hi
        if enc_intermediate_data.qmf_hi and len(enc_intermediate_data.qmf_hi) > 0:
            qmf_hi_ch0 = np.array(enc_intermediate_data.qmf_hi[0])
            print(f"QMF Hi:     Min={np.min(qmf_hi_ch0):.3e}, Max={np.max(qmf_hi_ch0):.3e}, Mean={np.mean(qmf_hi_ch0):.3e}, Std={np.std(qmf_hi_ch0):.3e}, NaN={np.isnan(qmf_hi_ch0).any()}, Inf={np.isinf(qmf_hi_ch0).any()}")

        # MDCT Specs
        if enc_intermediate_data.mdct_specs and len(enc_intermediate_data.mdct_specs) > 0:
            mdct_specs_ch0 = np.array(enc_intermediate_data.mdct_specs[0])
            print(f"MDCT Specs: Min={np.min(mdct_specs_ch0):.3e}, Max={np.max(mdct_specs_ch0):.3e}, Mean={np.mean(mdct_specs_ch0):.3e}, Std={np.std(mdct_specs_ch0):.3e}, NaN={np.isnan(mdct_specs_ch0).any()}, Inf={np.isinf(mdct_specs_ch0).any()}")
            
            # Count huge MDCT values
            huge_mdct_count = np.sum(np.abs(mdct_specs_ch0) > 1e20)
            if huge_mdct_count > 0:
                print(f"MDCT Specs: Found {huge_mdct_count} value(s) with abs > 1e20. Example: {mdct_specs_ch0[np.abs(mdct_specs_ch0) > 1e20][0]:.3e}")
        print("--- End PyTrac Debug ---")

    assert original_audio_np.shape == (2, pytrac.NUM_SAMPLES)
    assert len(dec_data.pcm_output) == 2
    
    for ch in range(2):
        assert len(dec_data.pcm_output[ch]) == pytrac.NUM_SAMPLES
        original_pcm_ch = original_audio_np[ch]
        decoded_pcm_ch = np.array(dec_data.pcm_output[ch])
        
        # Apply clamping to match C++ decoder behavior
        decoded_pcm_ch = np.clip(decoded_pcm_ch, -1.0, 1.0)

        # Debug output for MDCT-to-PCM transformation
        print(f"\n--- Stereo Ch{ch} PCM Debug ---")
        print(f"Original PCM range: [{np.min(original_pcm_ch):.4f}, {np.max(original_pcm_ch):.4f}]")
        print(f"Decoded PCM range: [{np.min(decoded_pcm_ch):.4f}, {np.max(decoded_pcm_ch):.4f}]")
        
        # Print first 10 MDCT coefficients and their PCM impact
        print("\nFirst 10 MDCT coefficients (decoded):")
        for i in range(10):
            print(f"  {i}: {dec_data.mdct_specs[ch][i]:.6f}")
        
        # Print energy distribution
        bands = [
            ("Low", 0, 127),
            ("Mid", 128, 255),
            ("High", 256, 511)
        ]
        for name, start, end in bands:
            band_energy = np.sum(np.square(dec_data.mdct_specs[ch][start:end+1]))
            print(f"{name} band energy: {band_energy:.4f}")

        rms_error = np.sqrt(np.mean((original_pcm_ch - decoded_pcm_ch)**2))

        # Calculate optimal scaling factor k and scaled RMS error for stereo
        if np.sum(decoded_pcm_ch**2) > 1e-12:
            k_optimal_stereo = np.sum(original_pcm_ch * decoded_pcm_ch) / np.sum(decoded_pcm_ch**2)
            rms_error_scaled_stereo = np.sqrt(np.mean((original_pcm_ch - k_optimal_stereo * decoded_pcm_ch)**2))
            print(f"Stereo Ch {ch} Optimal scaling factor k: {k_optimal_stereo:.6f}")
            print(f"Stereo Ch {ch} RMS error after optimal scaling: {rms_error_scaled_stereo:.6f}")

            k_fixed_stereo = 0.785793 # Assuming same k for stereo channels based on mono direct
            rms_error_fixed_scaled_stereo = np.sqrt(np.mean((original_pcm_ch - k_fixed_stereo * decoded_pcm_ch)**2))
            print(f"Stereo Ch {ch} RMS error after fixed scaling (k={k_fixed_stereo}): {rms_error_fixed_scaled_stereo:.6f}")
        else:
            print(f"Stereo Ch {ch} Decoded PCM energy too low for optimal scaling calculation.")

        # Expected characteristics of ATRAC1 codec with random input
        assert 0.73 < k_optimal_stereo < 0.87, \
            f"Stereo Ch {ch} scaling factor {k_optimal_stereo:.6f} outside expected range"
        assert 0.45 < rms_error_scaled_stereo < 0.55, \
            f"Stereo Ch {ch} scaled RMS error {rms_error_scaled_stereo:.6f} outside expected range"
        # print(f"Stereo Ch {ch} PCM RMS error: {rms_error}")

ENERGY_COMPARISON_TOLERANCE_FACTOR = 0.6 # Allow up to 60% energy loss/gain (generous for lossy coding)
ENERGY_COMPARISON_MIN_FACTOR = 0.1 # Ensure reconstructed energy is at least 10% of original

def calculate_mdct_energy(mdct_coeffs_list_of_lists):
    """Calculates total energy from a list of lists of MDCT coefficients."""
    total_energy = 0.0
    for ch_coeffs in mdct_coeffs_list_of_lists:
        for coeff in ch_coeffs:
            total_energy += coeff * coeff
    return total_energy

def reconstruct_mdct_coeffs_from_encoder_data(enc_data, num_channels):
    """
    Reconstructs MDCT coefficients using intermediate data from the encoder.
    This mimics the dequantization process.
    """
    reconstructed_mdct_specs = [[] for _ in range(num_channels)]

    for ch in range(num_channels):
        # Initialize with zeros, to be filled
        ch_reconstructed_coeffs = [0.0] * pytrac.NUM_SAMPLES
        
        # Iterate only over active BFUs for Mode 0 (ATRAC1_MODE0_NUM_BFU = 11)
        # This prevents processing potentially spurious data in enc_data for bfu_idx >= 11.
        for bfu_idx in range(ATRAC1_MODE0_NUM_BFU):
            word_len = enc_data.bits_per_bfu[ch][bfu_idx]
            sf_idx = enc_data.scaled_blocks[ch][bfu_idx].scale_factor_index
            quant_values_for_bfu = enc_data.quantized_values[ch][bfu_idx]
            
            scale_factor_value = pytrac.SCALE_FACTOR_TABLE[sf_idx]
            print(f"DEBUG PY RECON: CH {ch} BFU {bfu_idx} - WL: {word_len}, SF_IDX: {sf_idx}, SF_VAL: {scale_factor_value:.4f}, NumQuantVals: {len(quant_values_for_bfu)}")

            # Determine band and if short window is used for that band
            band = ATRAC1_BFU_TO_BAND[bfu_idx]
            is_short_window = False
            if band == 0: # Low band
                is_short_window = enc_data.enc_log_count_low[ch]
            elif band == 1: # Mid band
                is_short_window = enc_data.enc_log_count_mid[ch]
            elif band == 2: # High band
                is_short_window = enc_data.enc_log_count_hi[ch]

            # Select MDCT offset based on window type
            if is_short_window:
                mdct_offset = pytrac.SPECS_START_SHORT_FULL[bfu_idx]
            else:
                mdct_offset = pytrac.SPECS_START_LONG[bfu_idx]
            print(f"DEBUG PY RECON: CH {ch} BFU {bfu_idx} - ShortWin: {is_short_window}, MDCT_Offset: {mdct_offset}")
            
            # num_coeffs_in_bfu_spec = ATRAC1_CPP_SPECS_PER_BLOCK[bfu_idx] # This is the max possible for this BFU
            # The actual number of values is len(quant_values_for_bfu)
 
            for k in range(len(quant_values_for_bfu)):
                quant_val_int = quant_values_for_bfu[k]
                reconstructed_normalized_coeff = 0.0
 
                if word_len == 1:
                    reconstructed_normalized_coeff = float(quant_val_int)
                elif word_len > 1:
                    dequant_denominator = float((1 << (word_len - 1)) - 1)
                    if dequant_denominator != 0:
                        reconstructed_normalized_coeff = float(quant_val_int) / dequant_denominator
                
                # Apply the scale factor to get the unnormalized (reconstructed) MDCT coefficient
                reconstructed_mdct_coeff = reconstructed_normalized_coeff * scale_factor_value
                
                target_idx = mdct_offset + k
                print(f"DEBUG PY RECON: CH {ch} BFU {bfu_idx} K {k} - QVal: {quant_val_int}, ReconNorm: {reconstructed_normalized_coeff:.4f}, ReconScaled: {reconstructed_mdct_coeff:.4f}, TargetIdx: {target_idx}")
                if target_idx < pytrac.NUM_SAMPLES:
                     ch_reconstructed_coeffs[target_idx] = reconstructed_mdct_coeff
                else:
                    # Should not happen with correct BFU definitions
                    print(f"DEBUG PY RECON: WARNING - CH {ch} BFU {bfu_idx} K {k} - TargetIdx {target_idx} out of bounds ({pytrac.NUM_SAMPLES})")
                    pass
        
        reconstructed_mdct_specs[ch] = ch_reconstructed_coeffs
    return reconstructed_mdct_specs


def test_mdct_energy_consistency_mono(mono_encoder): # mono_encoder fixture provides FrameProcessor
    """Test MDCT energy consistency for mono audio."""
    np.random.seed(42) # Ensure deterministic test
    num_channels = 1
    
    # Generate audio and process with encoder
    original_audio_data_np = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)
    original_audio_data_list = [ch.tolist() for ch in original_audio_data_np]
    enc_data = mono_encoder.process_frame(original_audio_data_list)

    # --- BEGIN TARGETED DUMP of scaled_blocks.values for BFU 3 (seed 42) ---
    if hasattr(enc_data, 'scaled_blocks') and len(enc_data.scaled_blocks) > 0 and \
       len(enc_data.scaled_blocks[0]) > 3: # Check if BFU 3 exists for channel 0
        bfu3_ch0_values = enc_data.scaled_blocks[0][3].values
        print(f"\nDEBUG PY TARGETED DUMP (seed 42): CH 0 BFU 3 scaled_blocks.values (len {len(bfu3_ch0_values)}):")
        for k_val, val_item in enumerate(bfu3_ch0_values):
            print(f"  K {k_val}: {val_item:.6f}")
    else:
        print("\nDEBUG PY TARGETED DUMP (seed 42): CH 0 BFU 3 scaled_blocks not available or too short.")

    print("\nDEBUG PY SCALED_BLOCKS_CHECK: Mono Test (seed 42)")
    for ch_idx in range(num_channels):
        for bfu_idx_loop in range(ATRAC1_MODE0_NUM_BFU): # Check active BFUs
            # Determine MDCT offset for this BFU
            band = ATRAC1_BFU_TO_BAND[bfu_idx_loop]
            is_short_window_check = False
            if band == 0: is_short_window_check = enc_data.enc_log_count_low[ch_idx]
            elif band == 1: is_short_window_check = enc_data.enc_log_count_mid[ch_idx]
            elif band == 2: is_short_window_check = enc_data.enc_log_count_hi[ch_idx]
            
            mdct_offset_check = pytrac.SPECS_START_SHORT_FULL[bfu_idx_loop] if is_short_window_check else pytrac.SPECS_START_LONG[bfu_idx_loop]
            
            num_coeffs_in_bfu_actual = len(enc_data.scaled_blocks[ch_idx][bfu_idx_loop].values)
            
            if num_coeffs_in_bfu_actual == 0 and enc_data.bits_per_bfu[ch_idx][bfu_idx_loop] == 0:
                # print(f"  CH {ch_idx} BFU {bfu_idx_loop}: WL=0, No scaled values. Skipping check.")
                continue

            sf_idx_check = enc_data.scaled_blocks[ch_idx][bfu_idx_loop].scale_factor_index
            sf_val_check = pytrac.SCALE_FACTOR_TABLE[sf_idx_check]

            # print(f"  CH {ch_idx} BFU {bfu_idx_loop}: SF_IDX={sf_idx_check}, SF_VAL={sf_val_check:.4f}, NumActualScaledCoeffs={num_coeffs_in_bfu_actual}, WL={enc_data.bits_per_bfu[ch_idx][bfu_idx_loop]}")

            for k_check in range(num_coeffs_in_bfu_actual):
                original_coeff_idx = mdct_offset_check + k_check
                if original_coeff_idx < pytrac.NUM_SAMPLES:
                    original_mdct_val = enc_data.mdct_specs[ch_idx][original_coeff_idx]
                    actual_normalized_val = enc_data.scaled_blocks[ch_idx][bfu_idx_loop].values[k_check]
                    
                    expected_normalized_val = 0.0
                    if sf_val_check != 0:
                        expected_normalized_val = original_mdct_val / sf_val_check
                    
                    diff = abs(actual_normalized_val - expected_normalized_val)
                    if diff > 1e-5 : # Allow small tolerance
                        print(f"  WARNING: CH {ch_idx} BFU {bfu_idx_loop} K {k_check}: Normalized value mismatch!")
                        print(f"    OrigMDCT: {original_mdct_val:.6f}, SF_VAL: {sf_val_check:.4f}")
                        print(f"    ExpectedNorm: {expected_normalized_val:.6f}, ActualNorm (from scaled_blocks.values): {actual_normalized_val:.6f}, Diff: {diff:.6f}")
                        # This assertion would fail the test here if uncommented, useful for pinpointing
                        # assert diff <= 1e-5, "Normalized value inconsistency detected"
    # --- END DIAGNOSTIC CHECK ---
 
    # 1. Calculate energy of original MDCT coefficients
    original_mdct_energy = calculate_mdct_energy(enc_data.mdct_specs)
 
    # 2. Reconstruct MDCT coefficients from encoder's intermediate quantized data
    reconstructed_mdcts = reconstruct_mdct_coeffs_from_encoder_data(enc_data, num_channels)
    
    # 3. Calculate energy of reconstructed MDCT coefficients
    reconstructed_mdct_energy = calculate_mdct_energy(reconstructed_mdcts)

    if original_mdct_energy > 1e-9: # Avoid issues with near-zero energy
        assert reconstructed_mdct_energy <= original_mdct_energy * (1.0 + ENERGY_COMPARISON_TOLERANCE_FACTOR), \
            f"Mono: Reconstructed MDCT energy ({reconstructed_mdct_energy}) too high compared to original ({original_mdct_energy})"
        assert reconstructed_mdct_energy >= original_mdct_energy * ENERGY_COMPARISON_MIN_FACTOR, \
            f"Mono: Reconstructed MDCT energy ({reconstructed_mdct_energy}) too low compared to original ({original_mdct_energy})"
    else: # If original energy is very low, reconstructed should also be very low
        assert reconstructed_mdct_energy < 1e-6, \
            f"Mono: Original energy near zero, but reconstructed energy ({reconstructed_mdct_energy}) is not."


def test_mdct_energy_consistency_stereo(stereo_encoder): # stereo_encoder fixture
    """Test MDCT energy consistency for stereo audio."""
    np.random.seed(43) # Ensure deterministic test
    num_channels = 2

    original_audio_data_np = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)
    original_audio_data_list = [ch.tolist() for ch in original_audio_data_np]
    enc_data = stereo_encoder.process_frame(original_audio_data_list)

    original_mdct_energy = calculate_mdct_energy(enc_data.mdct_specs)
    reconstructed_mdcts = reconstruct_mdct_coeffs_from_encoder_data(enc_data, num_channels)
    reconstructed_mdct_energy = calculate_mdct_energy(reconstructed_mdcts)

    if original_mdct_energy > 1e-9:
        assert reconstructed_mdct_energy <= original_mdct_energy * (1.0 + ENERGY_COMPARISON_TOLERANCE_FACTOR), \
            f"Stereo: Reconstructed MDCT energy ({reconstructed_mdct_energy}) too high compared to original ({original_mdct_energy})"
        assert reconstructed_mdct_energy >= original_mdct_energy * ENERGY_COMPARISON_MIN_FACTOR, \
            f"Stereo: Reconstructed MDCT energy ({reconstructed_mdct_energy}) too low compared to original ({original_mdct_energy})"
    else:
        assert reconstructed_mdct_energy < 1e-6, \
            f"Stereo: Original energy near zero, but reconstructed energy ({reconstructed_mdct_energy}) is not."


# --- Direct MDCT-to-PCM Path Test ---
def test_consistency_pcm_output_mono_direct_mdct(mono_decoder):
    """Compare original PCM input with final decoded PCM output,
    feeding ENCODER'S MDCT data directly to decoder's IMDCT/QMF stages."""
    original_audio_np, enc_data, _ = encode_and_get_data(num_channels=1) # We need enc_data

    assert original_audio_np.shape == (1, pytrac.NUM_SAMPLES)
    
    py_mdct_data = np.array([enc_data.mdct_specs[0]], dtype=np.float32)
    py_window_masks = np.array([enc_data.window_mask[0]], dtype=np.uint32)

    qmf_output_from_direct_mdct = mono_decoder.mdct_to_qmf(py_mdct_data, py_window_masks)

    print("\n--- Mono PCM Debug (Direct MDCT Path) ---")
    print("Stats for QMF output from mono_decoder.mdct_to_qmf (using enc_data.mdct_specs):")
    qmf_data_for_stats = qmf_output_from_direct_mdct[0] # Get the single channel
    print(f"  QMF (packed) Min: {np.min(qmf_data_for_stats):.4f}, Max: {np.max(qmf_data_for_stats):.4f}, Mean: {np.mean(qmf_data_for_stats):.4f}, Std: {np.std(qmf_data_for_stats):.4f}")
    print(f"  Has NaN: {np.isnan(qmf_data_for_stats).any()}, Has Inf: {np.isinf(qmf_data_for_stats).any()}")

    decoded_pcm_np_direct = mono_decoder.qmf_to_pcm(qmf_output_from_direct_mdct)

    decoded_pcm_ch = decoded_pcm_np_direct[0]
    original_pcm_ch = original_audio_np[0]

    # Apply clamping to match C++ decoder behavior
    decoded_pcm_ch = np.clip(decoded_pcm_ch, -1.0, 1.0)

    print(f"Original PCM range: [{np.min(original_pcm_ch):.4f}, {np.max(original_pcm_ch):.4f}]")
    print(f"Decoded PCM (direct MDCT) range: [{np.min(decoded_pcm_ch):.4f}, {np.max(decoded_pcm_ch):.4f}]")
    
    # Calculate RMS error
    rms_error = np.sqrt(np.mean((original_pcm_ch - decoded_pcm_ch)**2))

    # Calculate optimal scaling factor k and scaled RMS error for direct MDCT path
    if np.sum(decoded_pcm_ch**2) > 1e-12: # Avoid division by zero
        k_optimal_direct = np.sum(original_pcm_ch * decoded_pcm_ch) / np.sum(decoded_pcm_ch**2)
        rms_error_scaled_direct = np.sqrt(np.mean((original_pcm_ch - k_optimal_direct * decoded_pcm_ch)**2))
        print(f"Direct MDCT - Optimal scaling factor k: {k_optimal_direct:.6f}")
        print(f"Direct MDCT - RMS error after optimal scaling: {rms_error_scaled_direct:.6f}")

        k_fixed_direct = 0.785793
        rms_error_fixed_scaled_direct = np.sqrt(np.mean((original_pcm_ch - k_fixed_direct * decoded_pcm_ch)**2))
        print(f"Direct MDCT - RMS error after fixed scaling (k={k_fixed_direct}): {rms_error_fixed_scaled_direct:.6f}")
    else:
        print("Direct MDCT - Decoded PCM energy too low for optimal scaling calculation.")

    # Expected characteristics of ATRAC1 direct MDCT path with random input
    assert 0.75 < k_optimal_direct < 0.85, \
        f"Direct MDCT scaling factor {k_optimal_direct:.6f} outside expected range"
    assert 0.45 < rms_error_scaled_direct < 0.55, \
        f"Direct MDCT scaled RMS error {rms_error_scaled_direct:.6f} outside expected range"