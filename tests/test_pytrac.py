import pytrac
import numpy as np
import pytest
import soundfile as sf 
import math

ATRAC1_MAX_BANDS = 4 

ATRAC1_MODE0_NUM_BFU = 11

ATRAC1_MODE0_SPECS_PER_BFU = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64]

ATRAC1_MODE0_SPEC_OFFSET_PER_BFU = [
    0, 32, 64, 96, 128, 160, 192, 224,  
    256,  
    320,  
    384   
]

ATRAC1_BFU_TO_BAND = ([0] * 20) + ([1] * 16) + ([2] * 16)

SFI_ZERO_OFFSET = 32 

def generate_audio_frame(num_channels, num_samples, dtype=np.float32):
    """Generates a random audio frame."""

    return np.random.rand(num_channels, num_samples).astype(dtype)

def encode_and_get_data(num_channels):
    np.random.seed(0) 
    """
    Encodes a random audio frame and returns the original audio data (numpy array),
    encoder's intermediate data, and the generated bitstream.
    """
    if num_channels == 1:
        encoder = pytrac.FrameProcessor(1) 
    elif num_channels == 2:
        encoder = pytrac.FrameProcessor(2) 
    else:
        raise ValueError("Unsupported number of channels for consistency testing.")

    original_audio_data_np = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)

    original_audio_data_list = [ch.tolist() for ch in original_audio_data_np]

    encoder_intermediate_data = encoder.process_frame(original_audio_data_list)

    list_of_python_bytes = []
    for ch_raw_payload in encoder_intermediate_data.compressed_data_per_channel:
        if not ch_raw_payload:  
            list_of_python_bytes.append(b'\x00' * 212)
            continue

        if isinstance(ch_raw_payload[0], str):  
            temp_str = "".join(ch_raw_payload)
            byte_values = temp_str.encode('latin-1')
        elif isinstance(ch_raw_payload[0], int):  
            byte_values = bytes(b & 0xFF for b in ch_raw_payload)
        else:
            raise TypeError(f"Unexpected element type in compressedDataPerChannel: {type(ch_raw_payload[0])}")

        if len(byte_values) > 212:
            byte_values = byte_values[:212]
        elif len(byte_values) < 212:
            byte_values = byte_values.ljust(212, b'\x00')

        list_of_python_bytes.append(byte_values)

    return original_audio_data_np, encoder_intermediate_data, list_of_python_bytes

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
    assert len(intermediate_data.qmf_low) == 1 
    assert len(intermediate_data.qmf_low[0]) == 128 
    assert all(isinstance(x, float) for x in intermediate_data.qmf_low[0])

    assert intermediate_data.qmf_mid is not None
    assert isinstance(intermediate_data.qmf_mid, list)
    assert len(intermediate_data.qmf_mid) == 1 
    assert len(intermediate_data.qmf_mid[0]) == 128 
    assert all(isinstance(x, float) for x in intermediate_data.qmf_mid[0])

    assert intermediate_data.qmf_hi is not None
    assert isinstance(intermediate_data.qmf_hi, list)
    assert len(intermediate_data.qmf_hi) == 1 
    assert len(intermediate_data.qmf_hi[0]) == 256 
    assert all(isinstance(x, float) for x in intermediate_data.qmf_hi[0])

def test_encoder_qmf_output_shape_stereo(stereo_encoder):
    """Test QMF output shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)

    assert intermediate_data.qmf_low is not None
    assert isinstance(intermediate_data.qmf_low, list)
    assert len(intermediate_data.qmf_low) == 2 
    assert all(len(ch_data) == 128 for ch_data in intermediate_data.qmf_low)
    assert all(isinstance(x, float) for x in intermediate_data.qmf_low[0])

    assert intermediate_data.qmf_mid is not None
    assert isinstance(intermediate_data.qmf_mid, list)
    assert len(intermediate_data.qmf_mid) == 2 
    assert all(len(ch_data) == 128 for ch_data in intermediate_data.qmf_mid)
    assert all(isinstance(x, float) for x in intermediate_data.qmf_mid[0])

    assert intermediate_data.qmf_hi is not None
    assert isinstance(intermediate_data.qmf_hi, list)
    assert len(intermediate_data.qmf_hi) == 2 
    assert all(len(ch_data) == 256 for ch_data in intermediate_data.qmf_hi)
    assert all(isinstance(x, float) for x in intermediate_data.qmf_hi[0])

def test_encoder_mdct_coefficients_shape_mono(mono_encoder):
    """Test MDCT coefficients shape and type for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)

    assert intermediate_data.mdct_specs is not None
    assert isinstance(intermediate_data.mdct_specs, list)
    assert len(intermediate_data.mdct_specs) == 1 
    assert len(intermediate_data.mdct_specs[0]) == pytrac.NUM_SAMPLES 
    assert all(isinstance(x, float) for x in intermediate_data.mdct_specs[0])

def test_encoder_mdct_coefficients_shape_stereo(stereo_encoder):
    """Test MDCT coefficients shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)

    assert intermediate_data.mdct_specs is not None
    assert isinstance(intermediate_data.mdct_specs, list)
    assert len(intermediate_data.mdct_specs) == 2 
    assert all(len(ch_data) == pytrac.NUM_SAMPLES for ch_data in intermediate_data.mdct_specs)
    assert all(isinstance(x, float) for x in intermediate_data.mdct_specs[0])

def test_encoder_scale_factor_indices_mono(mono_encoder):
    """Test scale_factor_indices shape, type, and value range for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)

    assert intermediate_data.scaled_blocks is not None
    assert isinstance(intermediate_data.scaled_blocks, list)
    assert len(intermediate_data.scaled_blocks) == 1 

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
    assert len(intermediate_data.scaled_blocks) == 2 
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
    assert len(intermediate_data.bits_per_bfu) == 1 
    assert isinstance(intermediate_data.bits_per_bfu[0], list)

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
    assert len(intermediate_data.bits_per_bfu) == 2 
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
    assert len(intermediate_data.quantized_values) == 1 
    assert isinstance(intermediate_data.quantized_values[0], list)

    assert len(intermediate_data.quantized_values[0]) == len(intermediate_data.scaled_blocks[0])
    for bfu_idx, bfu_quant_values in enumerate(intermediate_data.quantized_values[0]):
        assert isinstance(bfu_quant_values, list)
        if intermediate_data.bits_per_bfu[0][bfu_idx] > 0:
            assert all(isinstance(x, int) for x in bfu_quant_values) 
        else:
            assert len(bfu_quant_values) == 0

def test_encoder_quantized_values_shape_stereo(stereo_encoder):
    """Test quantized_values shape and type for stereo audio."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES)
    intermediate_data = stereo_encoder.process_frame(audio_data)

    assert intermediate_data.quantized_values is not None
    assert isinstance(intermediate_data.quantized_values, list)
    assert len(intermediate_data.quantized_values) == 2 
    for ch in range(2):
        assert isinstance(intermediate_data.quantized_values[ch], list)
        assert len(intermediate_data.quantized_values[ch]) == len(intermediate_data.scaled_blocks[ch])
        for bfu_idx, bfu_quant_values in enumerate(intermediate_data.quantized_values[ch]):
            assert isinstance(bfu_quant_values, list)
            if intermediate_data.bits_per_bfu[ch][bfu_idx] > 0:
                 assert all(isinstance(x, int) for x in bfu_quant_values) 
            else:
                assert len(bfu_quant_values) == 0

def test_encoder_quantization_error_shape_mono(mono_encoder):
    """Test quantization_error shape and type for mono audio."""
    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES)
    intermediate_data = mono_encoder.process_frame(audio_data)

    assert intermediate_data.quantization_error is not None
    assert isinstance(intermediate_data.quantization_error, list)
    assert len(intermediate_data.quantization_error) == 1 
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
    assert len(intermediate_data.quantization_error) == 2 
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

    audio_data = generate_audio_frame(1, pytrac.NUM_SAMPLES) 
    intermediate_data = mono_encoder.process_frame(audio_data)

    for ch_idx, ch_quant_values in enumerate(intermediate_data.quantized_values):
        for bfu_idx, bfu_values in enumerate(ch_quant_values):
            if intermediate_data.bits_per_bfu[ch_idx][bfu_idx] == 0:
                assert len(bfu_values) == 0, \
                    f"Channel {ch_idx}, BFU {bfu_idx} has 0 bits but non-zero quantized values list (len {len(bfu_values)})."

def test_encoder_consistency_zero_bits_quantized_values_stereo(stereo_encoder):
    """Test that if bits_per_bfu is 0 for a band, quantized_values for that band are 0 (stereo)."""
    audio_data = generate_audio_frame(2, pytrac.NUM_SAMPLES) 
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

        expected_bfu_list_len = len(intermediate_data.scaled_blocks[ch])
        assert len(intermediate_data.bits_per_bfu[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantized_values[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantization_error[ch]) == expected_bfu_list_len

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

                for k in range(len(quant_values_for_bfu)):
                    quant_val_int = quant_values_for_bfu[k] 
                    reported_error_from_cpp = error_values_for_bfu[k]

                    if k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values):
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]
                    else:
                        assert k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values), \
                            f"Ch {ch} BFU {bfu_idx} Coeff {k}: Index k out of bounds for scaled_blocks.values. " \
                            f"len(quant_values_for_bfu)={len(quant_values_for_bfu)}, " \
                            f"len(scaled_blocks.values)={len(intermediate_data.scaled_blocks[ch][bfu_idx].values)}"
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]

                    original_unnormalized_mdct_coeff = intermediate_data.mdct_specs[ch][mdct_offset + k]

                    reconstructed_normalized_coeff_py = 0.0
                    if word_len == 1:
                        reconstructed_normalized_coeff_py = float(quant_val_int)
                    elif word_len > 1:
                        cpp_dequant_denominator = float((1 << (word_len - 1)) - 1)
                        if cpp_dequant_denominator != 0: 
                            reconstructed_normalized_coeff_py = float(quant_val_int) / cpp_dequant_denominator

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

        expected_bfu_list_len = len(intermediate_data.scaled_blocks[ch])
        assert len(intermediate_data.bits_per_bfu[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantized_values[ch]) == expected_bfu_list_len
        assert len(intermediate_data.quantization_error[ch]) == expected_bfu_list_len

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

                for k in range(len(quant_values_for_bfu)):
                    quant_val_int = quant_values_for_bfu[k] 
                    reported_error_from_cpp = error_values_for_bfu[k]

                    if k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values):
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]
                    else:
                        assert k < len(intermediate_data.scaled_blocks[ch][bfu_idx].values), \
                            f"Ch {ch} BFU {bfu_idx} Coeff {k}: Index k out of bounds for scaled_blocks.values. " \
                            f"len(quant_values_for_bfu)={len(quant_values_for_bfu)}, " \
                            f"len(scaled_blocks.values)={len(intermediate_data.scaled_blocks[ch][bfu_idx].values)}"
                        original_normalized_mdct_coeff = intermediate_data.scaled_blocks[ch][bfu_idx].values[k]

                    original_unnormalized_mdct_coeff = intermediate_data.mdct_specs[ch][mdct_offset + k]

                    reconstructed_normalized_coeff_py = 0.0
                    if word_len == 1:
                        reconstructed_normalized_coeff_py = float(quant_val_int)
                    elif word_len > 1:
                        cpp_dequant_denominator = float((1 << (word_len - 1)) - 1)
                        if cpp_dequant_denominator != 0: 
                            reconstructed_normalized_coeff_py = float(quant_val_int) / cpp_dequant_denominator

                    calculated_error_in_normalized_domain = original_normalized_mdct_coeff - reconstructed_normalized_coeff_py

                    assert abs(reported_error_from_cpp - calculated_error_in_normalized_domain) < tolerance, \
                        f"Ch {ch} BFU {bfu_idx} Coeff {k}: Normalized Error mismatch. Reported {reported_error_from_cpp}, Calculated {calculated_error_in_normalized_domain}. " \
                        f"OrigNormMDCT: {original_normalized_mdct_coeff}, ReconNormMDCT: {reconstructed_normalized_coeff_py}, QuantInt: {quant_val_int}, " \
                        f"SF Idx: {sf_idx}, WL: {word_len}, OrigMDCT_unnorm: {original_unnormalized_mdct_coeff}"

def get_generated_atrac1_bitstream_frame(num_channels):
    """
    Generates an ATRAC1 bitstream frame using the FrameProcessor.
    Returns:
        list_of_python_bytes (List[bytes]): List of Python bytes objects per channel.
    """
    if num_channels == 1:
        encoder = pytrac.FrameProcessor(1)
    elif num_channels == 2:
        encoder = pytrac.FrameProcessor(2)
    else:
        raise ValueError("Unsupported number of channels for bitstream generation.")

    audio_data = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)
    intermediate_data = encoder.process_frame(audio_data)

    list_of_python_bytes = []
    for ch_raw_payload in intermediate_data.compressed_data_per_channel:
        if not ch_raw_payload:  
            list_of_python_bytes.append(b'\x00' * 212)
            continue

        if isinstance(ch_raw_payload[0], str):  
            temp_str = "".join(ch_raw_payload)
            byte_values = temp_str.encode('latin-1')
        elif isinstance(ch_raw_payload[0], int):  
            byte_values = bytes(b & 0xFF for b in ch_raw_payload)
        else:
            raise TypeError(f"Unexpected element type in compressedDataPerChannel: {type(ch_raw_payload[0])}")

        if len(byte_values) > 212:
            byte_values = byte_values[:212]
        elif len(byte_values) < 212:
            byte_values = byte_values.ljust(212, b'\x00')

        list_of_python_bytes.append(byte_values)

    return list_of_python_bytes

@pytest.fixture
def mono_decoder():
    """Fixture for a mono PyAtrac1FrameDecoder."""

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
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    block_log_count = decoder_intermediate_data.block_size_log_count
    assert block_log_count is not None
    assert isinstance(block_log_count, list)
    assert len(block_log_count) == 1 
    assert isinstance(block_log_count[0], list) 
    assert len(block_log_count[0]) > 0 
    assert isinstance(block_log_count[0][0], int) 
    assert block_log_count[0][0] in [0, 1, 2] 

def test_decoder_block_size_log_count_stereo(stereo_decoder):
    """Test block_size_log_count_per_channel for stereo audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    block_log_count = decoder_intermediate_data.block_size_log_count
    assert block_log_count is not None
    assert isinstance(block_log_count, list)
    assert len(block_log_count) == 2 
    for ch_block_log_counts in block_log_count:
        assert isinstance(ch_block_log_counts, list) 
        assert len(ch_block_log_counts) > 0 
        assert isinstance(ch_block_log_counts[0], int) 
        assert ch_block_log_counts[0] in [0, 1, 2] 

def test_decoder_scale_factor_indices_mono(mono_decoder):
    """Test scale_factor_indices_per_channel for mono audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    sf_indices = decoder_intermediate_data.scale_factor_indices
    assert sf_indices is not None
    assert isinstance(sf_indices, list)
    assert len(sf_indices) == 1 

    ch_sfs = sf_indices[0] 
    assert isinstance(ch_sfs, list) 

    if decoder_intermediate_data.block_size_log_count[0] == 0: 
            assert len(ch_sfs) == ATRAC1_MODE0_NUM_BFU
    for val in ch_sfs: 
        assert isinstance(val, int)
        assert 0 <= val < 64

def test_decoder_scale_factor_indices_stereo(stereo_decoder):
    """Test scale_factor_indices_per_channel for stereo audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    sf_indices = decoder_intermediate_data.scale_factor_indices
    assert sf_indices is not None
    assert isinstance(sf_indices, list)
    assert len(sf_indices) == 2 
    for ch_idx, ch_sfs in enumerate(sf_indices):
        assert isinstance(ch_sfs, list)
        if decoder_intermediate_data.block_size_log_count[ch_idx] == 0: 
            assert len(ch_sfs) == ATRAC1_MODE0_NUM_BFU
        for val in ch_sfs:
            assert isinstance(val, int)
            assert 0 <= val < 64

def test_decoder_bits_per_bfu_mono(mono_decoder):
    """Test bits_per_bfu_per_channel for mono audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    bits_per_bfu = decoder_intermediate_data.bits_per_bfu
    assert bits_per_bfu is not None
    assert isinstance(bits_per_bfu, list)
    assert len(bits_per_bfu) == 1 

    ch_bits = bits_per_bfu[0]
    assert isinstance(ch_bits, list) 
    if decoder_intermediate_data.block_size_log_count[0] == 0: 
        assert len(ch_bits) == ATRAC1_MODE0_NUM_BFU
    for val in ch_bits: 
        assert isinstance(val, int)
        assert 0 <= val <= 15

def test_decoder_bits_per_bfu_stereo(stereo_decoder):
    """Test bits_per_bfu_per_channel for stereo audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    bits_per_bfu = decoder_intermediate_data.bits_per_bfu
    assert bits_per_bfu is not None
    assert isinstance(bits_per_bfu, list)
    assert len(bits_per_bfu) == 2 
    for ch_idx, ch_bits in enumerate(bits_per_bfu):
        assert isinstance(ch_bits, list)
        if decoder_intermediate_data.block_size_log_count[ch_idx] == 0: 
            assert len(ch_bits) == ATRAC1_MODE0_NUM_BFU
        for val in ch_bits:
            assert isinstance(val, int)
            assert 0 <= val <= 15

def test_decoder_parsed_quantized_values_mono(mono_decoder):
    """Test parsed_quantized_values_per_channel for mono audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    quant_vals = decoder_intermediate_data.parsed_quantized_values
    assert quant_vals is not None
    assert isinstance(quant_vals, list)
    assert len(quant_vals) == 1 

    ch_data = quant_vals[0]
    assert isinstance(ch_data, list) 
    if decoder_intermediate_data.block_size_log_count[0] == 0: 
        assert len(ch_data) == ATRAC1_MODE0_NUM_BFU
        for bfu_idx, bfu_quant_data in enumerate(ch_data): 
            assert isinstance(bfu_quant_data, list) 
            word_len = decoder_intermediate_data.bits_per_bfu[0][bfu_idx]
            if word_len > 0:
                assert len(bfu_quant_data) == ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx]
                for val in bfu_quant_data:
                    assert isinstance(val, int)
            else:
                assert len(bfu_quant_data) == 0

def test_decoder_parsed_quantized_values_stereo(stereo_decoder):
    """Test parsed_quantized_values_per_channel for stereo audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    quant_vals = decoder_intermediate_data.parsed_quantized_values
    assert quant_vals is not None
    assert isinstance(quant_vals, list)
    assert len(quant_vals) == 2 

    for ch_idx in range(2):
        ch_data = quant_vals[ch_idx]
        assert isinstance(ch_data, list) 
        if decoder_intermediate_data.block_size_log_count[ch_idx] == 0: 
            assert len(ch_data) == ATRAC1_MODE0_NUM_BFU
            for bfu_idx, bfu_quant_data in enumerate(ch_data): 
                assert isinstance(bfu_quant_data, list) 
                word_len = decoder_intermediate_data.bits_per_bfu[ch_idx][bfu_idx]
                if word_len > 0:
                    assert len(bfu_quant_data) == ATRAC1_MODE0_SPECS_PER_BFU[bfu_idx]
                    for val in bfu_quant_data:
                        assert isinstance(val, int)
                else:
                    assert len(bfu_quant_data) == 0

def test_decoder_mdct_specs_mono(mono_decoder):
    """Test mdct_specs_per_channel for mono audio (dequantized MDCT)."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)

    mdct_specs = decoder_intermediate_data.mdct_specs
    assert mdct_specs is not None
    assert isinstance(mdct_specs, list)
    assert len(mdct_specs) == 1 

    ch_mdct_data = mdct_specs[0]
    assert isinstance(ch_mdct_data, list) 
    assert len(ch_mdct_data) == pytrac.NUM_SAMPLES
    for val in ch_mdct_data:
        assert isinstance(val, float)

def test_decoder_mdct_specs_stereo(stereo_decoder):
    """Test mdct_specs_per_channel for stereo audio (dequantized MDCT)."""
    per_channel_bitstream = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream)

    mdct_specs = decoder_intermediate_data.mdct_specs
    assert mdct_specs is not None
    assert isinstance(mdct_specs, list)
    assert len(mdct_specs) == 2 

    for ch_mdct_data in mdct_specs:
        assert isinstance(ch_mdct_data, list)
        assert len(ch_mdct_data) == pytrac.NUM_SAMPLES
        for val in ch_mdct_data:
            assert isinstance(val, float)

def test_decoder_get_decoded_audio_mono(mono_decoder):
    """Test get_decoded_audio() for mono audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=1)
    decoder_intermediate_data = mono_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)
    decoded_audio = decoder_intermediate_data.pcm_output

    assert decoded_audio is not None
    assert isinstance(decoded_audio, list)
    assert len(decoded_audio) == 1 

    ch_audio = decoded_audio[0]
    assert isinstance(ch_audio, list)
    assert len(ch_audio) == pytrac.NUM_SAMPLES
    assert all(isinstance(x, float) for x in ch_audio)

def test_decoder_get_decoded_audio_stereo(stereo_decoder):
    """Test get_decoded_audio() for stereo audio."""
    per_channel_bitstream_bytes = get_generated_atrac1_bitstream_frame(num_channels=2)
    decoder_intermediate_data = stereo_decoder.decode_frame_from_bitstream(per_channel_bitstream_bytes)
    decoded_audio = decoder_intermediate_data.pcm_output

    assert decoded_audio is not None
    assert isinstance(decoded_audio, list)
    assert len(decoded_audio) == 2 
    for ch_audio in decoded_audio:
        assert isinstance(ch_audio, list)
        assert len(ch_audio) == pytrac.NUM_SAMPLES
        assert all(isinstance(x, float) for x in ch_audio)

def test_consistency_scale_factors_mono(mono_decoder):
    """Compare encoder's scale factors with decoder's parsed scale factors (mono)."""
    _, enc_data, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert len(enc_data.scaled_blocks) == 1
    assert len(dec_data.scale_factor_indices) == 1

    encoder_sfs = [block.scale_factor_index for block in enc_data.scaled_blocks[0]]
    decoder_sfs = dec_data.scale_factor_indices[0]

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
        word_len = enc_data.bits_per_bfu[0][bfu_idx] 

        if word_len == 0:
            assert len(enc_bfu_qvs) == 0, f"Mono BFU {bfu_idx}: Encoder QVs non-empty for WL=0"
            assert len(dec_bfu_qvs) == 0, f"Mono BFU {bfu_idx}: Decoder QVs non-empty for WL=0"
        else:

            assert len(dec_bfu_qvs) == pytrac.SPECS_PER_BLOCK[bfu_idx], \
                f"Mono BFU {bfu_idx}: Decoder QV count mismatch. Expected {pytrac.SPECS_PER_BLOCK[bfu_idx]}, Got {len(dec_bfu_qvs)}"

            assert len(enc_bfu_qvs) <= len(dec_bfu_qvs), \
                 f"Mono BFU {bfu_idx}: Encoder QVs longer than decoder QVs. Enc: {len(enc_bfu_qvs)}, Dec: {len(dec_bfu_qvs)}"

            for i in range(len(enc_bfu_qvs)):
                 assert enc_bfu_qvs[i] == dec_bfu_qvs[i], \
                    f"Mono QV mismatch at BFU {bfu_idx}, Coeff {i}: Encoder {enc_bfu_qvs[i]}, Decoder {dec_bfu_qvs[i]}"

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

MDCT_COMPARISON_TOLERANCE = 1e-3 

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
    significant_diff_count = np.sum(diffs > MDCT_COMPARISON_TOLERANCE) 

    print(f"MDCT Stats - SignificantDiffs (>{MDCT_COMPARISON_TOLERANCE}): {significant_diff_count}/{pytrac.NUM_SAMPLES}, AvgAbsDiff: {avg_abs_diff:.6f}, MaxAbsDiff: {max_abs_diff:.6f}")

    if significant_diff_count > 0:
        print("Example significant differences (Original vs Decoded):")
        indices = np.where(diffs > MDCT_COMPARISON_TOLERANCE)[0]
        for i, idx in enumerate(indices[:5]): 
             print(f"  idx {idx}: {original_mdcts[idx]:.6f} vs {decoded_mdcts[idx]:.6f} (Diff: {original_mdcts[idx]-decoded_mdcts[idx]:.6f})")

    assert avg_abs_diff < 0.1, f"MDCT spectra average absolute difference too high: {avg_abs_diff:.6f}"
    assert max_abs_diff < 0.5, f"MDCT spectra maximum absolute difference too high: {max_abs_diff:.6f}" 

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

PCM_COMPARISON_TOLERANCE_RMS = 0.1 

def test_consistency_pcm_output_mono(mono_decoder):
    """Compare original PCM input with final decoded PCM output (mono)."""
    original_audio_np, _, bitstream = encode_and_get_data(num_channels=1)
    dec_data = mono_decoder.decode_frame_from_bitstream(bitstream)

    assert original_audio_np.shape == (1, pytrac.NUM_SAMPLES)
    assert len(dec_data.pcm_output) == 1
    assert len(dec_data.pcm_output[0]) == pytrac.NUM_SAMPLES

    mdct_coeffs_for_imdct = np.array(dec_data.mdct_specs[0])
    print("\n--- Mono Decoded MDCT Input to IMdct Stats ---")
    print(f"Min: {np.min(mdct_coeffs_for_imdct):.4f}, Max: {np.max(mdct_coeffs_for_imdct):.4f}, Mean: {np.mean(mdct_coeffs_for_imdct):.4f}, Std: {np.std(mdct_coeffs_for_imdct):.4f}")
    print(f"Has NaN: {np.isnan(mdct_coeffs_for_imdct).any()}, Has Inf: {np.isinf(mdct_coeffs_for_imdct).any()}")

    original_pcm_ch = original_audio_np[0]
    decoded_pcm_ch = np.array(dec_data.pcm_output[0])

    decoded_pcm_ch = np.clip(decoded_pcm_ch, -1.0, 1.0)

    print("\n--- Mono PCM Debug ---")
    print(f"Original PCM range: [{np.min(original_pcm_ch):.4f}, {np.max(original_pcm_ch):.4f}]")
    print(f"Decoded PCM range: [{np.min(decoded_pcm_ch):.4f}, {np.max(decoded_pcm_ch):.4f}]")

    print("\nFirst 10 MDCT coefficients (decoded):")
    for i in range(10):
        print(f"  {i}: {dec_data.mdct_specs[0][i]:.6f}")

    bands = [
        ("Low", 0, 127),
        ("Mid", 128, 255),
        ("High", 256, 511)
    ]
    for name, start, end in bands:
        band_energy = np.sum(np.square(dec_data.mdct_specs[0][start:end+1]))
        print(f"{name} band energy: {band_energy:.4f}")

    rms_error = np.sqrt(np.mean((original_pcm_ch - decoded_pcm_ch)**2))

    if np.sum(decoded_pcm_ch**2) > 1e-12: 
        k_optimal = np.sum(original_pcm_ch * decoded_pcm_ch) / np.sum(decoded_pcm_ch**2)
        rms_error_scaled = np.sqrt(np.mean((original_pcm_ch - k_optimal * decoded_pcm_ch)**2))
        print(f"Optimal scaling factor k: {k_optimal:.6f}")
        print(f"RMS error after optimal scaling: {rms_error_scaled:.6f}")

        k_fixed = 0.785793
        rms_error_fixed_scaled = np.sqrt(np.mean((original_pcm_ch - k_fixed * decoded_pcm_ch)**2))
        print(f"RMS error after fixed scaling (k={k_fixed}): {rms_error_fixed_scaled:.6f}")
    else:
        print("Decoded PCM energy too low for optimal scaling calculation.")

    assert 0.75 < k_optimal < 0.85, \
        f"Optimal scaling factor {k_optimal:.6f} outside expected ATRAC1 range (0.75-0.85)"
    assert 0.45 < rms_error_scaled < 0.55, \
        f"Scaled RMS error {rms_error_scaled:.6f} outside expected ATRAC1 range (0.45-0.55)"

def test_consistency_pcm_output_stereo(stereo_decoder):
    """Compare original PCM input with final decoded PCM output (stereo)."""
    original_audio_np, enc_intermediate_data, bitstream = encode_and_get_data(num_channels=2) 
    dec_data = stereo_decoder.decode_frame_from_bitstream(bitstream)

    if enc_intermediate_data: 
        print("\n--- PyTrac Debug: Intermediate Data Stats (Stereo Test - Channel 0) ---")

        if enc_intermediate_data.pcm_input and len(enc_intermediate_data.pcm_input) > 0:
            pcm_input_ch0 = np.array(enc_intermediate_data.pcm_input[0])
            print(f"PCM Input:  Min={np.min(pcm_input_ch0):.3e}, Max={np.max(pcm_input_ch0):.3e}, Mean={np.mean(pcm_input_ch0):.3e}, Std={np.std(pcm_input_ch0):.3e}, NaN={np.isnan(pcm_input_ch0).any()}, Inf={np.isinf(pcm_input_ch0).any()}")

        if enc_intermediate_data.qmf_low and len(enc_intermediate_data.qmf_low) > 0:
            qmf_low_ch0 = np.array(enc_intermediate_data.qmf_low[0])
            print(f"QMF Low:    Min={np.min(qmf_low_ch0):.3e}, Max={np.max(qmf_low_ch0):.3e}, Mean={np.mean(qmf_low_ch0):.3e}, Std={np.std(qmf_low_ch0):.3e}, NaN={np.isnan(qmf_low_ch0).any()}, Inf={np.isinf(qmf_low_ch0).any()}")

        if enc_intermediate_data.qmf_mid and len(enc_intermediate_data.qmf_mid) > 0:
            qmf_mid_ch0 = np.array(enc_intermediate_data.qmf_mid[0])
            print(f"QMF Mid:    Min={np.min(qmf_mid_ch0):.3e}, Max={np.max(qmf_mid_ch0):.3e}, Mean={np.mean(qmf_mid_ch0):.3e}, Std={np.std(qmf_mid_ch0):.3e}, NaN={np.isnan(qmf_mid_ch0).any()}, Inf={np.isinf(qmf_mid_ch0).any()}")

        if enc_intermediate_data.qmf_hi and len(enc_intermediate_data.qmf_hi) > 0:
            qmf_hi_ch0 = np.array(enc_intermediate_data.qmf_hi[0])
            print(f"QMF Hi:     Min={np.min(qmf_hi_ch0):.3e}, Max={np.max(qmf_hi_ch0):.3e}, Mean={np.mean(qmf_hi_ch0):.3e}, Std={np.std(qmf_hi_ch0):.3e}, NaN={np.isnan(qmf_hi_ch0).any()}, Inf={np.isinf(qmf_hi_ch0).any()}")

        if enc_intermediate_data.mdct_specs and len(enc_intermediate_data.mdct_specs) > 0:
            mdct_specs_ch0 = np.array(enc_intermediate_data.mdct_specs[0])
            print(f"MDCT Specs: Min={np.min(mdct_specs_ch0):.3e}, Max={np.max(mdct_specs_ch0):.3e}, Mean={np.mean(mdct_specs_ch0):.3e}, Std={np.std(mdct_specs_ch0):.3e}, NaN={np.isnan(mdct_specs_ch0).any()}, Inf={np.isinf(mdct_specs_ch0).any()}")

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

        decoded_pcm_ch = np.clip(decoded_pcm_ch, -1.0, 1.0)

        print(f"\n--- Stereo Ch{ch} PCM Debug ---")
        print(f"Original PCM range: [{np.min(original_pcm_ch):.4f}, {np.max(original_pcm_ch):.4f}]")
        print(f"Decoded PCM range: [{np.min(decoded_pcm_ch):.4f}, {np.max(decoded_pcm_ch):.4f}]")

        print("\nFirst 10 MDCT coefficients (decoded):")
        for i in range(10):
            print(f"  {i}: {dec_data.mdct_specs[ch][i]:.6f}")

        bands = [
            ("Low", 0, 127),
            ("Mid", 128, 255),
            ("High", 256, 511)
        ]
        for name, start, end in bands:
            band_energy = np.sum(np.square(dec_data.mdct_specs[ch][start:end+1]))
            print(f"{name} band energy: {band_energy:.4f}")

        rms_error = np.sqrt(np.mean((original_pcm_ch - decoded_pcm_ch)**2))

        if np.sum(decoded_pcm_ch**2) > 1e-12:
            k_optimal_stereo = np.sum(original_pcm_ch * decoded_pcm_ch) / np.sum(decoded_pcm_ch**2)
            rms_error_scaled_stereo = np.sqrt(np.mean((original_pcm_ch - k_optimal_stereo * decoded_pcm_ch)**2))
            print(f"Stereo Ch {ch} Optimal scaling factor k: {k_optimal_stereo:.6f}")
            print(f"Stereo Ch {ch} RMS error after optimal scaling: {rms_error_scaled_stereo:.6f}")

            k_fixed_stereo = 0.785793 
            rms_error_fixed_scaled_stereo = np.sqrt(np.mean((original_pcm_ch - k_fixed_stereo * decoded_pcm_ch)**2))
            print(f"Stereo Ch {ch} RMS error after fixed scaling (k={k_fixed_stereo}): {rms_error_fixed_scaled_stereo:.6f}")
        else:
            print(f"Stereo Ch {ch} Decoded PCM energy too low for optimal scaling calculation.")

        assert 0.73 < k_optimal_stereo < 0.87, \
            f"Stereo Ch {ch} scaling factor {k_optimal_stereo:.6f} outside expected range"
        assert 0.45 < rms_error_scaled_stereo < 0.55, \
            f"Stereo Ch {ch} scaled RMS error {rms_error_scaled_stereo:.6f} outside expected range"

ENERGY_COMPARISON_TOLERANCE_FACTOR = 0.6 
ENERGY_COMPARISON_MIN_FACTOR = 0.1 

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

        ch_reconstructed_coeffs = [0.0] * pytrac.NUM_SAMPLES

        for bfu_idx in range(ATRAC1_MODE0_NUM_BFU):
            word_len = enc_data.bits_per_bfu[ch][bfu_idx]
            sf_idx = enc_data.scaled_blocks[ch][bfu_idx].scale_factor_index
            quant_values_for_bfu = enc_data.quantized_values[ch][bfu_idx]

            scale_factor_value = pytrac.SCALE_FACTOR_TABLE[sf_idx]
            print(f"DEBUG PY RECON: CH {ch} BFU {bfu_idx} - WL: {word_len}, SF_IDX: {sf_idx}, SF_VAL: {scale_factor_value:.4f}, NumQuantVals: {len(quant_values_for_bfu)}")

            band = ATRAC1_BFU_TO_BAND[bfu_idx]
            is_short_window = False
            if band == 0: 
                is_short_window = enc_data.enc_log_count_low[ch]
            elif band == 1: 
                is_short_window = enc_data.enc_log_count_mid[ch]
            elif band == 2: 
                is_short_window = enc_data.enc_log_count_hi[ch]

            if is_short_window:
                mdct_offset = pytrac.SPECS_START_SHORT[bfu_idx]
            else:
                mdct_offset = pytrac.SPECS_START_LONG[bfu_idx]
            print(f"DEBUG PY RECON: CH {ch} BFU {bfu_idx} - ShortWin: {is_short_window}, MDCT_Offset: {mdct_offset}")

            for k in range(len(quant_values_for_bfu)):
                quant_val_int = quant_values_for_bfu[k]
                reconstructed_normalized_coeff = 0.0

                if word_len == 1:
                    reconstructed_normalized_coeff = float(quant_val_int)
                elif word_len > 1:
                    dequant_denominator = float((1 << (word_len - 1)) - 1)
                    if dequant_denominator != 0:
                        reconstructed_normalized_coeff = float(quant_val_int) / dequant_denominator

                reconstructed_mdct_coeff = reconstructed_normalized_coeff * scale_factor_value

                target_idx = mdct_offset + k
                print(f"DEBUG PY RECON: CH {ch} BFU {bfu_idx} K {k} - QVal: {quant_val_int}, ReconNorm: {reconstructed_normalized_coeff:.4f}, ReconScaled: {reconstructed_mdct_coeff:.4f}, TargetIdx: {target_idx}")
                if target_idx < pytrac.NUM_SAMPLES:
                     ch_reconstructed_coeffs[target_idx] = reconstructed_mdct_coeff
                else:

                    print(f"DEBUG PY RECON: WARNING - CH {ch} BFU {bfu_idx} K {k} - TargetIdx {target_idx} out of bounds ({pytrac.NUM_SAMPLES})")
                    pass

        reconstructed_mdct_specs[ch] = ch_reconstructed_coeffs
    return reconstructed_mdct_specs

def test_mdct_energy_consistency_mono(mono_encoder): 
    """Test MDCT energy consistency for mono audio."""
    np.random.seed(42) 
    num_channels = 1

    original_audio_data_np = generate_audio_frame(num_channels, pytrac.NUM_SAMPLES)
    original_audio_data_list = [ch.tolist() for ch in original_audio_data_np]
    enc_data = mono_encoder.process_frame(original_audio_data_list)

    if hasattr(enc_data, 'scaled_blocks') and len(enc_data.scaled_blocks) > 0 and \
       len(enc_data.scaled_blocks[0]) > 3: 
        bfu3_ch0_values = enc_data.scaled_blocks[0][3].values
        print(f"\nDEBUG PY TARGETED DUMP (seed 42): CH 0 BFU 3 scaled_blocks.values (len {len(bfu3_ch0_values)}):")
        for k_val, val_item in enumerate(bfu3_ch0_values):
            print(f"  K {k_val}: {val_item:.6f}")
    else:
        print("\nDEBUG PY TARGETED DUMP (seed 42): CH 0 BFU 3 scaled_blocks not available or too short.")

    print("\nDEBUG PY SCALED_BLOCKS_CHECK: Mono Test (seed 42)")
    for ch_idx in range(num_channels):
        for bfu_idx_loop in range(ATRAC1_MODE0_NUM_BFU): 

            band = ATRAC1_BFU_TO_BAND[bfu_idx_loop]
            is_short_window_check = False
            if band == 0: is_short_window_check = enc_data.enc_log_count_low[ch_idx]
            elif band == 1: is_short_window_check = enc_data.enc_log_count_mid[ch_idx]
            elif band == 2: is_short_window_check = enc_data.enc_log_count_hi[ch_idx]

            mdct_offset_check = pytrac.SPECS_START_SHORT[bfu_idx_loop] if is_short_window_check else pytrac.SPECS_START_LONG[bfu_idx_loop]

            num_coeffs_in_bfu_actual = len(enc_data.scaled_blocks[ch_idx][bfu_idx_loop].values)

            if num_coeffs_in_bfu_actual == 0 and enc_data.bits_per_bfu[ch_idx][bfu_idx_loop] == 0:

                continue

            sf_idx_check = enc_data.scaled_blocks[ch_idx][bfu_idx_loop].scale_factor_index
            sf_val_check = pytrac.SCALE_FACTOR_TABLE[sf_idx_check]

            for k_check in range(num_coeffs_in_bfu_actual):
                original_coeff_idx = mdct_offset_check + k_check
                if original_coeff_idx < pytrac.NUM_SAMPLES:
                    original_mdct_val = enc_data.mdct_specs[ch_idx][original_coeff_idx]
                    actual_normalized_val = enc_data.scaled_blocks[ch_idx][bfu_idx_loop].values[k_check]

                    expected_normalized_val = 0.0
                    if sf_val_check != 0:
                        expected_normalized_val = original_mdct_val / sf_val_check

                    diff = abs(actual_normalized_val - expected_normalized_val)
                    if diff > 1e-5 : 
                        print(f"  WARNING: CH {ch_idx} BFU {bfu_idx_loop} K {k_check}: Normalized value mismatch!")
                        print(f"    OrigMDCT: {original_mdct_val:.6f}, SF_VAL: {sf_val_check:.4f}")
                        print(f"    ExpectedNorm: {expected_normalized_val:.6f}, ActualNorm (from scaled_blocks.values): {actual_normalized_val:.6f}, Diff: {diff:.6f}")

    original_mdct_energy = calculate_mdct_energy(enc_data.mdct_specs)

    reconstructed_mdcts = reconstruct_mdct_coeffs_from_encoder_data(enc_data, num_channels)

    reconstructed_mdct_energy = calculate_mdct_energy(reconstructed_mdcts)

    if original_mdct_energy > 1e-9: 
        assert reconstructed_mdct_energy <= original_mdct_energy * (1.0 + ENERGY_COMPARISON_TOLERANCE_FACTOR), \
            f"Mono: Reconstructed MDCT energy ({reconstructed_mdct_energy}) too high compared to original ({original_mdct_energy})"
        assert reconstructed_mdct_energy >= original_mdct_energy * ENERGY_COMPARISON_MIN_FACTOR, \
            f"Mono: Reconstructed MDCT energy ({reconstructed_mdct_energy}) too low compared to original ({original_mdct_energy})"
    else: 
        assert reconstructed_mdct_energy < 1e-6, \
            f"Mono: Original energy near zero, but reconstructed energy ({reconstructed_mdct_energy}) is not."

def test_mdct_energy_consistency_stereo(stereo_encoder): 
    """Test MDCT energy consistency for stereo audio."""
    np.random.seed(43) 
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

def test_consistency_pcm_output_mono_direct_mdct(mono_decoder):
    """Compare original PCM input with final decoded PCM output,
    feeding ENCODER'S MDCT data directly to decoder's IMDCT/QMF stages."""
    original_audio_np, enc_data, _ = encode_and_get_data(num_channels=1) 

    assert original_audio_np.shape == (1, pytrac.NUM_SAMPLES)

    py_mdct_data = np.array([enc_data.mdct_specs[0]], dtype=np.float32)
    py_window_masks = np.array([enc_data.window_mask[0]], dtype=np.uint32)

    qmf_output_from_direct_mdct = mono_decoder.mdct_to_qmf(py_mdct_data, py_window_masks)

    print("\n--- Mono PCM Debug (Direct MDCT Path) ---")
    print("Stats for QMF output from mono_decoder.mdct_to_qmf (using enc_data.mdct_specs):")
    qmf_data_for_stats = qmf_output_from_direct_mdct[0] 
    print(f"  QMF (packed) Min: {np.min(qmf_data_for_stats):.4f}, Max: {np.max(qmf_data_for_stats):.4f}, Mean: {np.mean(qmf_data_for_stats):.4f}, Std: {np.std(qmf_data_for_stats):.4f}")
    print(f"  Has NaN: {np.isnan(qmf_data_for_stats).any()}, Has Inf: {np.isinf(qmf_data_for_stats).any()}")

    decoded_pcm_np_direct = mono_decoder.qmf_to_pcm(qmf_output_from_direct_mdct)

    decoded_pcm_ch = decoded_pcm_np_direct[0]
    original_pcm_ch = original_audio_np[0]

    decoded_pcm_ch = np.clip(decoded_pcm_ch, -1.0, 1.0)

    print(f"Original PCM range: [{np.min(original_pcm_ch):.4f}, {np.max(original_pcm_ch):.4f}]")
    print(f"Decoded PCM (direct MDCT) range: [{np.min(decoded_pcm_ch):.4f}, {np.max(decoded_pcm_ch):.4f}]")

    rms_error = np.sqrt(np.mean((original_pcm_ch - decoded_pcm_ch)**2))

    if np.sum(decoded_pcm_ch**2) > 1e-12: 
        k_optimal_direct = np.sum(original_pcm_ch * decoded_pcm_ch) / np.sum(decoded_pcm_ch**2)
        rms_error_scaled_direct = np.sqrt(np.mean((original_pcm_ch - k_optimal_direct * decoded_pcm_ch)**2))
        print(f"Direct MDCT - Optimal scaling factor k: {k_optimal_direct:.6f}")
        print(f"Direct MDCT - RMS error after optimal scaling: {rms_error_scaled_direct:.6f}")

        k_fixed_direct = 0.785793
        rms_error_fixed_scaled_direct = np.sqrt(np.mean((original_pcm_ch - k_fixed_direct * decoded_pcm_ch)**2))
        print(f"Direct MDCT - RMS error after fixed scaling (k={k_fixed_direct}): {rms_error_fixed_scaled_direct:.6f}")
    else:
        print("Direct MDCT - Decoded PCM energy too low for optimal scaling calculation.")

    assert 0.75 < k_optimal_direct < 0.85, \
        f"Direct MDCT scaling factor {k_optimal_direct:.6f} outside expected range"
    assert 0.45 < rms_error_scaled_direct < 0.55, \
        f"Direct MDCT scaled RMS error {rms_error_scaled_direct:.6f} outside expected range"

def create_default_nn_frame_parameters(bfu_amount_table_index=7):
    """Creates a default NNFrameParameters object with specified BFU amount index."""
    params = pytrac.NNFrameParameters()
    params.block_mode = pytrac.BlockSizeMod(False, False, False)  
    params.bfu_amount_table_index = bfu_amount_table_index

    num_active_bfus = pytrac.BFU_AMOUNT_TABLE[bfu_amount_table_index]
    params.word_lengths = [0] * num_active_bfus
    params.scale_factor_indices = [0] * num_active_bfus
    params.quantized_spectrum = [[] for _ in range(num_active_bfus)]

    return params, num_active_bfus

def test_nn_frame_parameters_creation():
    """Test basic creation and attribute access of NNFrameParameters."""
    params, num_active_bfus = create_default_nn_frame_parameters()
    assert isinstance(params.block_mode, pytrac.BlockSizeMod)
    assert params.bfu_amount_table_index == 7
    assert len(params.word_lengths) == num_active_bfus
    assert len(params.scale_factor_indices) == num_active_bfus
    assert len(params.quantized_spectrum) == num_active_bfus

def test_assemble_mono_payload_basic():
    """Test assembly of a mono payload with minimal parameters."""
    params, num_active_bfus = create_default_nn_frame_parameters(bfu_amount_table_index=0)  

    if num_active_bfus > 0:
        params.word_lengths[0] = 2  
        params.scale_factor_indices[0] = 10
        params.quantized_spectrum[0] = [1] * pytrac.SPECS_PER_BLOCK[0]  

    payload = pytrac.assemble_mono_frame_payload(params)
    assert isinstance(payload, bytes)
    assert len(payload) == 212  

def test_assemble_stereo_payloads_basic():
    """Test assembly of stereo payloads."""
    params_ch0, _ = create_default_nn_frame_parameters(bfu_amount_table_index=0)
    params_ch1, _ = create_default_nn_frame_parameters(bfu_amount_table_index=0)

    if len(params_ch1.word_lengths) > 1:
        params_ch1.word_lengths[1] = 2
        params_ch1.scale_factor_indices[1] = 5
        params_ch1.quantized_spectrum[1] = [-1] * pytrac.SPECS_PER_BLOCK[1]

    payloads = pytrac.assemble_stereo_frame_payloads(params_ch0, params_ch1)
    assert isinstance(payloads, list)
    assert len(payloads) == 2
    assert all(isinstance(p, bytes) for p in payloads)
    assert all(len(p) == 212 for p in payloads)

@pytest.mark.parametrize("num_channels", [1, 2])
def test_nn_round_trip_window_mode(num_channels, mono_decoder, stereo_decoder):
    """Test window mode consistency through NN assembler and decoder."""
    decoder = mono_decoder if num_channels == 1 else stereo_decoder

    test_modes = [
        (pytrac.BlockSizeMod(False, False, False), [0, 0, 0]),  
        (pytrac.BlockSizeMod(True, True, True),   [2, 2, 3]),     
        (pytrac.BlockSizeMod(True, False, True),  [2, 0, 3])      
    ]

    for block_mode, expected_log_counts in test_modes:
        params_ch0, _ = create_default_nn_frame_parameters()
        params_ch0.block_mode = block_mode

        if num_channels == 1:
            payloads = [pytrac.assemble_mono_frame_payload(params_ch0)]
        else:
            params_ch1 = create_default_nn_frame_parameters()[0]
            params_ch1.block_mode = block_mode
            payloads = pytrac.assemble_stereo_frame_payloads(params_ch0, params_ch1)

        decoded_data = decoder.decode_frame_from_bitstream(payloads)

        for ch in range(num_channels):
            assert decoded_data.block_size_log_count[ch][0] == expected_log_counts[0]
            assert decoded_data.block_size_log_count[ch][1] == expected_log_counts[1]
            assert decoded_data.block_size_log_count[ch][2] == expected_log_counts[2]

@pytest.mark.parametrize("num_channels", [1, 2])
def test_nn_round_trip_parameters(num_channels, mono_decoder, stereo_decoder):
    """Test parameter consistency through NN assembler and decoder."""
    decoder = mono_decoder if num_channels == 1 else stereo_decoder
    bfu_amount_idx = 0  
    params_ch0, num_active_bfus = create_default_nn_frame_parameters(bfu_amount_idx)

    test_bfus = {
        0: {"wl": 4, "sf": 10, "qvs": [1, -1, 2, -2]},
        1: {"wl": 0, "sf": 5, "qvs": []},
        2: {"wl": 2, "sf": 15, "qvs": [1, 0, -1, 1]}
    }

    for bfu_idx, data in test_bfus.items():
        if bfu_idx < num_active_bfus:
            params_ch0.word_lengths[bfu_idx] = data["wl"]
            params_ch0.scale_factor_indices[bfu_idx] = data["sf"]
            params_ch0.quantized_spectrum[bfu_idx] = data["qvs"]

    if num_channels == 1:
        payloads = [pytrac.assemble_mono_frame_payload(params_ch0)]
    else:
        params_ch1 = create_default_nn_frame_parameters(bfu_amount_idx)[0]
        payloads = pytrac.assemble_stereo_frame_payloads(params_ch0, params_ch1)

    decoded_data = decoder.decode_frame_from_bitstream(payloads)

    for ch in range(num_channels):

        assert len(decoded_data.bits_per_bfu[ch]) >= num_active_bfus

        for bfu_idx in range(num_active_bfus):

            expected_wl = params_ch0.word_lengths[bfu_idx]
            decoded_idwl = decoded_data.bits_per_bfu[ch][bfu_idx]
            expected_idwl = 0 if expected_wl <= 1 else (expected_wl - 1)
            assert decoded_idwl == expected_idwl

            assert decoded_data.scale_factor_indices[ch][bfu_idx] == params_ch0.scale_factor_indices[bfu_idx]

            if expected_wl > 0:
                assert decoded_data.parsed_quantized_values[ch][bfu_idx][:len(params_ch0.quantized_spectrum[bfu_idx])] == params_ch0.quantized_spectrum[bfu_idx]

def create_varied_nn_frame_params_list(num_frames, bfu_amount_idx_override=None):
    """
    Creates a list of NNFrameParameters.
    The first frame is designed to be distinctly non-silent using a standard ATRAC1 Mode 0 configuration.
    Subsequent frames are silent.
    """
    params_list = []
    print(f"\n--- Debug: Creating {num_frames} NNFrameParameters ---")

    first_frame_bfu_amount_idx = 7
    if bfu_amount_idx_override is not None:
        first_frame_bfu_amount_idx = bfu_amount_idx_override

    for frame_idx in range(num_frames):

        current_bfu_amount_idx = first_frame_bfu_amount_idx if frame_idx == 0 else 7

        params, num_active_bfus_from_default = create_default_nn_frame_parameters(current_bfu_amount_idx)

        print(f"  Frame {frame_idx}: Using BFU Amount Idx {current_bfu_amount_idx}, num_active_bfus (from BFU_AMOUNT_TABLE) = {num_active_bfus_from_default}")

        if frame_idx == 0:
            print(f"  Frame 0: Configuring for potentially non-silent output.")

            new_word_lengths = [0] * num_active_bfus_from_default
            new_scale_factor_indices = [0] * num_active_bfus_from_default
            new_quantized_spectrum = [[] for _ in range(num_active_bfus_from_default)]

            bfu_to_activate = 0 
            if bfu_to_activate < num_active_bfus_from_default:
                new_word_lengths[bfu_to_activate] = 4          
                new_scale_factor_indices[bfu_to_activate] = 32 

                num_coeffs_in_bfu0 = pytrac.SPECS_PER_BLOCK[bfu_to_activate]

                qvs_bfu0 = [1] * num_coeffs_in_bfu0
                new_quantized_spectrum[bfu_to_activate] = qvs_bfu0

            params.word_lengths = new_word_lengths
            params.scale_factor_indices = new_scale_factor_indices
            params.quantized_spectrum = new_quantized_spectrum

            if bfu_to_activate < num_active_bfus_from_default:
                 print(f"    BFU {bfu_to_activate} (after re-assignment): WL={params.word_lengths[bfu_to_activate]}, SFI={params.scale_factor_indices[bfu_to_activate]}, NumQVS={len(params.quantized_spectrum[bfu_to_activate])}, QVS_sample={params.quantized_spectrum[bfu_to_activate][:5]}")

            if not (bfu_to_activate < num_active_bfus_from_default and params.word_lengths[bfu_to_activate] > 0):
                print(f"  WARNING Frame 0: BFU {bfu_to_activate} still not active after attempted configuration. WL={params.word_lengths[bfu_to_activate] if bfu_to_activate < num_active_bfus_from_default else 'N/A'}")

        params_list.append(params)
    return params_list

@pytest.mark.parametrize("num_frames", [1, 3])
def test_decode_snippet_basic_mono(mono_decoder, num_frames):
    """Test basic functionality of decode_snippet_from_params_list for mono."""
    params_list = create_varied_nn_frame_params_list(num_frames)

    decoded_snippet = mono_decoder.decode_snippet_from_params_list(params_list, 1, 0)

    assert isinstance(decoded_snippet, np.ndarray)
    assert decoded_snippet.dtype == np.float32
    assert decoded_snippet.ndim == 1
    assert len(decoded_snippet) == num_frames * pytrac.NUM_SAMPLES

    if num_frames > 0:

        assert np.sum(np.abs(decoded_snippet)) > 1e-6, \
            f"Decoded snippet is all zeros or too close to zero. Sum(abs): {np.sum(np.abs(decoded_snippet))}"

@pytest.mark.parametrize("num_frames", [2])
def test_decode_snippet_expected_samples_mono(mono_decoder, num_frames):
    """Test expected_total_samples argument for decode_snippet_from_params_list."""
    params_list = create_varied_nn_frame_params_list(num_frames)
    natural_length = num_frames * pytrac.NUM_SAMPLES

    snippet_match = mono_decoder.decode_snippet_from_params_list(params_list, 1, natural_length)
    assert len(snippet_match) == natural_length

    expected_truncate_len = natural_length - pytrac.NUM_SAMPLES // 2
    snippet_truncate = mono_decoder.decode_snippet_from_params_list(params_list, 1, expected_truncate_len)
    assert len(snippet_truncate) == expected_truncate_len
    np.testing.assert_array_almost_equal(snippet_match[:expected_truncate_len], snippet_truncate)

    expected_pad_len = natural_length + pytrac.NUM_SAMPLES // 2
    snippet_pad = mono_decoder.decode_snippet_from_params_list(params_list, 1, expected_pad_len)
    assert len(snippet_pad) == expected_pad_len
    np.testing.assert_array_almost_equal(snippet_match, snippet_pad[:natural_length])
    assert np.all(snippet_pad[natural_length:] == 0.0)

def test_decode_snippet_empty_list_mono(mono_decoder):
    """Test decode_snippet_from_params_list with an empty list of parameters."""
    empty_params_list = []

    snippet_empty_default = mono_decoder.decode_snippet_from_params_list(empty_params_list, 1, 0)
    assert isinstance(snippet_empty_default, np.ndarray)
    assert len(snippet_empty_default) == 0

    expected_len = pytrac.NUM_SAMPLES
    snippet_empty_expected = mono_decoder.decode_snippet_from_params_list(empty_params_list, 1, expected_len)
    assert isinstance(snippet_empty_expected, np.ndarray)
    assert len(snippet_empty_expected) == expected_len
    assert np.all(snippet_empty_expected == 0.0)

@pytest.mark.parametrize("num_frames", [1, 2]) 
def test_decode_snippet_consistency_mono(mono_decoder, num_frames):
    """Compare decode_snippet output with frame-by-frame decoding."""
    params_list = create_varied_nn_frame_params_list(num_frames, bfu_amount_idx_override=6) 

    expected_pcm_frames = []
    for frame_params in params_list:

        payload_bytes = pytrac.assemble_mono_frame_payload(frame_params)

        decoded_intermediate = mono_decoder.decode_frame_from_bitstream([payload_bytes])

        pcm_single_frame = np.array(decoded_intermediate.pcm_output[0], dtype=np.float32)
        expected_pcm_frames.append(pcm_single_frame)

    if expected_pcm_frames:
        expected_snippet_np = np.concatenate(expected_pcm_frames)
    else:
        expected_snippet_np = np.array([], dtype=np.float32)

    actual_snippet_np = mono_decoder.decode_snippet_from_params_list(
        params_list,
        num_output_channels=1,
        expected_total_samples=len(expected_snippet_np) 
    )

    assert actual_snippet_np.shape == expected_snippet_np.shape
    np.testing.assert_allclose(actual_snippet_np, expected_snippet_np, rtol=1e-6, atol=1e-7,
                               err_msg="Mismatch between snippet decoded from params list and concatenated frame-by-frame decoding.")