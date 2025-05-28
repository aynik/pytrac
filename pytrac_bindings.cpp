/*
 * PyTrac - Python bindings for ATRAC1 audio codec
 *
 * This file provides comprehensive Python bindings for the atracdenc library,
 * exposing the full functionality of the ATRAC1 encoder and decoder.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <cmath>

// ATRAC1 headers
#include "atrac1denc.h"
#include "aea.h"
#include "wav.h"
#include "atrac/atrac1.h"
#include "atrac/atrac1_qmf.h"
#include "atrac/atrac_scale.h"
#include "atrac/atrac1_bitalloc.h"
#include "atrac/atrac1_dequantiser.h"
#include "pcmengin.h"
#include "transient_detector.h"
#include "util.h"
#include "compressed_io.h"
#include "bitstream/bitstream.h"

namespace py = pybind11;

using namespace NAtracDEnc;
using namespace NAtrac1;

// Forward declarations
class PyAtrac1Encoder;
class PyAtrac1Decoder;

// ==================== Core Data Structures ====================

// Define constants as simple values instead of struct members
constexpr uint32_t ATRAC_NUM_SAMPLES = 512;
constexpr uint8_t ATRAC_MAX_BFUS = 52; // Matches TAtrac1Data::MaxBfus
constexpr uint8_t ATRAC_NUM_QMF = 3;
constexpr uint32_t ATRAC_SAMPLE_RATE = 44100;

// ==================== Intermediate Data Structures ====================

struct PyEncoderIntermediateData {
    // QMF band outputs (3 bands per channel) - Corrected sizes
    std::vector<std::vector<float>> qmfLow;   // [channel][128]
    std::vector<std::vector<float>> qmfMid;   // [channel][128]
    std::vector<std::vector<float>> qmfHi;    // [channel][256]
    
    // MDCT coefficients
    std::vector<std::vector<float>> mdctSpecs; // [channel][512]
    
    // Windowing decisions
    std::vector<uint32_t> windowMask; // [channel] 3-bit mask per channel
    std::vector<std::array<uint8_t, 3>> block_size_log_count; // [channel][3 bands: low, mid, hi]
    
    // Scale factors and quantized data
    std::vector<std::vector<TScaledBlock>> scaledBlocks; // [channel][bfu]
    
    // Bit allocation per BFU
    std::vector<std::vector<uint32_t>> bitsPerBfu; // [channel][bfu]
    
    // Quantized integer values (what actually goes in bitstream)
    std::vector<std::vector<std::vector<int32_t>>> quantizedValues; // [channel][bfu][values]
    
    // Quantization errors (for loss analysis)
    std::vector<std::vector<std::vector<float>>> quantizationError; // [channel][bfu][values]
    
    // Original PCM
    std::vector<std::vector<float>> pcmInput; // [channel][512]
    
    // Bitstream data (compressed frame per channel)
    std::vector<std::vector<char>> compressedDataPerChannel;

    // Encoder's windowing decisions (true if short window)
    std::vector<bool> enc_log_count_low;    // [channel]
    std::vector<bool> enc_log_count_mid;    // [channel]
    std::vector<bool> enc_log_count_hi;     // [channel]
};
 
struct PyDecoderIntermediateData {
    // Input compressed data
    std::vector<char> compressedInput;
    
    // Bitstream parsing results
    std::vector<uint32_t> windowMask; // [channel] 3-bit mask per channel
    std::vector<std::array<uint8_t, 3>> block_size_log_count; // [channel][3 bands: low, mid, hi] - parsed
    std::vector<std::vector<uint32_t>> bitsPerBfu; // [channel][bfu] - extracted from bitstream
    
    // Dequantized values (reconstructed from bitstream)
    // Fields for the new true decoder path (to be populated by TAtrac1Dequantiser)
    std::vector<std::vector<std::vector<int32_t>>> parsed_quantized_values; // [channel][bfu][values] - Parsed by TAtrac1Dequantiser
    // Note: dequantizedFloats will effectively be mdctSpecs after TAtrac1Dequantiser runs.
    
    // Fields primarily for the decodeFromIntermediate path (placeholders/comparisons using encoder's simplified quantization)
    std::vector<std::vector<std::vector<int32_t>>> placeholder_quantized_ints_from_encoder_data; // [channel][bfu][values] - Copied from encoder's PyEncoderIntermediateData.quantizedValues
    std::vector<std::vector<std::vector<float>>> placeholder_dequantized_floats_from_encoder_data; // [channel][bfu][values] - Reconstructed in Python from placeholder_quantized_ints
    
    // Reconstructed scale factors (will be populated by true decoder path from TAtrac1Dequantiser)
    std::vector<std::vector<uint32_t>> scaleFactorIndices; // [channel][bfu] - Changed to uint32_t
    
    // MDCT coefficients (after dequantization and scaling)
    std::vector<std::vector<float>> mdctSpecs; // [channel][512]
    
    // QMF synthesis outputs - Corrected sizes
    std::vector<std::vector<float>> qmfLow;   // [channel][128]
    std::vector<std::vector<float>> qmfMid;   // [channel][128]
    std::vector<std::vector<float>> qmfHi;    // [channel][256]
    
    // Final PCM output
    std::vector<std::vector<float>> pcmOutput; // [channel][512]
};

// ==================== Frame-Level Processing Classes ====================

class PyAtrac1FrameProcessor : public ICompressedOutput, public virtual TAtrac1Data {
private:
    // Inner class to serve as a heap-allocated ICompressedOutput for TAtrac1Encoder
    class DummyOutput : public ICompressedOutput {
    private:
        uint32_t d_numChannels; // Store numChannels for GetChannelNum
    public:
        DummyOutput(uint32_t channels) : d_numChannels(channels) {}
        ~DummyOutput() override = default; // Ensure virtual destructor

        void WriteFrame(std::vector<char> /*data*/) override {
            // This is a no-op because PyAtrac1FrameProcessor retrieves
            // intermediate data directly from TAtrac1Encoder's internal state.
            // The actual compressed data is collected from m_last_frame_intermediate_data.
        }

        size_t GetChannelNum() const override {
            // This needs to return the correct number of channels for the encoder.
            return d_numChannels;
        }

        std::string GetName() const override {
            return "PyAtrac1FrameProcessor::DummyOutput";
        }
    };

    std::unique_ptr<NAtracDEnc::TAtrac1Encoder> encoder_instance_;
    NAtracDEnc::NAtrac1::TAtrac1EncodeSettings settings_;
    uint32_t numChannels;
    
public:
    PyAtrac1FrameProcessor(uint32_t channels, const NAtracDEnc::NAtrac1::TAtrac1EncodeSettings& settings = NAtracDEnc::NAtrac1::TAtrac1EncodeSettings())
        : settings_(settings) // Store a copy of the settings
        , numChannels(channels)
    {
        TCompressedOutputPtr owned_dummy_output = std::make_unique<DummyOutput>(this->numChannels);
        
        encoder_instance_ = std::make_unique<NAtracDEnc::TAtrac1Encoder>(
            std::move(owned_dummy_output),
            NAtracDEnc::NAtrac1::TAtrac1EncodeSettings(settings_) // Pass a copy of settings
        );
    }
    
    // PyAtrac1FrameProcessor's own ICompressedOutput implementation.
    // These methods are called if PyAtrac1FrameProcessor itself is used as an ICompressedOutput
    // (e.g., if it were passed to another component expecting an ICompressedOutput).
    // They are NOT directly used by its internally owned encoder_instance_, which uses the DummyOutput.
    void WriteFrame(std::vector<char> /*data*/) override {
        // This implementation is for when PyAtrac1FrameProcessor itself is the target.
        // For the internal encoder, data is retrieved from intermediate structures.
        // So, this can be a conceptual no-op in the context of processFrame().
    }

    size_t GetChannelNum() const override {
        return numChannels;
    }

    std::string GetName() const override {
        return "PyAtrac1FrameProcessorOutput"; // Name for PyAtrac1FrameProcessor itself as an output
    }

    PyEncoderIntermediateData processFrame(py::array_t<float> input) {
        auto buf = input.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input array must be 2D (channels x samples)");
        }
        if (buf.shape[0] != numChannels || buf.shape[1] != ATRAC_NUM_SAMPLES) {
            throw std::runtime_error("Input must be [channels, 512] samples");
        }
        
        PyEncoderIntermediateData result;
        result.qmfLow.resize(numChannels);
        result.qmfMid.resize(numChannels);
        result.qmfHi.resize(numChannels);
        result.mdctSpecs.resize(numChannels);
        result.windowMask.resize(numChannels);
        result.block_size_log_count.resize(numChannels);
        result.scaledBlocks.resize(numChannels);
        result.bitsPerBfu.resize(numChannels);
        result.quantizedValues.resize(numChannels);
        result.quantizationError.resize(numChannels);
        result.pcmInput.resize(numChannels);
        result.compressedDataPerChannel.resize(numChannels);
        result.enc_log_count_low.resize(numChannels);
        result.enc_log_count_mid.resize(numChannels);
        result.enc_log_count_hi.resize(numChannels);
        
        float* inputData = static_cast<float*>(buf.ptr);
        
        // 1. Prepare interleaved PCM data and store original PCM input
        std::vector<float> interleaved_pcm_data(numChannels * ATRAC_NUM_SAMPLES);
        for (uint32_t ch = 0; ch < numChannels; ++ch) {
            result.pcmInput[ch].resize(ATRAC_NUM_SAMPLES);
            for (size_t i = 0; i < ATRAC_NUM_SAMPLES; ++i) {
                // Assuming inputData is from a C-contiguous py::array [channels, samples]
                float sample = inputData[ch * ATRAC_NUM_SAMPLES + i];
                result.pcmInput[ch][i] = sample;
                interleaved_pcm_data[i * numChannels + ch] = sample;
            }
        }

        // 2. Invoke encoder's processing lambda
        auto process_lambda = encoder_instance_->GetLambda();
        TPCMEngine::ProcessMeta meta{static_cast<uint16_t>(numChannels)};
        process_lambda(interleaved_pcm_data.data(), meta);

        // 3. Retrieve C++ intermediate data
        const auto& cpp_intermediate_data = encoder_instance_->m_last_frame_intermediate_data;

        // 6. Translate C++ intermediate data to PyEncoderIntermediateData
        for (uint32_t ch = 0; ch < numChannels; ++ch) {
            if (ch >= cpp_intermediate_data.channel_data.size()) {
                continue;
            }
            const auto& cpp_ch_data = cpp_intermediate_data.channel_data[ch];

            // QMF Outputs
            result.qmfLow[ch].assign(cpp_ch_data.qmf_output_low.begin(), cpp_ch_data.qmf_output_low.end());
            result.qmfMid[ch].assign(cpp_ch_data.qmf_output_mid.begin(), cpp_ch_data.qmf_output_mid.end());
            result.qmfHi[ch].assign(cpp_ch_data.qmf_output_hi.begin(), cpp_ch_data.qmf_output_hi.end());

            // MDCT Specs
            result.mdctSpecs[ch].assign(cpp_ch_data.mdct_specs.begin(), cpp_ch_data.mdct_specs.end());

            // Windowing (from effective_block_size_mod)
            const auto& eff_block_size_mod = cpp_ch_data.effective_block_size_mod;
            result.block_size_log_count[ch] = {
                static_cast<uint8_t>(eff_block_size_mod.LogCount[0]),
                static_cast<uint8_t>(eff_block_size_mod.LogCount[1]),
                static_cast<uint8_t>(eff_block_size_mod.LogCount[2])
            };
            result.enc_log_count_low[ch] = (eff_block_size_mod.LogCount[0] == 1);
            result.enc_log_count_mid[ch] = (eff_block_size_mod.LogCount[1] == 1);
            result.enc_log_count_hi[ch] = (eff_block_size_mod.LogCount[2] == 1);
            
            uint32_t current_window_mask = 0;
            if (eff_block_size_mod.LogCount[0]) current_window_mask |= 1; // Low band short
            if (eff_block_size_mod.LogCount[1]) current_window_mask |= 2; // Mid band short
            if (eff_block_size_mod.LogCount[2]) current_window_mask |= 4; // Hi band short
            result.windowMask[ch] = current_window_mask;

            // Scaled Blocks (direct copy)
            result.scaledBlocks[ch] = cpp_ch_data.scaled_blocks_data;

            // Bits Per BFU
            result.bitsPerBfu[ch] = cpp_ch_data.final_bits_per_bfu;
            if (result.bitsPerBfu[ch].size() < ATRAC_MAX_BFUS) {
                result.bitsPerBfu[ch].resize(ATRAC_MAX_BFUS, 0); // Pad with 0
            }

            result.quantizedValues[ch].resize(ATRAC_MAX_BFUS);
            result.quantizationError[ch].resize(ATRAC_MAX_BFUS);

            size_t current_quant_idx = 0;
            size_t current_error_idx = 0;

            for (size_t bfu_idx = 0; bfu_idx < ATRAC_MAX_BFUS; ++bfu_idx) {
                uint32_t bits_for_this_bfu = 0;
                if (bfu_idx < result.bitsPerBfu[ch].size()) { // Safety check
                    bits_for_this_bfu = result.bitsPerBfu[ch][bfu_idx];
                }

                if (bits_for_this_bfu == 0) {
                    result.quantizedValues[ch][bfu_idx].clear();
                    result.quantizationError[ch][bfu_idx].clear();
                } else {
                    if (bfu_idx >= NAtrac1::TAtrac1Data::MaxBfus) { // MaxBfus is 52
                         result.quantizedValues[ch][bfu_idx].clear();
                         result.quantizationError[ch][bfu_idx].clear();
                         continue;
                    }
                    uint8_t band_idx = static_cast<uint8_t>(NAtrac1::TAtrac1Data::BfuToBand(bfu_idx));
                    
                    constexpr size_t num_bands_for_specs = sizeof(NAtrac1::TAtrac1Data::SpecsPerBlock) / sizeof(NAtrac1::TAtrac1Data::SpecsPerBlock[0]);
                    if (band_idx >= num_bands_for_specs) {
                        result.quantizedValues[ch][bfu_idx].clear();
                        result.quantizationError[ch][bfu_idx].clear();
                        continue;
                    }
                    size_t num_coeffs_to_extract = 0;
                    if (bfu_idx < cpp_ch_data.scaled_blocks_data.size()) { // Ensure bfu_idx is valid for scaled_blocks_data
                        num_coeffs_to_extract = cpp_ch_data.scaled_blocks_data[bfu_idx].Values.size();
                    } else {
                        result.quantizedValues[ch][bfu_idx].clear();
                        result.quantizationError[ch][bfu_idx].clear();
                        continue;
                    }

                    // Populate Quantized Values
                    if (num_coeffs_to_extract > 0) { // Only proceed if there are coeffs to extract
                        if (current_quant_idx + num_coeffs_to_extract <= cpp_ch_data.quantized_values.size()) {
                            result.quantizedValues[ch][bfu_idx].reserve(num_coeffs_to_extract);
                            for (size_t k = 0; k < num_coeffs_to_extract; ++k) {
                                result.quantizedValues[ch][bfu_idx].push_back(
                                    static_cast<int32_t>(cpp_ch_data.quantized_values[current_quant_idx + k])
                                );
                            }
                            current_quant_idx += num_coeffs_to_extract;
                        } else {
                            // Data mismatch: not enough data in cpp_ch_data.quantized_values
                            result.quantizedValues[ch][bfu_idx].clear(); // Clear to indicate error/missing data
                        }
                    } else { // No coeffs for this BFU (e.g. Values.size() was 0)
                        result.quantizedValues[ch][bfu_idx].clear();
                    }

                    // Populate Quantization Error
                    if (num_coeffs_to_extract > 0) { // Only proceed if there are coeffs to extract
                        if (current_error_idx + num_coeffs_to_extract <= cpp_ch_data.quantization_error.size()) {
                            result.quantizationError[ch][bfu_idx].assign(
                                cpp_ch_data.quantization_error.begin() + current_error_idx,
                                cpp_ch_data.quantization_error.begin() + current_error_idx + num_coeffs_to_extract
                            );
                            current_error_idx += num_coeffs_to_extract;
                        } else {
                            // Data mismatch: not enough data in cpp_ch_data.quantization_error
                            result.quantizationError[ch][bfu_idx].clear(); // Clear to indicate error/missing data
                        }
                    } else { // No coeffs for this BFU
                        result.quantizationError[ch][bfu_idx].clear();
                    }
                }
            }
            
            // Compressed Data Per Channel - ensure exactly 212 bytes
            std::vector<char> ch_payload = cpp_ch_data.frame_bitstream_payload;
            if (ch_payload.size() > TAtrac1Data::SoundUnitSize) {
                ch_payload.resize(TAtrac1Data::SoundUnitSize); // Truncate
            } else if (ch_payload.size() < TAtrac1Data::SoundUnitSize) {
                ch_payload.resize(TAtrac1Data::SoundUnitSize, 0); // Pad
            }
            result.compressedDataPerChannel[ch] = ch_payload;
        }

        return result;
    }

};

class PyAtrac1FrameDecoder : public TAtrac1MDCT, public virtual TAtrac1Data {
private:
    float PcmBufLow[2][256 + 16];
    float PcmBufMid[2][256 + 16];
    float PcmBufHi[2][512 + 16];
    
    Atrac1SynthesisFilterBank<float> SynthesisFilterBank[2];
    TAtrac1Dequantiser Dequantiser;
    
    uint32_t numChannels;
    
public:
    PyAtrac1FrameDecoder(uint32_t channels) : numChannels(channels) {
        for (uint32_t ch = 0; ch < 2; ch++) {
            memset(PcmBufLow[ch], 0, sizeof(PcmBufLow[ch]));
            memset(PcmBufMid[ch], 0, sizeof(PcmBufMid[ch]));
            memset(PcmBufHi[ch], 0, sizeof(PcmBufHi[ch]));
        }
    }
    
    PyDecoderIntermediateData decodeFromIntermediate(const PyEncoderIntermediateData& encoderData) {
        PyDecoderIntermediateData result;
        
        const uint32_t channels = encoderData.mdctSpecs.size();
        
        result.windowMask = encoderData.windowMask;
        result.bitsPerBfu = encoderData.bitsPerBfu;
        result.block_size_log_count.resize(channels); // Resize the new field
        result.mdctSpecs = encoderData.mdctSpecs;
        result.qmfLow.resize(channels);
        result.qmfMid.resize(channels);
        result.qmfHi.resize(channels);
        result.pcmOutput.resize(channels);
        
        result.scaleFactorIndices.resize(channels);
        for (uint32_t ch = 0; ch < channels; ++ch) {
            if (ch < encoderData.scaledBlocks.size()) {
                result.scaleFactorIndices[ch].resize(encoderData.scaledBlocks[ch].size());
                for (size_t bfu = 0; bfu < encoderData.scaledBlocks[ch].size(); ++bfu) {
                    result.scaleFactorIndices[ch][bfu] = encoderData.scaledBlocks[ch][bfu].ScaleFactorIndex;
                }
            }
        }
        
        // Populate placeholder fields for decodeFromIntermediate
        result.placeholder_quantized_ints_from_encoder_data = encoderData.quantizedValues;
        result.placeholder_dequantized_floats_from_encoder_data.resize(channels);
        
        for (uint32_t ch = 0; ch < channels; ++ch) {
            if (ch < encoderData.quantizedValues.size()) {
                result.placeholder_dequantized_floats_from_encoder_data[ch].resize(encoderData.quantizedValues[ch].size());
                for (size_t bfu = 0; bfu < encoderData.quantizedValues[ch].size(); ++bfu) {
                    const auto& quantInts = encoderData.quantizedValues[ch][bfu];
                    auto& quantFloats = result.placeholder_dequantized_floats_from_encoder_data[ch][bfu];
                    quantFloats.resize(quantInts.size());
                    
                    if (!quantInts.empty() && bfu < encoderData.bitsPerBfu[ch].size() && encoderData.bitsPerBfu[ch][bfu] > 0) {
                        const float multiple = static_cast<float>((1 << (encoderData.bitsPerBfu[ch][bfu] - 1)) - 1);
                        for (size_t i = 0; i < quantInts.size(); ++i) {
                            quantFloats[i] = (multiple > 0) ? (static_cast<float>(quantInts[i]) / multiple) : 0.0f;
                        }
                    }
                }
            }
        }
        
        for (uint32_t channel = 0; channel < channels; ++channel) {
            TAtrac1Data::TBlockSizeMod blockSize(encoderData.windowMask[channel] & 1,
                                 encoderData.windowMask[channel] & 2,
                                 encoderData.windowMask[channel] & 4);
            result.block_size_log_count[channel] = {
                static_cast<uint8_t>(blockSize.LogCount[0]),
                static_cast<uint8_t>(blockSize.LogCount[1]),
                static_cast<uint8_t>(blockSize.LogCount[2])
            };
            
            IMdct(&result.mdctSpecs[channel][0], blockSize,
                  &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0]);
            
            // Assign correct QMF band sizes from IMDCT output buffers
            result.qmfLow[channel].assign(&PcmBufLow[channel][0], &PcmBufLow[channel][128]);
            result.qmfMid[channel].assign(&PcmBufMid[channel][0], &PcmBufMid[channel][128]);
            result.qmfHi[channel].assign(&PcmBufHi[channel][0], &PcmBufHi[channel][256]);
            
            result.pcmOutput[channel].resize(ATRAC_NUM_SAMPLES);
            SynthesisFilterBank[channel].Synthesis(&result.pcmOutput[channel][0],
                &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0]);
        }
        
        return result;
    }

    PyDecoderIntermediateData decode_frame_from_bitstream(
        const std::vector<std::vector<char>>& per_channel_bitstream_data
    ) {
        PyDecoderIntermediateData result;
        const uint32_t channels = per_channel_bitstream_data.size();
        if (channels == 0 || channels > this->numChannels) {
            throw std::runtime_error("Invalid number of channels in bitstream data.");
        }

        result.compressedInput = per_channel_bitstream_data[0]; // Store first channel's bitstream for reference
        if (channels > 1 && !per_channel_bitstream_data[1].empty()) {
            // If stereo, append second channel's bitstream for reference (simple concatenation)
            result.compressedInput.insert(result.compressedInput.end(),
                                          per_channel_bitstream_data[1].begin(),
                                          per_channel_bitstream_data[1].end());
        }
        
        result.windowMask.resize(channels);
        result.block_size_log_count.resize(channels);
        result.bitsPerBfu.resize(channels);
        result.scaleFactorIndices.resize(channels);
        result.parsed_quantized_values.resize(channels);
        result.mdctSpecs.resize(channels);
        for(uint32_t ch = 0; ch < channels; ++ch) {
            result.mdctSpecs[ch].resize(ATRAC_NUM_SAMPLES); // Ensure mdctSpecs is sized
        }
        result.qmfLow.resize(channels);
        result.qmfMid.resize(channels);
        result.qmfHi.resize(channels);
        result.pcmOutput.resize(channels);

        for (uint32_t channel = 0; channel < channels; ++channel) {
            if (per_channel_bitstream_data[channel].empty()) {
                // Handle empty bitstream for a channel if necessary, or throw error
                // For now, just fill with zeros or skip processing for this channel
                std::fill(result.mdctSpecs[channel].begin(), result.mdctSpecs[channel].end(), 0.0f);
                // Initialize other per-channel data to empty/default if skipping
                result.qmfLow[channel].assign(128, 0.0f);
                result.qmfMid[channel].assign(128, 0.0f);
                result.qmfHi[channel].assign(256, 0.0f);
                result.pcmOutput[channel].assign(ATRAC_NUM_SAMPLES, 0.0f);
                continue;
            }

            // per_channel_bitstream_data[channel] is now std::vector<char>, so .data() is char*
            NBitStream::TBitStream bitstream_reader(
                (const char*)per_channel_bitstream_data[channel].data(), // Ensure const char*
                per_channel_bitstream_data[channel].size() // Size in BYTES
            );

            // Call Dequant to parse the bitstream and populate result.mdctSpecs[channel]
            Dequantiser.Dequant(&bitstream_reader, result.mdctSpecs[channel].data(), channel);

            // Retrieve parsed data from Dequantiser
            TAtrac1Data::TBlockSizeMod parsed_mode = Dequantiser.GetBlockSizeMod(channel);
            result.block_size_log_count[channel] = {
                static_cast<uint8_t>(parsed_mode.LogCount[0]),
                static_cast<uint8_t>(parsed_mode.LogCount[1]),
                static_cast<uint8_t>(parsed_mode.LogCount[2])
            };
            
            // Construct windowMask from LogCount
            uint32_t current_window_mask = 0;
            if (parsed_mode.LogCount[0]) current_window_mask |= 1;
            if (parsed_mode.LogCount[1]) current_window_mask |= 2;
            if (parsed_mode.LogCount[2]) current_window_mask |= 4;
            result.windowMask[channel] = current_window_mask;

            result.bitsPerBfu[channel] = Dequantiser.get_parsed_word_lengths(channel);
            result.scaleFactorIndices[channel] = Dequantiser.get_parsed_scale_factor_indices(channel);
            result.parsed_quantized_values[channel] = Dequantiser.get_parsed_quantized_values(channel);
            
            IMdct(&result.mdctSpecs[channel][0], parsed_mode, // Use parsed_mode here
                  &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0]);
            
            result.qmfLow[channel].assign(&PcmBufLow[channel][0], &PcmBufLow[channel][128]);
            result.qmfMid[channel].assign(&PcmBufMid[channel][0], &PcmBufMid[channel][128]);
            result.qmfHi[channel].assign(&PcmBufHi[channel][0], &PcmBufHi[channel][256]);
            
            result.pcmOutput[channel].resize(ATRAC_NUM_SAMPLES);
            SynthesisFilterBank[channel].Synthesis(&result.pcmOutput[channel][0],
                &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0]);
        }
        return result;
    }
    
    // Individual component access
    py::array_t<float> qmfToPcm(py::array_t<float> qmfData) {
        auto buf = qmfData.request();
        // Corrected total QMF samples: 128 (low) + 128 (mid) + 256 (high) = 512
        constexpr int total_qmf_samples_per_channel = 128 + 128 + 256;
        if (buf.ndim != 2 || buf.shape[0] != numChannels || buf.shape[1] != total_qmf_samples_per_channel) {
            throw std::runtime_error("QMF data must be [channels, 512 (128L+128M+256H)]");
        }
        
        float* qmfPtr = static_cast<float*>(buf.ptr);
        auto result = py::array_t<float>(std::vector<int>{(int)numChannels, ATRAC_NUM_SAMPLES});
        auto resultPtr = static_cast<float*>(result.template mutable_unchecked<2>().mutable_data(0, 0));
        
        for (uint32_t ch = 0; ch < numChannels; ++ch) {
            // Extract QMF bands: low(128) + mid(128) + hi(256)
            float* channel_base_ptr = &qmfPtr[ch * total_qmf_samples_per_channel];
            float* low = channel_base_ptr;
            float* mid = channel_base_ptr + 128;
            float* hi = channel_base_ptr + 128 + 128;
            
            SynthesisFilterBank[ch].Synthesis(&resultPtr[ch * ATRAC_NUM_SAMPLES], low, mid, hi);
        }
        
        return result;
    }
    
    py::array_t<float> mdctToQmf(py::array_t<float> mdctData, py::array_t<uint32_t> windowMasks) {
        auto mdctBuf = mdctData.request();
        auto maskBuf = windowMasks.request();
        
        if (mdctBuf.ndim != 2 || mdctBuf.shape[0] != numChannels || mdctBuf.shape[1] != 512) {
            throw std::runtime_error("MDCT data must be [channels, 512]");
        }
        if (maskBuf.ndim != 1 || maskBuf.shape[0] != numChannels) {
            throw std::runtime_error("Window masks must be [channels]");
        }
        
        float* mdctPtr = static_cast<float*>(mdctBuf.ptr);
        uint32_t* maskPtr = static_cast<uint32_t*>(maskBuf.ptr);
        
        // Corrected total QMF samples: 128 (low) + 128 (mid) + 256 (high) = 512
        constexpr int total_qmf_samples_per_channel = 128 + 128 + 256;
        auto result = py::array_t<float>(std::vector<int>{(int)numChannels, total_qmf_samples_per_channel});
        auto resultPtr = static_cast<float*>(result.template mutable_unchecked<2>().mutable_data(0, 0));
        
        for (uint32_t ch = 0; ch < numChannels; ++ch) {
            TAtrac1Data::TBlockSizeMod blockSize(maskPtr[ch] & 1, maskPtr[ch] & 2, maskPtr[ch] & 4);
            
            // IMdct populates PcmBufLow/Mid/Hi with 128/128/256 samples respectively
            IMdct(&mdctPtr[ch * 512], blockSize,
                  &PcmBufLow[ch][0], &PcmBufMid[ch][0], &PcmBufHi[ch][0]);
            
            // Pack into result: low(128) + mid(128) + hi(256)
            float* channel_base_ptr = &resultPtr[ch * total_qmf_samples_per_channel];
            memcpy(channel_base_ptr, &PcmBufLow[ch][0], 128 * sizeof(float));
            memcpy(channel_base_ptr + 128, &PcmBufMid[ch][0], 128 * sizeof(float));
            memcpy(channel_base_ptr + 128 + 128, &PcmBufHi[ch][0], 256 * sizeof(float));
        }
        
        return result;
    }
};

// ==================== File-based Encoder/Decoder ====================

class PyAtrac1FileEncoder {
private:
    std::unique_ptr<TAtrac1Encoder> encoder;
    std::unique_ptr<TPCMEngine> pcmEngine;
    std::unique_ptr<TWav> wavInput;
    TCompressedOutputPtr aeaOutput;
    uint64_t totalSamples;
    uint32_t numChannels;

public:
    PyAtrac1FileEncoder(const std::string& inputFile, const std::string& outputFile,
                        const TAtrac1EncodeSettings& settings) {
        
        wavInput = std::make_unique<TWav>(inputFile);
        if (wavInput->GetSampleRate() != ATRAC_SAMPLE_RATE) {
            throw std::runtime_error("Unsupported sample rate. Only 44100 Hz is supported.");
        }

        numChannels = wavInput->GetChannelNum();
        totalSamples = wavInput->GetTotalSamples();

        const uint64_t numFrames = numChannels * totalSamples / ATRAC_NUM_SAMPLES;
        if (numFrames >= UINT32_MAX) {
            throw std::runtime_error("Input file too large for ATRAC1 format");
        }

        aeaOutput = CreateAeaOutput(outputFile, "pytrac", numChannels, (uint32_t)numFrames);

        pcmEngine = std::make_unique<TPCMEngine>(4096, numChannels,
            TPCMEngine::TReaderPtr(wavInput->GetPCMReader()));

        encoder = std::make_unique<TAtrac1Encoder>(std::move(aeaOutput),
            TAtrac1EncodeSettings(settings));
    }

    void encode() {
        auto lambda = encoder->GetLambda();
        while (totalSamples > (pcmEngine->ApplyProcess(ATRAC_NUM_SAMPLES, lambda))) {
            // Continue processing
        }
    }

    std::tuple<uint32_t, uint64_t, uint32_t> getInfo() const {
        return std::make_tuple(numChannels, totalSamples, wavInput->GetSampleRate());
    }
};

class PyAtrac1FileDecoder {
private:
    std::unique_ptr<TAtrac1Decoder> decoder;
    std::unique_ptr<TPCMEngine> pcmEngine;
    std::unique_ptr<TWav> wavOutput;
    TCompressedInputPtr aeaInput;
    uint64_t totalSamples;
    uint32_t numChannels;

public:
    PyAtrac1FileDecoder(const std::string& inputFile, const std::string& outputFile) {
        aeaInput = CreateAeaInput(inputFile);
        totalSamples = aeaInput->GetLengthInSamples();
        numChannels = aeaInput->GetChannelNum();

        wavOutput = std::make_unique<TWav>(outputFile, numChannels, ATRAC_SAMPLE_RATE);

        pcmEngine = std::make_unique<TPCMEngine>(4096, numChannels,
            TPCMEngine::TWriterPtr(wavOutput->GetPCMWriter()));

        decoder = std::make_unique<TAtrac1Decoder>(std::move(aeaInput));
    }

    void decode() {
        auto lambda = decoder->GetLambda();
        while (totalSamples > (pcmEngine->ApplyProcess(ATRAC_NUM_SAMPLES, lambda))) {
            // Continue processing
        }
    }

    std::tuple<uint32_t, uint64_t, std::string> getInfo() const {
        return std::make_tuple(numChannels, totalSamples, "ATRAC1");
    }
};

// ==================== Utility Functions ====================

void encode_file(const std::string& inputFile, const std::string& outputFile,
                 uint32_t bfuIdxConst = 0, bool fastBfuNumSearch = false,
                 const std::string& windowMode = "auto", uint32_t windowMask = 0) {
    
    TAtrac1EncodeSettings::EWindowMode wMode = TAtrac1EncodeSettings::EWindowMode::EWM_AUTO;
    if (windowMode == "notransient") {
        wMode = TAtrac1EncodeSettings::EWindowMode::EWM_NOTRANSIENT;
    }
    
    TAtrac1EncodeSettings settings(bfuIdxConst, fastBfuNumSearch, wMode, windowMask);
    PyAtrac1FileEncoder encoder(inputFile, outputFile, settings);
    encoder.encode();
}

void decode_file(const std::string& inputFile, const std::string& outputFile) {
    PyAtrac1FileDecoder decoder(inputFile, outputFile);
    decoder.decode();
}

// ==================== Utility Functions ====================

py::array_t<float> quantize_values(py::array_t<float> values, py::array_t<uint32_t> bits_per_value) {
    auto valuesBuf = values.request();
    auto bitsBuf = bits_per_value.request();
    
    if (valuesBuf.size != bitsBuf.size) {
        throw std::runtime_error("Values and bits arrays must have same size");
    }
    
    auto result = py::array_t<float>(valuesBuf.size);
    float* valPtr = static_cast<float*>(valuesBuf.ptr);
    uint32_t* bitsPtr = static_cast<uint32_t*>(bitsBuf.ptr);
    float* resPtr = static_cast<float*>(result.template mutable_unchecked<1>().mutable_data(0));
    
    for (py::ssize_t i = 0; i < valuesBuf.size; ++i) {
        if (bitsPtr[i] > 0) {
            const float multiple = static_cast<float>((1 << (bitsPtr[i] - 1)) - 1);
            int32_t qVal = static_cast<int32_t>(std::round(valPtr[i] * multiple));
            qVal = std::max(-(1 << (bitsPtr[i] - 1)), std::min((1 << (bitsPtr[i] - 1)) - 1, qVal));
            resPtr[i] = (multiple > 0) ? (static_cast<float>(qVal) / multiple) : 0.0f;
        } else {
            resPtr[i] = 0.0f;
        }
    }
    
    return result;
}

py::array_t<int32_t> quantize_to_integers(py::array_t<float> values, py::array_t<uint32_t> bits_per_value) {
    auto valuesBuf = values.request();
    auto bitsBuf = bits_per_value.request();
    
    if (valuesBuf.size != bitsBuf.size) {
        throw std::runtime_error("Values and bits arrays must have same size");
    }
    
    auto result = py::array_t<int32_t>(valuesBuf.size);
    float* valPtr = static_cast<float*>(valuesBuf.ptr);
    uint32_t* bitsPtr = static_cast<uint32_t*>(bitsBuf.ptr);
    int32_t* resPtr = static_cast<int32_t*>(result.template mutable_unchecked<1>().mutable_data(0));
    
    for (py::ssize_t i = 0; i < valuesBuf.size; ++i) {
        if (bitsPtr[i] > 0) {
            const float multiple = static_cast<float>((1 << (bitsPtr[i] - 1)) - 1);
            int32_t qVal = static_cast<int32_t>(std::round(valPtr[i] * multiple));
            resPtr[i] = std::max(-(1 << (bitsPtr[i] - 1)), std::min((1 << (bitsPtr[i] - 1)) - 1, qVal));
        } else {
            resPtr[i] = 0;
        }
    }
    
    return result;
}

py::array_t<float> dequantize_integers(py::array_t<int32_t> quantized, py::array_t<uint32_t> bits_per_value) {
    auto quantBuf = quantized.request();
    auto bitsBuf = bits_per_value.request();
    
    if (quantBuf.size != bitsBuf.size) {
        throw std::runtime_error("Quantized and bits arrays must have same size");
    }
    
    auto result = py::array_t<float>(quantBuf.size);
    int32_t* quantPtr = static_cast<int32_t*>(quantBuf.ptr);
    uint32_t* bitsPtr = static_cast<uint32_t*>(bitsBuf.ptr);
    float* resPtr = static_cast<float*>(result.template mutable_unchecked<1>().mutable_data(0));
    
    for (py::ssize_t i = 0; i < quantBuf.size; ++i) {
        if (bitsPtr[i] > 0) {
            const float multiple = static_cast<float>((1 << (bitsPtr[i] - 1)) - 1);
            resPtr[i] = (multiple > 0) ? (static_cast<float>(quantPtr[i]) / multiple) : 0.0f;
        } else {
            resPtr[i] = 0.0f;
        }
    }
    
    return result;
}

// Get ATRAC1 constants
py::dict get_atrac_constants() {
    py::dict constants;
    constants["num_samples"] = ATRAC_NUM_SAMPLES;
    constants["max_bfus"] = ATRAC_MAX_BFUS;
    constants["num_qmf"] = ATRAC_NUM_QMF;
    constants["sample_rate"] = ATRAC_SAMPLE_RATE;
    
    // Add BFU structure information
    py::list specs_per_block, blocks_per_band;
    for (int i = 0; i < ATRAC_MAX_BFUS; i++) {
        specs_per_block.append(TAtrac1Data::SpecsPerBlock[i]);
    }
    for (int i = 0; i <= ATRAC_NUM_QMF; i++) {
        blocks_per_band.append(TAtrac1Data::BlocksPerBand[i]);
    }
    
    constants["specs_per_block"] = specs_per_block;
    constants["blocks_per_band"] = blocks_per_band;
    
    return constants;
}

// ==================== Bitstream Generation/Parsing ====================

class PyBitstreamWriter {
private:
    std::vector<uint8_t> bitstreamData;
    
public:
    std::vector<uint8_t> writeFrame(
        const std::vector<uint32_t>& windowMasks,
        const std::vector<std::vector<uint8_t>>& scaleFactorIndices,
        const std::vector<std::vector<uint32_t>>& bitsPerBfu,
        const std::vector<std::vector<std::vector<int32_t>>>& quantizedValues) {
        
        // Create a simple frame bitstream
        // This is a simplified version - full implementation would use TAtrac1BitStreamWriter
        bitstreamData.clear();
        
        // Frame header - simplified
        bitstreamData.push_back(0x80); // Frame sync
        
        // Window masks for each channel
        for (size_t ch = 0; ch < windowMasks.size(); ++ch) {
            bitstreamData.push_back(static_cast<uint8_t>(windowMasks[ch] & 0x07));
        }
        
        // BFU count and scale factors
        for (size_t ch = 0; ch < scaleFactorIndices.size(); ++ch) {
            bitstreamData.push_back(static_cast<uint8_t>(scaleFactorIndices[ch].size()));
            for (uint8_t sfi : scaleFactorIndices[ch]) {
                bitstreamData.push_back(sfi);
            }
        }
        
        // Bit allocation
        for (size_t ch = 0; ch < bitsPerBfu.size(); ++ch) {
            for (uint32_t bits : bitsPerBfu[ch]) {
                bitstreamData.push_back(static_cast<uint8_t>(bits & 0x0F));
            }
        }
        
        // Quantized values (simplified packing)
        for (size_t ch = 0; ch < quantizedValues.size(); ++ch) {
            for (const auto& bfuValues : quantizedValues[ch]) {
                bitstreamData.push_back(static_cast<uint8_t>(bfuValues.size()));
                for (int32_t val : bfuValues) {
                    // Pack as 16-bit values (simplified)
                    bitstreamData.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
                    bitstreamData.push_back(static_cast<uint8_t>(val & 0xFF));
                }
            }
        }
        
        return bitstreamData;
    }
};

struct ParsedBitstreamData {
    std::vector<uint32_t> windowMasks;
    std::vector<std::vector<uint8_t>> scaleFactorIndices;
    std::vector<std::vector<uint32_t>> bitsPerBfu;
    std::vector<std::vector<std::vector<int32_t>>> quantizedValues;
};

class PyBitstreamParser {
public:
    ParsedBitstreamData parseFrame(const std::vector<uint8_t>& frameData) {
        ParsedBitstreamData result;
        
        if (frameData.empty()) {
            throw std::runtime_error("Empty frame data");
        }
        
        size_t pos = 0;
        
        // Skip frame sync
        if (frameData[pos] != 0x80) {
            throw std::runtime_error("Invalid frame sync");
        }
        pos++;
        
        // Read window masks (assuming stereo for now)
        uint32_t numChannels = 2;
        result.windowMasks.resize(numChannels);
        for (uint32_t ch = 0; ch < numChannels && pos < frameData.size(); ++ch) {
            result.windowMasks[ch] = frameData[pos] & 0x07;
            pos++;
        }
        
        // Read scale factors
        result.scaleFactorIndices.resize(numChannels);
        for (uint32_t ch = 0; ch < numChannels && pos < frameData.size(); ++ch) {
            uint8_t bfuCount = frameData[pos++];
            result.scaleFactorIndices[ch].resize(bfuCount);
            for (uint8_t bfu = 0; bfu < bfuCount && pos < frameData.size(); ++bfu) {
                result.scaleFactorIndices[ch][bfu] = frameData[pos++];
            }
        }
        
        // Read bit allocation
        result.bitsPerBfu.resize(numChannels);
        for (uint32_t ch = 0; ch < numChannels; ++ch) {
            size_t bfuCount = result.scaleFactorIndices[ch].size();
            result.bitsPerBfu[ch].resize(bfuCount);
            for (size_t bfu = 0; bfu < bfuCount && pos < frameData.size(); ++bfu) {
                result.bitsPerBfu[ch][bfu] = frameData[pos++] & 0x0F;
            }
        }
        
        // Read quantized values
        result.quantizedValues.resize(numChannels);
        for (uint32_t ch = 0; ch < numChannels; ++ch) {
            size_t bfuCount = result.scaleFactorIndices[ch].size();
            result.quantizedValues[ch].resize(bfuCount);
            for (size_t bfu = 0; bfu < bfuCount && pos < frameData.size(); ++bfu) {
                if (pos >= frameData.size()) break;
                uint8_t valueCount = frameData[pos++];
                result.quantizedValues[ch][bfu].resize(valueCount);
                for (uint8_t v = 0; v < valueCount && pos + 1 < frameData.size(); ++v) {
                    int16_t val = (static_cast<int16_t>(frameData[pos]) << 8) | frameData[pos + 1];
                    result.quantizedValues[ch][bfu][v] = static_cast<int32_t>(val);
                    pos += 2;
                }
            }
        }
        
        return result;
    }
};

// ==================== Python Module Definition ====================

PYBIND11_MODULE(pytrac, m) {
    m.doc() = "Python bindings for ATRAC1 audio codec";

    // Bind TAtrac1NNFrameParameters
    py::class_<NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters>(m, "NNFrameParameters")
        .def(py::init<>(), "Initializes an empty NNFrameParameters object.")
        .def_readwrite("block_mode", &NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters::BlockMode,
                       "BlockSizeMod object indicating windowing for low, mid, hi bands.")
        .def_readwrite("bfu_amount_table_index", &NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters::BfuAmountTableIndex,
                       "Index (0-7) into TAtrac1Data::BfuAmountTab, determines active BFUs.")
        .def_readwrite("word_lengths", &NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters::WordLengths,
                       "List of actual bits (0-16) to use for quantizing each active BFU.")
        .def_readwrite("scale_factor_indices", &NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters::ScaleFactorIndices,
                       "List of scale factor indices (0-63) for each active BFU.")
        .def_readwrite("quantized_spectrum", &NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters::QuantizedSpectrum,
                       "List of lists. Outer list per BFU, inner list has integer quantized MDCT coefficients for that BFU.");

    // Bind TAtrac1NNBitStreamAssembler
    py::class_<NAtracDEnc::NAtrac1::TAtrac1NNBitStreamAssembler>(m, "NNBitStreamAssembler")
        .def(py::init<>(), "Initializes the NN bitstream assembler.")
        .def("assemble_channel_payload", &NAtracDEnc::NAtrac1::TAtrac1NNBitStreamAssembler::AssembleChannelPayload,
             "Assembles a single ATRAC channel payload from NN-predicted parameters. "
             "Returns a list of characters (bytes).",
             py::arg("params").noconvert());

    // Python-friendly helper functions for mono/stereo assembly
    m.def("assemble_mono_frame_payload",
          [](const NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters& params_ch0) {
              NAtracDEnc::NAtrac1::TAtrac1NNBitStreamAssembler assembler;
              std::vector<char> payload_vector = assembler.AssembleChannelPayload(params_ch0);
              return py::bytes(payload_vector.data(), payload_vector.size());
          },
          "Assembles ATRAC frame payload for a single (mono) channel from NN-predicted parameters.",
          py::arg("params_ch0").noconvert());

    m.def("assemble_stereo_frame_payloads",
          [](const NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters& params_ch0,
             const NAtracDEnc::NAtrac1::TAtrac1NNFrameParameters& params_ch1) {
              NAtracDEnc::NAtrac1::TAtrac1NNBitStreamAssembler assembler;
              py::list stereo_payloads;

              std::vector<char> payload_vector_ch0 = assembler.AssembleChannelPayload(params_ch0);
              stereo_payloads.append(py::bytes(payload_vector_ch0.data(), payload_vector_ch0.size()));

              std::vector<char> payload_vector_ch1 = assembler.AssembleChannelPayload(params_ch1);
              stereo_payloads.append(py::bytes(payload_vector_ch1.data(), payload_vector_ch1.size()));

              return stereo_payloads;
          },
          "Assembles ATRAC frame payloads for stereo channels from NN-predicted parameters.",
          py::arg("params_ch0").noconvert(), py::arg("params_ch1").noconvert());

    // Constants
    // Expose BFU amount table
    py::list bfu_amount_tab_list;
    for(int i = 0; i < 8; ++i) {
        bfu_amount_tab_list.append(NAtracDEnc::NAtrac1::TAtrac1Data::BfuAmountTab[i]);
    }
    m.attr("BFU_AMOUNT_TABLE") = bfu_amount_tab_list;

    m.attr("NUM_SAMPLES") = ATRAC_NUM_SAMPLES;
    m.attr("SAMPLE_RATE") = ATRAC_SAMPLE_RATE;
    m.attr("MAX_BFUS") = ATRAC_MAX_BFUS;
    m.attr("NUM_QMF") = ATRAC_NUM_QMF;

    // SpecsStartLong
    py::list specs_start_long_list;
    for(int i = 0; i < NAtrac1::TAtrac1Data::MaxBfus; ++i) {
        specs_start_long_list.append(NAtrac1::TAtrac1Data::SpecsStartLong[i]);
    }
    m.attr("SPECS_START_LONG") = specs_start_long_list;

    // SpecsStartShort
    py::list specs_start_short_list;
    for(int i = 0; i < NAtrac1::TAtrac1Data::MaxBfus; ++i) {
        specs_start_short_list.append(NAtrac1::TAtrac1Data::SpecsStartShort[i]);
    }
    m.attr("SPECS_START_SHORT") = specs_start_short_list;

    py::list specs_per_block_list;
    for(int i = 0; i < NAtrac1::TAtrac1Data::MaxBfus; ++i) {
        specs_per_block_list.append(NAtrac1::TAtrac1Data::SpecsPerBlock[i]);
    }
    m.attr("SPECS_PER_BLOCK") = specs_per_block_list;

    py::list blocks_per_band_list;
    for(int i = 0; i <= NAtrac1::TAtrac1Data::NumQMF; ++i) {
        blocks_per_band_list.append(NAtrac1::TAtrac1Data::BlocksPerBand[i]);
    }
    m.attr("BLOCKS_PER_BAND") = blocks_per_band_list;

    py::list bfu_to_band_list;
    for(uint32_t i = 0; i < NAtrac1::TAtrac1Data::MaxBfus; ++i) {
        bfu_to_band_list.append(NAtrac1::TAtrac1Data::BfuToBand(i)); // Call as function
    }
    m.attr("BFU_TO_BAND") = bfu_to_band_list;
    
    py::list scale_table_list;
    for(int i = 0; i < 64; ++i) { 
        scale_table_list.append(NAtrac1::TAtrac1Data::ScaleTable[i]);
    }
    m.attr("SCALE_FACTOR_TABLE") = scale_table_list; 

    // Main encoding/decoding functions
    m.def("encode_file", &encode_file,
          "Encode WAV file to ATRAC1 format",
          py::arg("input_file"), py::arg("output_file"),
          py::arg("bfu_idx_const") = 0, py::arg("fast_bfu_search") = false,
          py::arg("window_mode") = "auto", py::arg("window_mask") = 0);

    m.def("decode_file", &decode_file,
          "Decode ATRAC1 file to WAV format",
          py::arg("input_file"), py::arg("output_file"));

    // Utility functions
    m.def("quantize_values", &quantize_values,
          "Quantize float values using specified bits per value",
          py::arg("values"), py::arg("bits_per_value"));
    
    m.def("quantize_to_integers", &quantize_to_integers,
          "Quantize float values to integers using specified bits per value",
          py::arg("values"), py::arg("bits_per_value"));
    
    m.def("dequantize_integers", &dequantize_integers,
          "Dequantize integer values back to floats",
          py::arg("quantized"), py::arg("bits_per_value"));
    
    m.def("get_atrac_constants", &get_atrac_constants,
          "Get all ATRAC1 constants and structure information");

    // TScaledBlock for intermediate data access
    py::class_<TScaledBlock>(m, "ScaledBlock")
        .def(py::init<uint8_t>(), py::arg("scale_factor_index"))
        .def_readwrite("scale_factor_index", &TScaledBlock::ScaleFactorIndex)
        .def_readwrite("values", &TScaledBlock::Values)
        .def_readwrite("max_energy", &TScaledBlock::MaxEnergy);

    // Intermediate data structure
    py::class_<PyEncoderIntermediateData>(m, "EncoderIntermediateData")
        .def(py::init<>())
        .def_readwrite("qmf_low", &PyEncoderIntermediateData::qmfLow)
        .def_readwrite("qmf_mid", &PyEncoderIntermediateData::qmfMid)
        .def_readwrite("qmf_hi", &PyEncoderIntermediateData::qmfHi)
        .def_readwrite("mdct_specs", &PyEncoderIntermediateData::mdctSpecs)
        .def_readwrite("window_mask", &PyEncoderIntermediateData::windowMask)
        .def_readwrite("block_size_log_count", &PyEncoderIntermediateData::block_size_log_count)
        .def_readwrite("scaled_blocks", &PyEncoderIntermediateData::scaledBlocks)
        .def_readwrite("bits_per_bfu", &PyEncoderIntermediateData::bitsPerBfu)
        .def_readwrite("quantized_values", &PyEncoderIntermediateData::quantizedValues)
        .def_readwrite("quantization_error", &PyEncoderIntermediateData::quantizationError)
        .def_readwrite("pcm_input", &PyEncoderIntermediateData::pcmInput)
        .def_readwrite("compressed_data_per_channel", &PyEncoderIntermediateData::compressedDataPerChannel)
        .def_readonly("enc_log_count_low", &PyEncoderIntermediateData::enc_log_count_low)
        .def_readonly("enc_log_count_mid", &PyEncoderIntermediateData::enc_log_count_mid)
        .def_readonly("enc_log_count_hi", &PyEncoderIntermediateData::enc_log_count_hi);
 
    // Decoder intermediate data structure
    py::class_<PyDecoderIntermediateData>(m, "DecoderIntermediateData")
        .def(py::init<>())
        .def_readwrite("compressed_input", &PyDecoderIntermediateData::compressedInput)
        .def_readwrite("window_mask", &PyDecoderIntermediateData::windowMask)
        .def_readwrite("block_size_log_count", &PyDecoderIntermediateData::block_size_log_count)
        .def_readwrite("bits_per_bfu", &PyDecoderIntermediateData::bitsPerBfu)
        .def_readwrite("parsed_quantized_values", &PyDecoderIntermediateData::parsed_quantized_values)
        .def_readwrite("placeholder_quantized_ints_from_encoder_data", &PyDecoderIntermediateData::placeholder_quantized_ints_from_encoder_data)
        .def_readwrite("placeholder_dequantized_floats_from_encoder_data", &PyDecoderIntermediateData::placeholder_dequantized_floats_from_encoder_data)
        .def_readwrite("scale_factor_indices", &PyDecoderIntermediateData::scaleFactorIndices)
        .def_readwrite("mdct_specs", &PyDecoderIntermediateData::mdctSpecs)
        .def_readwrite("qmf_low", &PyDecoderIntermediateData::qmfLow)
        .def_readwrite("qmf_mid", &PyDecoderIntermediateData::qmfMid)
        .def_readwrite("qmf_hi", &PyDecoderIntermediateData::qmfHi)
        .def_readwrite("pcm_output", &PyDecoderIntermediateData::pcmOutput);

    // Encode settings class
    py::class_<TAtrac1EncodeSettings>(m, "EncodeSettings")
        .def(py::init<>())
        .def(py::init<uint32_t, bool, TAtrac1EncodeSettings::EWindowMode, uint32_t>(),
             py::arg("bfu_idx_const"), py::arg("fast_bfu_search"),
             py::arg("window_mode"), py::arg("window_mask"))
        .def("get_bfu_idx_const", &TAtrac1EncodeSettings::GetBfuIdxConst)
        .def("get_fast_bfu_search", &TAtrac1EncodeSettings::GetFastBfuNumSearch)
        .def("get_window_mode", &TAtrac1EncodeSettings::GetWindowMode)
        .def("get_window_mask", &TAtrac1EncodeSettings::GetWindowMask);

    // Window mode enum
    py::enum_<TAtrac1EncodeSettings::EWindowMode>(m, "WindowMode")
        .value("NO_TRANSIENT", TAtrac1EncodeSettings::EWindowMode::EWM_NOTRANSIENT)
        .value("AUTO", TAtrac1EncodeSettings::EWindowMode::EWM_AUTO);

    // Frame processor
    py::class_<PyAtrac1FrameProcessor>(m, "FrameProcessor")
        .def(py::init<uint32_t, const NAtracDEnc::NAtrac1::TAtrac1EncodeSettings&>(),
             py::arg("channels"), py::arg("settings") = NAtracDEnc::NAtrac1::TAtrac1EncodeSettings(),
             "Create frame processor for capturing intermediate ATRAC1 encoding stages. "
             "Optionally accepts EncodeSettings.")
        .def("process_frame", &PyAtrac1FrameProcessor::processFrame,
             "Process single audio frame and return all intermediate data",
             py::arg("input"));

    // Frame decoder
    py::class_<PyAtrac1FrameDecoder>(m, "FrameDecoder")
        .def(py::init<uint32_t>(), py::arg("channels"),
             "Create frame decoder for capturing intermediate ATRAC1 decoding stages")
        .def("decode_from_intermediate", &PyAtrac1FrameDecoder::decodeFromIntermediate,
             "Decode from encoder intermediate data to create training pairs",
             py::arg("encoder_data"))
        .def("decode_frame_from_bitstream",
             [](PyAtrac1FrameDecoder &self, const py::list& per_channel_py_bytes_list) {
                 std::vector<std::vector<char>> cpp_bitstream_data;
                 for (const auto& py_bytes_handle : per_channel_py_bytes_list) {
                     if (!py::isinstance<py::bytes>(py_bytes_handle)) {
                         throw py::type_error("Expected a list of bytes objects.");
                     }
                     py::bytes py_b = py_bytes_handle.cast<py::bytes>();
                     std::string s = py_b; // py::bytes can cast to std::string
                     cpp_bitstream_data.push_back(std::vector<char>(s.begin(), s.end()));
                 }
                 return self.decode_frame_from_bitstream(cpp_bitstream_data);
             },
             "Decode a frame from raw per-channel bitstream data (list of Python bytes objects).",
             py::arg("per_channel_bitstream_data"))
        .def("qmf_to_pcm", &PyAtrac1FrameDecoder::qmfToPcm,
             "Convert QMF bands to PCM", py::arg("qmf_data"))
        .def("mdct_to_qmf", &PyAtrac1FrameDecoder::mdctToQmf,
             "Convert MDCT to QMF bands", py::arg("mdct_data"), py::arg("window_masks"));

    // Bitstream generation and parsing
    py::class_<PyBitstreamWriter>(m, "BitstreamWriter")
        .def(py::init<>(), "Create bitstream writer")
        .def("write_frame", &PyBitstreamWriter::writeFrame,
             "Write frame data to bitstream bytes",
             py::arg("window_masks"), py::arg("scale_factor_indices"), 
             py::arg("bits_per_bfu"), py::arg("quantized_values"));

    py::class_<ParsedBitstreamData>(m, "ParsedBitstreamData")
        .def(py::init<>())
        .def_readwrite("window_masks", &ParsedBitstreamData::windowMasks)
        .def_readwrite("scale_factor_indices", &ParsedBitstreamData::scaleFactorIndices)
        .def_readwrite("bits_per_bfu", &ParsedBitstreamData::bitsPerBfu)
        .def_readwrite("quantized_values", &ParsedBitstreamData::quantizedValues);

    py::class_<PyBitstreamParser>(m, "BitstreamParser")
        .def(py::init<>(), "Create bitstream parser")
        .def("parse_frame", &PyBitstreamParser::parseFrame,
             "Parse bitstream bytes into frame data",
             py::arg("frame_data"));

    // Block size modifier for window control
    py::class_<TAtrac1Data::TBlockSizeMod>(m, "BlockSizeMod")
        .def(py::init<bool, bool, bool>(),
             py::arg("low_short"), py::arg("mid_short"), py::arg("hi_short"))
        .def(py::init<>())
        .def("short_win", &TAtrac1Data::TBlockSizeMod::ShortWin,
             "Check if band uses short window", py::arg("band"))
        .def_readwrite("log_count", &TAtrac1Data::TBlockSizeMod::LogCount);

    // Transient detector
    py::class_<TTransientDetector>(m, "TransientDetector")
        .def(py::init<uint32_t, uint32_t>(), py::arg("pos"), py::arg("block_size"))
        .def("detect", &TTransientDetector::Detect,
             "Detect transients in audio buffer", py::arg("buf"));

    // File-based encoder
    py::class_<PyAtrac1FileEncoder>(m, "FileEncoder")
        .def(py::init<const std::string&, const std::string&, const TAtrac1EncodeSettings&>(),
             py::arg("input_file"), py::arg("output_file"), py::arg("settings"),
             "Create file-based encoder")
        .def("encode", &PyAtrac1FileEncoder::encode,
             "Encode input file to output file")
        .def("get_info", &PyAtrac1FileEncoder::getInfo,
             "Get (channels, total_samples, sample_rate) tuple");

    // File-based decoder
    py::class_<PyAtrac1FileDecoder>(m, "FileDecoder")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("input_file"), py::arg("output_file"),
             "Create file-based decoder")
        .def("decode", &PyAtrac1FileDecoder::decode,
             "Decode input file to output file")
        .def("get_info", &PyAtrac1FileDecoder::getInfo,
             "Get (channels, total_samples, codec_name) tuple");

    // Version info
    m.attr("__version__") = "1.0.0";
}
