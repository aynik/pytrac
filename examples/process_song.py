import numpy as np
import soundfile as sf
import pytrac
import sys
import os

def process_song_pytrac(input_wav_path, output_wav_path, settings_dict=None):
    print(f"Loading input: {input_wav_path}")
    audio_data_orig, sr = sf.read(input_wav_path, dtype='float32')
    
    if sr != 44100:
        print(f"Error: Input sample rate is {sr} Hz. Must be 44100 Hz for ATRAC1.")
        return

    if audio_data_orig.ndim == 1:
        num_channels = 1
        audio_data_for_processing = audio_data_orig.reshape(1, -1)
        print(f"Input is MONO, {audio_data_for_processing.shape[1]} samples.")
    elif audio_data_orig.ndim == 2:
        num_channels = audio_data_orig.shape[1]
        audio_data_for_processing = audio_data_orig.T 
        print(f"Input is STEREO, {audio_data_for_processing.shape[1]} samples per channel.")
        if num_channels > 2:
            print("Warning: Input has more than 2 channels. Only processing first 2.")
            num_channels = 2
            audio_data_for_processing = audio_data_for_processing[:2, :]
    else:
        print("Error: Unsupported audio format (ndim > 2).")
        return

    if settings_dict:
        bfu_idx_const = settings_dict.get("bfu_idx_const", 0)
        fast_bfu_search = settings_dict.get("fast_bfu_search", False)
        window_mode_str = settings_dict.get("window_mode", "auto")
        window_mask = settings_dict.get("window_mask", 0)

        pytrac_window_mode = pytrac.WindowMode.AUTO
        if window_mode_str.lower() == "no_transient":
             pytrac_window_mode = pytrac.WindowMode.NO_TRANSIENT
        
        encode_settings = pytrac.EncodeSettings(
            bfu_idx_const, fast_bfu_search, pytrac_window_mode, window_mask
        )
        print(f"Using custom encode settings: {settings_dict}")
    else:
        encode_settings = pytrac.EncodeSettings()
        print("Using default encode settings.")


    encoder = pytrac.FrameProcessor(num_channels, encode_settings)
    decoder = pytrac.FrameDecoder(num_channels)
    
    num_samples_total = audio_data_for_processing.shape[1]
    num_frames = num_samples_total // pytrac.NUM_SAMPLES
    
    all_decoded_pcm_ch_lists = [[] for _ in range(num_channels)]

    print(f"Processing {num_frames} frames...")
    for i in range(num_frames):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_frames} frames...")

        start_idx = i * pytrac.NUM_SAMPLES
        end_idx = start_idx + pytrac.NUM_SAMPLES
        
        current_frame_list_of_lists = [ch_data.tolist() for ch_data in audio_data_for_processing[:, start_idx:end_idx]]
        
        enc_intermediate = encoder.process_frame(current_frame_list_of_lists)
        per_channel_bitstream = enc_intermediate.compressed_data_per_channel
        dec_intermediate = decoder.decode_frame_from_bitstream(per_channel_bitstream)
        
        for ch in range(num_channels):
            all_decoded_pcm_ch_lists[ch].extend(dec_intermediate.pcm_output[ch])
    print("Finished processing frames.")

    # Convert lists of decoded samples to numpy arrays
    final_decoded_pcm_np_channels = [np.array(ch_data, dtype=np.float32) for ch_data in all_decoded_pcm_ch_lists]

    if num_channels == 1:
        output_audio = np.clip(final_decoded_pcm_np_channels[0], -1.0, 1.0)
    else: # Stereo
        ch0_clipped = np.clip(final_decoded_pcm_np_channels[0], -1.0, 1.0)
        ch1_clipped = np.clip(final_decoded_pcm_np_channels[1], -1.0, 1.0)
        output_audio = np.stack((ch0_clipped, ch1_clipped), axis=-1) # For soundfile (samples, channels)

    print(f"Writing output: {output_wav_path}")
    sf.write(output_wav_path, output_audio, sr, subtype='FLOAT') # Or 'PCM_16'
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_song.py <input_wav_path> <output_wav_path> [settings_json_string]")
        print("Example settings_json_string: '{\"window_mode\": \"no_transient\"}'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    custom_settings = None
    if len(sys.argv) > 3:
        import json
        try:
            custom_settings = json.loads(sys.argv[3])
        except json.JSONDecodeError:
            print("Error: Invalid JSON string for settings.")
            sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    process_song_pytrac(input_file, output_file, custom_settings)