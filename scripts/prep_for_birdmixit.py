import soundfile as sf
import torchaudio.transforms as T
import torchaudio
import torch
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from glob import glob

MIXIT_SR = 22050  # BirdMixit was trained on 22050 kHz audio

FOLDERS = [f"/home/jupyter/data/voxaboxen_data/OZF_synthetic/overlap_{x}_slowed_0.5" for x in ["0", "0.2", "0.4", "0.6", "0.8", "1"]]
FOLDERS = FOLDERS + ['/home/jupyter/data/voxaboxen_data/OZF_slowed_0.5/formatted']

for FOLDER in FOLDERS:
    
    AUDIO_FOLDER = os.path.join(FOLDER, 'audio')
    AUDIO_FOLDER_PROCESSED = os.path.join(FOLDER, 'audio_22050')

    if not os.path.exists(AUDIO_FOLDER_PROCESSED):
        os.makedirs(AUDIO_FOLDER_PROCESSED)

    assumed_orig_sr = 16000
    resampler = T.Resample(orig_freq=assumed_orig_sr, new_freq=MIXIT_SR)
    count_global = 0
    len_global = 0

    def process_file(fp):
        try:
            x, sr = torchaudio.load(fp)
            if len(x.shape) > 1:
                x = torch.mean(x, dim = 0) 

            if x.size(0) < 100:
                return

            if sr == assumed_orig_sr:
                x = resampler(x)
            else:
                x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=MIXIT_SR)

            new_fn = os.path.basename(fp).replace('.flac', '.wav')
            new_fp = os.path.join(AUDIO_FOLDER_PROCESSED, new_fn)

            sf.write(new_fp, x.squeeze().cpu().numpy(), MIXIT_SR)
            print("resampled and wrote.")
        except Exception as e:
            print(f"Error processing {fp}: {e}")

    # Load file paths from a text file
    input_file_list = os.path.join(FOLDER, 'test_info.csv')
    files = sorted(pd.read_csv(input_file_list)['audio_fp'])
    len_global = len(files)

    # Use ThreadPoolExecutor for parallel processing with tqdm
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, fp) for fp in files]

        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            # This line just ensures that exceptions are raised, allowing tqdm to update correctly
            future.result()

    ## Make manifest
    unseparated_audio_dir = AUDIO_FOLDER_PROCESSED
    separated_audio_dir = AUDIO_FOLDER_PROCESSED.replace('audio_22050', 'audio_mixit')

    if not os.path.exists(separated_audio_dir):
        os.makedirs(separated_audio_dir)

    audio_fps = sorted(glob(os.path.join(unseparated_audio_dir, "*.wav")))
    output_audio_fps = [os.path.join(separated_audio_dir, os.path.basename(x)) for x in audio_fps]

    temp_df = pd.DataFrame({'input_fp' : audio_fps, 'output_fp' : output_audio_fps})
    temp_df_fp = os.path.join(FOLDER, f"mixit_manifest.csv")
    temp_df.to_csv(temp_df_fp)
    
    test_info_mixit_df = pd.DataFrame({"fn" : [os.path.basename(x)[:-4] for x in output_audio_fps], "audio_fp" : output_audio_fps, "selection_table_fp" : [x.replace("audio_mixit", "selection_tables").replace(".wav", ".txt") for x in output_audio_fps]})
    test_info_mixit_df.to_csv(os.path.join(FOLDER, "mixit_test_info.csv"))
