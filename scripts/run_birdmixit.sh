mixit_path="/home/jupyter/sound-separation/"
info_path="/home/jupyter/data/voxaboxen_data/OZF_slowed_0.5/formatted/mixit_manifest.csv"

cd ${mixit_path}
python3 ${mixit_path}models/tools/process_wav_stitching_opt.py \
    --model_dir ${mixit_path}weights/bird_mixit_model_checkpoints/output_sources4 \
    --checkpoint ${mixit_path}weights/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
    --block_size_in_seconds 10 --permutation_invariant True --window_type vorbis \
    --info ${info_path} \
    --sample_rate 22050


for DATASET in overlap_0_slowed_0.5;
#overlap_0.2_slowed_0.5 overlap_0.4_slowed_0.5 overlap_0.6_slowed_0.5 overlap_0.8_slowed_0.5 overlap_1_slowed_0.5;
do
mixit_path="/home/jupyter/sound-separation/"
info_path="/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/mixit_manifest.csv"

cd ${mixit_path}
python3 ${mixit_path}models/tools/process_wav_stitching_opt.py \
    --model_dir ${mixit_path}weights/bird_mixit_model_checkpoints/output_sources4 \
    --checkpoint ${mixit_path}weights/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
    --block_size_in_seconds 10 --permutation_invariant True --window_type vorbis \
    --info ${info_path} \
    --sample_rate 22050
done

