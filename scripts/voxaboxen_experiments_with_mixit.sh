cd ..

MODEL=aves;

for DATASET in overlap_0_slowed_0.5 overlap_0.2_slowed_0.5 overlap_0.4_slowed_0.5 overlap_0.6_slowed_0.5 overlap_0.8_slowed_0.5 overlap_1_slowed_0.5;
do
python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/mixit_test_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/OZF_synthetic/${DATASET}/mixit_test_info.csv --project-dir=projects/${DATASET}_${MODEL}_mixit_experiment
for lr in .00001;
do
python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_mixit_experiment/project_config.yaml --name=${MODEL}_${lr} --lr=${lr} --batch-size=8 --scale-factor=320 --clip-duration=10 --rho=1 --segmentation-based --encoder-type=$MODEL --n-epochs=0 --previous-checkpoint-fp=/home/jupyter/sound_event_detection/projects/${DATASET}_${MODEL}_experiment/${MODEL}_${lr}/final-model.pt --is-mixit --overwrite
done
done

for DATASET in OZF_slowed_0.5;
do
python main.py project-setup --train-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/train_info.csv --val-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/mixit_test_info.csv --test-info-fp=/home/jupyter/data/voxaboxen_data/${DATASET}/formatted/mixit_test_info.csv --project-dir=projects/${DATASET}_${MODEL}_mixit_experiment
for lr in .00001;
do
python main.py train-model --project-config-fp=projects/${DATASET}_${MODEL}_mixit_experiment/project_config.yaml --name=${MODEL}_${lr} --lr=${lr} --batch-size=8 --scale-factor=320 --clip-duration=10 --rho=1 --segmentation-based --encoder-type=$MODEL --n-epochs=0 --previous-checkpoint-fp=/home/jupyter/sound_event_detection/projects/${DATASET}_${MODEL}_experiment/${MODEL}_${lr}/final-model.pt --is-mixit --overwrite
done
done
