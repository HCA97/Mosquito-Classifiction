# Mosquito Classification

This is the 7th place solution for the MosquitoAlert Challenge 2023. The goal of this competition is to identify mosquitoes and determine their species.

## How to Run CLIP Classifier

1. **Install Datasets**
   - Download the competition dataset from [here](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files) and unzip it to a folder named `data_round_2` (the annotations files are included).
   - Install [lux's dataset](https://discourse.aicrowd.com/t/external-dataset-notice-on-usage-declaration/8999/4?u=hca97), unzip `gbif-cropped` and `inaturalist-six-cropped` (the annotations files are included).

2. **Install Dependencies**
   - Use the following command to install the necessary dependencies: `pip install -r requirements.txt`.

3. **Run the Classifier**
   - Navigate to the `experiments` directory and execute the following command: `python mosquito_clf_yolo_lux_ema.py`.

## How to Train YOLOv8-s Model

1. **Install Competition Dataset**
   - Download the competition dataset from [here](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files) and unzip it to a folder named `data_round_2`.

2. **Install Dependencies**
   - Use the following command to install the necessary dependencies: `pip install -r requirements.txt`.

3. **Prepare YOLO Dataset**
   - Navigate to the `experiments/yolo` directory and run the script: `python convert_mosquito_to_yolo.py`.

4. **Start Training**
   - Execute the command: `python yolo_training.py`.

## Annotation Files

### data_round_2

- `phase2_train_v0_cleaned.csv` was created using `owl-vit`. You can refer to `experiments/cleaning_annotations.ipynb` for details.
- `phase2_train_v0_cleaned_yolo_best_annotations.csv` uses `phase2_train_v0_cleaned.csv` along with YOLOv8-s model annotations. Refer to `extra_data/annotate_images_yolo.py` for more information.
- `best_model_val_data_yolo_annotations.csv` and `best_model_train_data_yolo_annotations.csv` are train/validation splits of `phase2_train_v0_cleaned_yolo_best_annotations.csv`.

### gbif-cropped and inaturalist-six-cropped

- `inaturalist.csv` contains annotations for lux's dataset. Since the images are already cropped, we used the entire image as the bounding box.
