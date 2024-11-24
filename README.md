# Quick start
```bash
bash install_apt_pachages.sh
pip3 install -r requirements.txt

mkdir data

# Download IAM dataset from https://www.kaggle.com/datasets/ngkinwang/iam-dataset
ls data/IAM  # -> words/  words.txt

# Download HCS dataset from https://sites.google.com/view/chess-scoresheet-dataset/home/
ls data/HCS  # -> 001_0.png 001_1.png ...

# Download kanagawa_champ_2023
python3 script/extract_image_from_pdf.py
ls data/kanagawa_champ_2023/images/  # -> 1.jpeg, 2.jpeg, ...

# Train the model with IAM dataset
python3 src/train_iam.py

# Train the model with HCS dataset with pre-trained model of IAM dataset
# TODO: add procedure of traing with HCS dataset

for f in $(ls data/kanagawa_champ_2023/images/);do
    # NOTE: set --move_num appropriately to remove blank move
    python3 src/ncs_split.py \
        --input data/kanagawa_champ_2023/images/${f} \
        --output data/kanagawa_champ_2023/images_move/${f}/
done
ls data/kanagawa_champ_2023/images_move/1/  # 0001_0white.png, 0001_1black.png, ...

python3 src/predict_ncs_sheet.py
```
