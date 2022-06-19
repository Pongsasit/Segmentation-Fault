# Segmentation-Fault
for hackathon purpose

## Prepare Data
```bash
python3 src/train/train.py -c src/configs/config.yml
```

## Split Data
```bash
python3 src/data/split_data.py -c src/configs/data_split_config.yml
```

## Train
```bash
python3 src/train/training_pipeline.py -c src/configs/train_config.yml
```

## Evaluate
```bash
python3 src/evaluation/evaluate.py -c src/configs/evaluate_config.yml
```

## Predict
```bash
python3 src/evaluate/predict.py -c src/configs/predict_config.yml
```