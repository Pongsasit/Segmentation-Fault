# Segmentation-Fault
for hackathon purpose

## Prepare Data

#### Prerequisites
In order to smoothly reproduce the experiments, it is recommended to use Anaconda.
You can find installation instruction here:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Enviroment file is provided. Please use it to recreate the enviroment using following command:
``` 
conda env create -f req.yml
```
And activate the enviroment:
``` 
conda activate hackathon
```
You will also need to install appropriate version of gdal package.

For Ubuntu users, please follow the instruction:
https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html



## Prepare Data
1. In order to prepare the data for the training, run:
    ```bash
    python src/data/crop_data.py -c src/configs/config.yml
    ```
    In the config file, you have to specify:
    - data_dir: path to directory shared by Varuna with the data
    - cropped_data_dir: target path, where the preprocessed data will be saved
  
    Please follow the provided example.

    If you would like to skip this step (as it is quite time cosuming and requires correctly installed gdal package), please download already prepared data from here:
    https://drive.google.com/file/d/1nQlKEAItAxCWUWoY-Y90lC6XHHNFwf3D/view?usp=sharing


## Split Data

In order to split the data, run:

```bash
python src/data/split_data.py -c src/configs/config.yml
```

  Please find the description of required parameters in provided config file exmaple from line above.

## Train

You can configure training parameters from the config file. To recreate out best results model, please use the parameters provided in the example.

Run the training using following command:

```bash
python src/training/training_pipeline.py -c src/configs/config.yml
```

## Evaluate
```bash
python src/evaluation/evaluate.py -c src/configs/evaluate_config.yml
```

## Predict
```bash
python3 src/evaluate/predict.py -c src/configs/predict_config.yml
```