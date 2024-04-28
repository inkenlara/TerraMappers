# DEFORESTATION DETECTION IN THE AMAZON WITH SENTINEL-1 SAR IMAGE TIME SERIES

<p align="center" width="100%">
    <img src="./pictures/DatasetFigure.svg"> 
</p>

This is the official implementation of the paper:  
- [Deforestation Detection in the Amazon with Sentinel-1 SAR Image Time Series (ISPRS 23)](https://isprs-annals.copernicus.org/articles/X-1-W1-2023/835/2023/)

## Abstract

Deforestation has a significant impact on the environment, accelerating global warming and causing irreversible damage to ecosystems. Large-scale deforestation monitoring techniques still mostly rely on statistical approaches and traditional machine learning models applied to multi-spectral, optical satellite imagery and meta-data like land cover maps. However, clouds often obstruct observations of land in optical satellite imagery, especially in the tropics, which limits their effectiveness. Moreover, statistical approaches and traditional machine learning methods may not capture the wide range of underlying distributions in deforestation data due to limited model capacity. To overcome these drawbacks, we apply an attention-based neural network architecture that learns to detect deforestation end-to-end from time series of synthetic aperture radar (SAR) images. Sentinel-1 C-Band SAR data are mostly independent of the weather conditions and our trained neural network model generalizes across a wide range of deforestation patterns of Amazon forests. We curate a new dataset, called BraDD-S1TS, comprising approximately 25,000 image sequences for deforested and unchanged land throughout the Brazilian Amazon. We experimentally evaluate our method on this dataset and compare it to state-of-the-art approaches. We find it outperforms still-in-use methods by 13.7 percentage points in intersection over union (IoU). We make BraDD-S1TS publicly available along with this publication to serve as a novel testbed for comparing different deforestation detection methods in future studies. 

# Setup
## Dataset (BraDD-S1TS)

<p align="center" width="100%">
    <img src="./pictures/CenterLocationMap.svg"> 
</p>

Please [download](https://zenodo.org/record/8060250/files/BraDD-S1TS_zenodo.zip?download=1) the files from [Zenodo Link](https://zenodo.org/record/8060250). 
Then, unzip them for a certain directory which should be also pointed out in the main code.

## Modeling

We have utilized the model architectures from this [repository](https://github.com/VSainteuf/utae-paps).
Please check it for further information.

# Experiments

Please check the file, named `run_via_parser.py`, for the experiments. 
You can also run such python file by
```
conda activate your_env_name

python run_via_parser.py \
--Storing_wandbProject your_project_name \
--Storing_wandbEntity your_username \
--Storing_savingPath /path/for/experiments/ \
--DataModule_Dataset_path /path/for/dataset/
```

# Citation
```
@Article{isprs-annals-X-1-W1-2023-835-2023,
AUTHOR = {Karaman, K. and Sainte Fare Garnot, V. and Wegner, J. D.},
TITLE = {DEFORESTATION DETECTION IN THE AMAZON WITH SENTINEL-1 SAR IMAGE TIME SERIES},
JOURNAL = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {X-1/W1-2023},
YEAR = {2023},
PAGES = {835--842},
URL = {https://isprs-annals.copernicus.org/articles/X-1-W1-2023/835/2023/},
DOI = {10.5194/isprs-annals-X-1-W1-2023-835-2023}
}
```
