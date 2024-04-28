# TerraMappers
## Inspiration

Deforestation and other geological and sensing diagnosis have been a long-running detection problem, of which every smallest improvement provides a significant impact on the longevity of the wildlife, vegetation and the carbon footprint across the world. 

Several GIS systems are capable of high resolution imagery with numerous bands, beyond the visible light spectrum that are capable of signifying forest cover apart from rest of the land masses. However, due to several factors, but not limited to the position of the landmass in earth, type of vegetation and sparse distribution makes it incredibly difficult for plain analytical models to derive to a logical insight. With advances in machine computing and algorithms in machine learning, it is now a little easier to help analytical models and algorithms bridge the gap to reasonable insights and predictions.

## What it does

Our product proposal is multi-faceted: while primarily it circles around functionality that determines the extent of forest coverage and more particularly the rate of change of forest coverage, we also were determined to create a system that can be made accessible to everyone, end-users and businesses alike. We want the product to hone a central hub of information from various reliable sources to determine forest and vegetation coverage, with a backend extensible to several other features.

## How we built it

Instead of a static data model archived regularly, our product relies on online resources such as Google Earth Engine and Copernicus Datasets that are capable of delivering multi-spectral bands, with state-of-the-art corrections and improvements on image quality. The data engine as well call it, are capable of seamlessly deriving data from several data models for a requested geo-point, and also calculates necessary spherical corrections for accurate tiling measurements.

With the data model, we host three major functionalities that complement each other.

Initially, a segmentation model, inspired from BraDD-S1TS was trained using their state-of-the art dataset of timelapse series, spun across the Amazons. The model reached expected metrics of IoU and Recall, was modified to accommodate several other improvements such as API calls for custom inputs, which the original repository do not provide. The output of the segmentation model delivers a clean cut binary masking information about current forest coverage and also for the future requested time samples, based on it's older image data.

This masking information is further run into a Convolutional Neural Network with other known time-samples where mask information is present, to determine a rate of change of forest cover within the stipulated area. Instead of running RNN models, it made a lot of sense to involve convolutional models because the nature of data can be volatile and convolutional filters are known to provide stability under dire environments.

We provide furthermore a system to also detect changes in the temperature trend of a wider location range due to rate of change of forest cover. This model uses the outputs from everything: the machine learning models discussed and also the data samples from the datasets.

## Challenges we ran into

Acquiring map data: the data layers required to acquire were quite cumbersome as the "tiled" images we acquired are through different satellites and the tiles are sometimes inconsistent. A significant amount of time was used to align all tiles irrespective of the data band we choose from any satellite, across both Google Earth Engine components and Copernicus.

Understanding GIS data assimilation with RADD and PRODES Data: There were papers that had already implementation and integration of these datasets, but it took a large effort to understand and correlate the data points from alarms provided by RADD and PRODES, but also make sure it is consistent with the pre-processing requirements of BraDD-S1TS model.

Integrating core-components in real-time: This is an unsolved challenge - the data while being up-to-date, are constantly built by the map engines and takes a long time until a bitmap or an array information is generated. We had to create several pipelines to reduce download time, and also take care of quota and rate-limiting.

## Accomplishments that we're proud of
1. That we finished we planned
2. Came together as group to bash at one git repository that took us 7 hours to fix what they did wrong
3. Worked an all-nighter to figure out how each outputs work and what output works significantly at which use case

## What we learned
1. A lot of GIS data manipulation methods
2. That earth isn't flat and 10m x 10m tile in a plane earth can cause approximation issues after 100 tiles if you don't take in spherical distortion correction and grand line into account
3. Learned to work and support each other at their problems and especially in Machine Learning models

## What's next for TerraMappers
1. Keep figuring out what's really wrong with existing GIS based repositories and why none really work now
2. Continue working on a much faster and powerful data acquisition for online map data
3. Integrate all ML models and create a fully functional web-app that has support for KML/KMZ models
