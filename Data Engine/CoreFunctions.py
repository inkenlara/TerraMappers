import ee
import requests
import ee
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch

ee.Authenticate()
ee.Initialize(project='ee-kesava89-tumai')
print(ee.String('Hello from the Earth Engine servers!').getInfo())

url = "meta.csv"
sampleGenPath = "/home/kesava89/tumai/sandbox/BraDD-S1TS/gensamples/"

import os.path
if not os.path.isfile(url):
    f = open(url, "w+")
    f.write(",alert_idx,center_idx,date,sampling_type,state,file,close_set\n")
    f.close()
            

lines=[]
with open(url, "r") as f:
    lines=f.readlines()

lines = lines[1:]

metaContent = pd.read_csv(url, names=["","alert_idx","center_idx","date","sampling_type","state","file","close_set"])

import math
def FindOffsetLatLongFromDistance(lat, lon, distanceLat, distanceLon):

    #Earth’s radius, sphere
    R=6378137

    #offsets in meters
    dn = distanceLat
    de = distanceLon

    #Coordinate offsets in radians
    dLat = dn/R
    dLon = de/(R*math.cos(math.pi*lat/180))

    #OffsetPosition, decimal degrees
    latO = lat + dLat * 180/math.pi
    lonO = lon + dLon * 180/math.pi 
    return latO, lonO


# A small region within the image.
top, left = (-44.84026673520523,-3.3934410412280656)
bottom, right = FindOffsetLatLongFromDistance(top, left, 480, 480)
pngArea =    ee.Geometry.Polygon(
        [[[10.966712295983333, 44.65340259358304],
          [10.966712295983333, 44.24754759759594],
          [11.738501846764583, 44.24754759759594],
          [11.738501846764583, 44.65340259358304]]])
print(bottom, right)
region = ee.Geometry.BBox(top, left, bottom, right)

def GetS2HarmonizedFromRegion(region, filterDateStart, filterDateEnd):
  # A Sentinel-2 surface reflectance image.
  dataset = (
      ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterDate(filterDateStart, filterDateEnd)
      # Pre-filter to get less cloudy granules.
  )
  img = dataset.mosaic().reproject(crs = ee.Projection('EPSG:4326'), scale=16)

  # Get 2-d pixel array for AOI - returns feature with 2-D pixel array as property per band.
  band_arrs = img.sampleRectangle(region=region)

  # Get individual band arrays.
  band_arr_b4 = band_arrs.get('B2')
  band_arr_b5 = band_arrs.get('B3')
  band_arr_b6 = band_arrs.get('B4')

  # Transfer the arrays from server to client and cast as np array.

  np_arr_b4 = band_arr_b4.getInfo()
  np_arr_b5 = band_arr_b5.getInfo()
  np_arr_b6 = band_arr_b6.getInfo()


  # Expand the dimensions of the images so they can be concatenated into 3-D.
  np_arr_b4 = np.expand_dims(np_arr_b4, 2)
  np_arr_b5 = np.expand_dims(np_arr_b5, 2)
  np_arr_b6 = np.expand_dims(np_arr_b6, 2)


  # # Stack the individual bands to make a 3-D array.
  rgb_img = np.concatenate((np_arr_b6, np_arr_b5, np_arr_b4), 2)

  # # Scale the data to [0, 255] to show as an RGB image.
  rgb_img_test = (255*((rgb_img)/3500)).astype('uint8')
  plt.imshow(rgb_img_test)
  plt.show()

  def mask_edge(image):
    edge = image.lt(-30.0)
    masked_image = image.mask().And(edge.Not())
    return image.updateMask(masked_image)

pol = ['VV', 'VH']

def GetSARImageFromRegion(region, filterDateStart, filterDateEnd):
  # A Sentinel-2 surface reflectance image.
  dataset = (
      ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterDate(filterDateStart, filterDateEnd).filter(ee.Filter.eq('transmitterReceiverPolarisation', pol))
      .filterMetadata('instrumentMode', 'equals', 'IW')
      .map(mask_edge)
      # Pre-filter to get less cloudy granules.
  )
  img = dataset.mosaic().reproject(crs = ee.Projection('EPSG:4326'), scale=16)

  # Get 2-d pixel array for AOI - returns feature with 2-D pixel array as property per band.
  band_arrs = img.sampleRectangle(region=region)

  # Get individual band arrays.
  band_arr_b4 = band_arrs.get('VV')
  band_arr_b5 = band_arrs.get('VH')

  # Transfer the arrays from server to client and cast as np array.

  np_arr_b4 = band_arr_b4.getInfo()
  np_arr_b5 = band_arr_b5.getInfo()
  return (np_arr_b4, np_arr_b5)

def GenerateSARForDefaultTimeSampling(region):
    dateStarts = [ "2020-07-01", "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01", "2020-11-01", "2020-12-01", "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01"]
    dateEnds = [ "2020-07-28", "2020-07-28", "2020-08-28", "2020-09-28", "2020-10-28", "2020-11-28", "2020-12-28", "2021-01-28", "2021-02-28", "2021-03-28", "2021-04-28", "2021-05-28", "2021-06-28", "2021-07-28", "2021-08-28", "2021-09-28", "2021-10-28", "2021-11-28"]
    payload = []
    import datetime
    for i in range(0, len(dateStarts)):
        sar = GetSARImageFromRegion(region, dateStarts[i], dateEnds[i])
        sar_pad = np.pad(sar, ((0, 0), (0, 48 - np.shape(sar)[1]), (0,48 - np.shape(sar)[2])), mode='wrap')
        dateStarts_date_time = datetime.datetime.strptime(dateStarts[i], '%Y-%m-%d')
        item = { "label_dates": dateStarts_date_time.date(), "sar": sar_pad}
        payload.append(item)
    sarPayload = []
    dateStartPayload = []
    for item in payload:
        sarPayload.append(item['sar'])
        dateStartPayload.append(item['label_dates'])
    pt = {"image_dates": ((dateStartPayload)), "image": torch.from_numpy(np.array(sarPayload))}
    filename="0000000_"+str(dateStarts[-1])+".pt"
    metaLine = "0,0,0,{0},positive,PA,{1},test\n".format(dateStarts[-1], filename)
    lines.append(metaLine)
    with open(url, 'a') as file:
        file.write(metaLine)
    torch.save(pt,sampleGenPath+filename)