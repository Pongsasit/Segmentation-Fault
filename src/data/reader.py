from os import makedirs
import imageio
import json
import numpy as np
import subprocess
from glob import glob
from osgeo import gdal, ogr, osr
from os.path import join, exists
from os import makedirs
from shapely.geometry import shape
from tqdm import tqdm


def extract_polygons(config, shape_file, gtif_orig, im_path, dataset):

    s = shape_file.GetLayer(0)
    for i in tqdm(range(s.GetFeatureCount())):
        feature = s.GetFeature(i)
        first = feature.ExportToJson()
        first = json.loads(first)
        shp = shape(first['geometry'])
        shp = shp.buffer(0) if not shp.is_valid else shp

        path_names = im_path.split('.')[:-1][0].split('/')[-4:]
        del path_names[2]

        year, time, filename = path_names
        
        export_dir = join(config.cropped_data_dir, '%s/%s/%s/%s' % (dataset, year, time, first['id']))
        
        if not exists(export_dir):
            makedirs(export_dir)

        if dataset == 'test':
            first['properties']['crop_type'] = 'X'

        print(shp.bounds, gtif_orig.GetGeoTransform())

        ds = gdal.Translate( join(config.cropped_data_dir, '%s/%s/%s/%s/%s_%s.tif' % (dataset, year, time, first['id'], first['properties']['crop_type'], filename)) , gtif_orig, projWin=[shp.bounds[0], shp.bounds[3], shp.bounds[2], shp.bounds[1]])
        ds = None
            

def prepre_geo_data(config):
    

    train_labels_shp_path = join(config.data_dir, 'training_area', 'traindata.shp')
    test_labels_shp_path = join(config.data_dir, 'testing_area', 'testdata.shp')

    train_labels = ogr.Open(train_labels_shp_path)
    test_labels = ogr.Open(test_labels_shp_path)

    
    im_paths = glob(join((config.data_dir), 'sentinel-2-image', '2020', '**', 'IMG_DATA', '*.jp2'))

    # reference_resolution = np.array([imageio.imread(p).shape[:2] for p in im_paths if 'TCI' in p]).max(0) # this line gets the biggest resolution from reference image, but it takes some time, so we know the biggest is 2051 so thats why its commented and hardcoded
    reference_resolution = [2051, 2051]

    for im_path in im_paths:
        print(im_path)
        
        # im = imageio.imread(im_path)

        gtif_orig = gdal.Open(im_path)

        # if the resolution is smaller, rescale to bigger (in order to correctly cut the area)
        # if gtif_orig.RasterXSize < reference_resolution[0] or gtif_orig.RasterYSize < reference_resolution[1]:

        # resize
        test = subprocess.Popen(["gdal_translate", str(im_path),  "-outsize", str(reference_resolution[0]), str(reference_resolution[1]) ,"tmp.tif"], stdout=subprocess.PIPE)
        _ = test.communicate()[0]

        # cut to shape
        test = subprocess.Popen(["gdalwarp", "-crop_to_cutline", "-cutline", train_labels_shp_path, 'tmp.tif', 'tmp_train.tif'], stdout=subprocess.PIPE)
        _ = test.communicate()[0]
        test = subprocess.Popen(["gdalwarp", "-crop_to_cutline", "-cutline", test_labels_shp_path, 'tmp.tif', 'tmp_test.tif'], stdout=subprocess.PIPE)
        _ = test.communicate()[0]

        gtif_orig_train = gdal.Open('tmp_train.tif')
        gtif_orig_test = gdal.Open('tmp_test.tif')

        extract_polygons(config, train_labels, gtif_orig_train, im_path, 'train')
        extract_polygons(config, test_labels, gtif_orig_test, im_path, 'test')



def main():
    pass

if __name__ == '__main__':
    main()