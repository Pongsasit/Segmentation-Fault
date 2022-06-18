from os import makedirs
import imageio
import json
from glob import glob
from osgeo import gdal, ogr, osr
from os.path import join, exists
from os import makedirs
from shapely.geometry import shape, MultiPolygon, LineString, Polygon, GeometryCollection


def extract_polygons(config, shape_file, gtif_orig, im_path, dataset):

    polygons = []
    s = shape_file.GetLayer(0)
    for i in range(s.GetFeatureCount()):
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

        ds = gdal.Translate( join(config.cropped_data_dir, '%s/%s/%s/%s/%s_%s.tif' % (dataset, year, time, first['id'], first['properties']['crop_type'], filename)) , gtif_orig, projWin=[shp.bounds[0], shp.bounds[3], shp.bounds[2], shp.bounds[1]])
        ds = None
            

def prepre_geo_data(config):
    

    train_labels_shp_path = join(config.data_dir, 'training_area', 'traindata.shp')
    test_labels_shp_path = join(config.data_dir, 'test_area', 'testdata.shp')

    train_labels = ogr.Open(train_labels_shp_path)
    test_labels = ogr.Open(test_labels_shp_path)

    
    im_paths = glob(join((config.data_dir), 'sentinel-2-image', '2021', '**', 'IMG_DATA', '*.jp2'))

    for im_path in im_paths:
        print(im_path)
        
        # im = imageio.imread(im_path)
        gtif_orig = gdal.Open(im_path)

        extract_polygons(config, train_labels, gtif_orig, im_path, 'train')
        extract_polygons(config, test_labels, gtif_orig, im_path, 'test')

        stop=1



def main():
    pass

if __name__ == '__main__':
    main()