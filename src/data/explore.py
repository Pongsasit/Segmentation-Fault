import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import imageio
import argparse
from osgeo import gdal, ogr, osr
from os.path import join

from utils.ymlParser import parse_yml

def main(args=None):

    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('-c', '--config_path', help='Path to config yml file', type=str)    
    parser = parser.parse_args(args)

    config = parse_yml(parser.config_path)

    
    
    im_topo = imageio.imread(join(config.data_dir, 'croptype', 'index_stack.tiff'))[..., :3]

    gtif_orig = gdal.Open(join(config.data_dir, 'croptype', 'index_stack.tiff'))

    stop=1



if __name__ == '__main__':
    main()