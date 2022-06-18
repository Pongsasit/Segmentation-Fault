
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from utils.ymlParser import parse_yml
from data.reader import prepre_geo_data

def main(args=None):

    # you can change the logic if you have better idea, i've just put a not so clear sctructure so we dont make too big mess in the code and we can track the experiments
    # lets write the code in a way, where everything is configurable from the config file, so we can set up some differet runs for night or basically not loose the control over the mess, thanks :D

    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('-c', '--config_path', help='Path to config yml file', type=str)    
    parser = parser.parse_args(args)

    config = parse_yml(parser.config_path)

    prepre_geo_data(config)



    stop=1



if __name__ == '__main__':
    main()