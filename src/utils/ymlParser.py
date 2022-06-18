import yaml

def _convert_dict_to_object(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, _convert_dict_to_object(j))
        elif isinstance(j, seqs):
            setattr(top, i, 
                type(j)(_convert_dict_to_object(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

def parse_yml(yml_path):

    with open(yml_path, 'r') as ymlfile:
        try:
            cfg = yaml.safe_load(ymlfile)
            cfg = _convert_dict_to_object(cfg)
            print('yml file parsed succefully')            
            return cfg

        except yaml.YAMLError as exc:
            print(exc)
