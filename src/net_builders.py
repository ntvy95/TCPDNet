from utils import get_config

# This function is developed by mtanaka@sc.e.titech.ac.jp
def build_base_net_configs(default_config_path, setting=None):

    configs = get_config(default_config_path)

    keys = []
    for k in configs:
        if type(configs[k]) == dict:
            keys = keys + list(configs[k].keys())
    keys = set(keys)

    if( setting is not None ):
        for k, v in setting.items():
            if( k in configs ):
                if type(configs[k]) != dict:
                    configs[k] = setting[k]
                    continue
                for _k, _v in v.items():
                    if( _k in configs[k].keys() ):
                        configs[k][_k] = _v
                    else:
                        msg = f'( {k}, {_k} ) is unknow.'
                        raise KeyError(msg)
    return configs
