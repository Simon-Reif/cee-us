import copy
import collections.abc

#similar to: https://stackoverflow.com/a/32357112
def _deep_update_rec(d, u):
    if not isinstance(d, collections.abc.Mapping):
        d = copy.deepcopy(u)
        return d
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _deep_update_rec(d.get(k, {}), v)
        # elif isinstance(v, list):
        #     d[k] = (d.get(k, []) + v)
        else:
            d[k] = v
    return d

# updated nested dictionary with nested dictionary
def deep_update(d, u, in_place=False):
    if in_place:
        _deep_update_rec(d, u)
    else:
        d = copy.deepcopy(d)
        return _deep_update_rec(d, u)

def _to_wandb_sweep_rec(d):
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            val = _to_wandb_sweep_rec(v)
            d[k] = {"parameters": val}
        else:
            val = {"value": v}
            d[k] = val
    return d

def to_wandb_sweep_config_format(config):
    new_config = copy.deepcopy(config)
    new_config = _to_wandb_sweep_rec(new_config)
    return new_config

def _fix_wandb_sweep_rec(d):
    if "value" in d and "values" in d:
        d.pop("value")
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            _fix_wandb_sweep_rec(v)

# TODO: fix problem of "values" being saved in addition to "value"
def fix_wandb_sweep_config(config):
    new_config = copy.deepcopy(config)
    _fix_wandb_sweep_rec(new_config)
    return new_config


# sweep_base defines sweep specific control parameters sweep name, method, etc
# params_base are the fixed parameters of the program to be overwritten with hyper_param_values              
# hyper_param_values are the hyperparameters to be swept over
#               needs to already be in format of sweep "parameters"
def make_sweep_config(sweep_config_base, params_base, hyper_param_values=None):
    if hyper_param_values is not None:
        hyper_param_sweep = hyper_param_values
    else:
        hyper_param_sweep = copy.deepcopy(sweep_config_base["parameters"])
    params_base = to_wandb_sweep_config_format(params_base)
    params = deep_update(params_base, hyper_param_sweep, in_place=False)
    params = fix_wandb_sweep_config(params)
    sweep_config = copy.deepcopy(sweep_config_base)
    #TODO: maybe later update parameters rather than overwriting
    sweep_config["parameters"]=params
    return sweep_config

# get attributes in AttributeDict by string in dot notation
def val_from_dot_string(d, keys_with_dots):
    for key in keys_with_dots.split('.'):
        d = d[key]
    return d