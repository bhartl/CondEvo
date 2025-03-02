DST = "data/es_benchmark/optimize"
H5_FILE = "ES_{ES}-objective_{objective}.h5"
PNG_FILE = "ES_{ES}-objective_{objective}-run_{run}-generation_{generation}.png"
GIF_FILE = "ES_{ES}-objective_{objective}-run_{run}.gif"


def to_json(obj):
    """ convert object to json """
    import json
    try:
        return json.dumps(obj)
    except TypeError:
        if not isinstance(obj, dict):
            obj = obj.to_dict() if hasattr(obj, 'to_dict') else obj.__dict__
        return json.dumps({k: repr(v) for k, v in obj.items()})


def load_file(file):
    """ load file as json or yaml """
    import os

    if os.path.exists(file):
        if file.endswith(".yml"):
            import yaml
            with open(file, 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        elif file.endswith(".json"):
            import json
            with open(file, 'r') as f:
                return json.load(f)

    else:
        # inspect whether file is a json-string
        import json
        try:
            return json.loads(file)
        except json.JSONDecodeError:
            pass

    raise FileNotFoundError(f"File `{file}` not found, or not supported")


def load_nn(nn_cls, config, num_params):
    """ load `condevo.nn.<nn_cls>` neural network instance  with `config` specifying the
        corresponding constructor `kwargs`, and `num_params` specifying the input size.

    :param nn_cls: str or class, neural network class.
    :param config: dict or str, configuration of the neural network, can also be a file path or json-string.
    :param num_params: int, input size of the neural network (extra to config, as this might depend on the
                       objective function)
    :return: tuple of (nn_instance, config-dict)
    """
    if config is None:
        import configs
        config = getattr(configs, nn_cls if isinstance(nn_cls, str) else nn_cls.__name__)

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "nn_config should be a dictionary"

    if "num_params" not in config:
        config["num_params"] = num_params

    # load nn
    if isinstance(nn_cls, str):
        from condevo import nn
        nn_cls = getattr(nn, nn_cls)

    nn_instance = nn_cls(**config)
    return nn_instance, config


def load_diffuser(diff_cls, config, nn_instance):
    """ load `condevo.diffusion.<diff_cls>` diffusion instance with `config` specifying the
        corresponding constructor `kwargs`, and `nn_instance` specifying the neural network instance.

    :param diff_cls: str or class, diffusion class
    :param config: dict or str, configuration of the diffusion model, can also be a file path or json-string.
    :param nn_instance: condevo.nn.<nn_cls>, neural network instance
    :return: tuple of (diff_instance, config-dict)
    """
    if config is None:
        import configs
        config = getattr(configs, diff_cls if isinstance(diff_cls, str) else diff_cls.__name__)

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "diff_config should be a dictionary"

    # load diffuser
    if isinstance(diff_cls, str):
        from condevo import diffusion
        diff_cls = getattr(diffusion, diff_cls)

    diff_instance = diff_cls(nn=nn_instance, **config)
    return diff_instance, config


def load_es(es_cls, config, diffuser, num_params):
    """ load `condevo.es.<es_cls>` evolutionary strategy instance with `config` specifying the
        corresponding constructor `kwargs`, and `diffuser` specifying the diffusion model instance.

    :param es_cls: str or class, `condevo` evolutionary strategy class
    :param config: dict or str, configuration of the evolutionary strategy, can also be a file path or json-string.
    :param diffuser: (Optional) condevo.diffusion.<diff_cls>, diffusion model instance. If provided, the model
                     will be passed to the evolutionary strategy constructor.
    :param num_params: int, number of parameters of the evolutionary search, which should be equal to input size of
                       the neural network if provided (extra to config, as this might depend on the objective function).
    :return: tuple of (es_instance, config-dict)
    """

    if config is None:
        import configs
        config = getattr(configs, es_cls if isinstance(es_cls, str) else es_cls.__name__)

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "es_config should be a dictionary"

    # load es
    if isinstance(es_cls, str):
        from condevo import es
        es_cls = getattr(es, es_cls)

    # inspect `es_cls` constructor whether "model" is a parameter
    try:
        es_instance = es_cls(num_params=num_params, model=diffuser, **config)
    except TypeError:
        es_instance = es_cls(num_params=num_params, **config)

    return es_instance, config
