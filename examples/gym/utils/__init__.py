from . import configs
from condevo.utils import load_file, to_json


def load_experiment(config):
    """ load experiment configuration file """
    if isinstance(config, str):
        try:
            return load_file(config)
        except FileNotFoundError:
            pass

    return config


def load_nn(nn_cls, config, num_params, num_conditions=0):
    """ load `hades.nn.<nn_cls>` neural network instance  with `config` specifying the
        corresponding constructor `kwargs`, and `num_params` specifying the input size.

    :param nn_cls: str or class, neural network class.
    :param config: dict or str, configuration of the neural network, can also be a file path or json-string.
    :param num_params: int, input size of the neural network (extra to config, as this might depend on the
                       objective function)
    :param num_conditions: int, number of conditions for the neural network
    :return: tuple of (nn_instance, config-dict)
    """
    default_config = getattr(configs, nn_cls if isinstance(nn_cls, str) else nn_cls.__name__)
    if config is None:
        config = {}

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "nn_config should be a dictionary"
    # merge passed config with default config
    config = {**default_config, **config}

    if "num_params" not in config:
        config["num_params"] = num_params + num_conditions

    # load nn
    if isinstance(nn_cls, str):
        from condevo import nn
        nn_cls = getattr(nn, nn_cls)

    nn_instance = nn_cls(**config)
    return nn_instance, config


def load_diffuser(diff_cls, config, nn_instance):
    """ load `hades.diffusion.<diff_cls>` diffusion instance with `config` specifying the
        corresponding constructor `kwargs`, and `nn_instance` specifying the neural network instance.

    :param diff_cls: str or class, diffusion class
    :param config: dict or str, configuration of the diffusion model, can also be a file path or json-string.
    :param nn_instance: hades.nn.<nn_cls>, neural network instance
    :return: tuple of (diff_instance, config-dict)
    """
    default_config = getattr(configs, diff_cls if isinstance(diff_cls, str) else diff_cls.__name__)
    if config is None:
        config = {}

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "diff_config should be a dictionary"
    # merge passed config with default config
    config = {**default_config, **config}

    # load diffuser
    if isinstance(diff_cls, str):
        from condevo import diffusion
        diff_cls = getattr(diffusion, diff_cls)

    diff_instance = diff_cls(nn=nn_instance, **config)
    return diff_instance, config


def load_es(es_cls, config, diffuser, num_params):
    """ load `hades.es.<es_cls>` evolutionary strategy instance with `config` specifying the
        corresponding constructor `kwargs`, and `diffuser` specifying the diffusion model instance.

    :param es_cls: str or class, `hades` evolutionary strategy class
    :param config: dict or str, configuration of the evolutionary strategy, can also be a file path or json-string.
    :param diffuser: (Optional) hades.diffusion.<diff_cls>, diffusion model instance. If provided, the model
                     will be passed to the evolutionary strategy constructor.
    :param num_params: int, number of parameters of the evolutionary search, which should be equal to input size of
                       the neural network if provided (extra to config, as this might depend on the objective function).
    :return: tuple of (es_instance, config-dict)
    """

    default_config = getattr(configs, es_cls if isinstance(es_cls, str) else es_cls.__name__)
    if config is None:
        config = {}

    elif isinstance(config, str):
        config = load_file(config)

    assert isinstance(config, dict), "es_config should be a dictionary"
    # merge passed config with default config
    config = {**default_config, **config}

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
