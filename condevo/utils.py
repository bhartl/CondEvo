import os
import json
import yaml


def to_json(obj):
    """ convert object to json """
    try:
        return json.dumps(obj)
    except TypeError:
        if not isinstance(obj, dict):
            obj = obj.to_dict() if hasattr(obj, 'to_dict') else obj.__dict__
        return json.dumps({k: repr(v) for k, v in obj.items()})


def load_file(file):
    """ load dict from file or str via json or yaml """

    if isinstance(file, dict):
        return file

    if os.path.exists(file):
        if file.endswith(".yml"):
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
