import json
import numpy as np
def transform_to_dict(obj):
    obj_dict = {}
    for attr_name in [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]:
        if attr_name in ["export_dict", "parent", "rotor"]:
            continue
        attr_val = getattr(obj, attr_name)
        obj_dict[attr_name] = attr_val
        obj_dict[attr_name] = check_class_in_dict(attr_val)

        if type(attr_val) is dict:
            if "<" in str(attr_val):
                for key in attr_val:
                    obj_dict[attr_name][key] = check_class_in_dict(attr_val[key])
                    if type(obj_dict[attr_name][key]) is dict:
                        for sub_key in obj_dict[attr_name][key]:
                            obj_dict[attr_name][key][sub_key] = check_class_in_dict(attr_val[key][sub_key])

    return obj_dict


def check_class_in_dict(attr_val):
    if str(attr_val)[0] == "<":
        return transform_to_dict(attr_val)
    if type(attr_val) is list:
        if "<" in str(attr_val):
            return_dict = {}
            for i, element in enumerate(attr_val):
                return_dict[str(i)] = transform_to_dict(element)
            return return_dict
        else:
            return attr_val
    else:
        return attr_val

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
