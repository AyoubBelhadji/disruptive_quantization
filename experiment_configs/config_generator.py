#!/usr/bin/env python
import json
import argparse
from collections.abc import MutableMapping
import os
import itertools

# From https://stackoverflow.com/a/6027615
def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def to_nested_dict(dictionary: dict, separator: str):
    nested_dict = {}
    for key, value in dictionary.items():
        keys = key.split(separator)
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested_dict

def cartesian_product_kwargs(**kwargs):
    """ Assume each value is an iterable. Create a cartesian product of the values. """
    keys = kwargs.keys()
    values = kwargs.values()
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]

def find_kwarg_combo_keys(flattened_template: dict, separator: str, **kwargs):
    """ For each kwarg key, get all applicable keys in the flattened template. """
    keys = kwargs.keys()
    kwarg_combo_keys = {}
    for key in keys:
        kwarg_combo_keys[key] = [k for k in flattened_template.keys() if k.endswith(separator + key)]
    return kwarg_combo_keys

def kwarg_combo_str(kwarg_combo: dict):
    """ Create a unique string identifier for a kwarg combo. """
    return "_".join([f"{key}{value}" for key, value in kwarg_combo.items()])

def create_new_config(flattened_template: dict, kwarg_combo: dict, kwarg_combo_keys: dict):
    """ Create a new config based on the template and a kwarg combo. """
    new_config = flattened_template.copy()
    for key, value in kwarg_combo.items():
        for k in kwarg_combo_keys[key]:
            new_config[k] = value
    return new_config

def generate_configs(dest_dir: str, template_file: str, separator: str, **kwargs):
    """
    Generate a set of configuration files based on a template file and a set of keyword arguments.

    Expects each kwarg to have iterable values. Will create a matrix of configurations based on
    the cartesian product of the values of the keyword arguments.
    """
    # Create the destination directory if not existing
    os.makedirs(dest_dir, exist_ok=True)

    # Load template file
    with open(template_file, "r") as f:
        template = json.load(f)
    
    # Flatten the template file
    flattened_template = flatten_dict(template, separator=separator)
    all_kwarg_combos = cartesian_product_kwargs(**kwargs)
    kwarg_combo_keys = find_kwarg_combo_keys(flattened_template, separator, **kwargs)
    print(f"Generating {len(all_kwarg_combos)} configurations...")
    for kwarg_combo in all_kwarg_combos:
        # Create a unique identifier for the configuration
        combo_str = kwarg_combo_str(kwarg_combo)
        # Create a new config based on the template
        new_config = create_new_config(flattened_template, kwarg_combo, kwarg_combo_keys)
        # Unflatten the new config
        nested_new_config = to_nested_dict(new_config, separator)
        # Write the new config to a file
        with open(f"{dest_dir}/config_{combo_str}.json", "w") as f:
            json.dump(nested_new_config, f, indent=4)
    print(f"Generated configurations in {dest_dir}.")

# Arg parser: Required args are template file, destination directory, and kwargs
# Optional args are separator
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a set of configuration files based on a template file and a set of keyword arguments.")
    parser.add_argument("template_file", type=str, help="Path to the template file.")
    parser.add_argument("dest_dir", type=str, help="Path to the destination directory.")
    parser.add_argument("kwargs", nargs="+", help="Keyword arguments to generate configurations.")
    parser.add_argument("--separator", type=str, default="__", help="Separator for nested keys.")
    return parser.parse_args()

def parse_kwarg_val(val_str):
    # Check if val_str is an int, float, or str
    try:
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return val_str

def main():
    args = parse_args()
    kwargs = {}
    for kwarg in args.kwargs:
        key, values = kwarg.split("=")
        kwargs[key] = list(map(parse_kwarg_val, values.split(",")))
    generate_configs(args.dest_dir, args.template_file, args.separator, **kwargs)

def example():
    example_json = "msip_sandbox/msip.json"
    dest_dir = "config_sandbox"
    separator = "__"
    generate_configs(dest_dir, example_json, separator, K=[10, 20], beta_ns=[0, 1])

if __name__ == "__main__":
    main()