import json
import os

from scripts.utils import get_abs_filepath, RF_PARAM_5G


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RF_PARAM_5G):
            return obj.value
        return json.JSONEncoder.default(self, obj)


def create_unique_filename(title: str) -> str:
    suffix = 0
    while True:
        try:
            dir = get_abs_filepath(f"data/results/experiments/{title}")
            if suffix > 0:
                dir += f"_{suffix}"
            os.mkdir(dir)
            break
        except FileExistsError:
            suffix += 1

    return dir


def save_experiment_result(title: str, config: dict, results: dict):

    dir = create_unique_filename(title)

    # store config in json
    config_file = os.path.join(dir, "config.json")
    with open(config_file, "w") as json_file:
        json.dump(config, json_file, indent=4, cls=EnumEncoder)

    # store dataframes
    for name, dataframe in results.items():
        dataframe.to_csv(os.path.join(dir, name + ".csv"))

    print(f"Experiment stored in {dir}")
