import json
import os


def create_unique_filename(title: str) -> str:
    suffix = 0
    while True:
        try:
            dir = f"./data/results/experiments/{title}"
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
        json.dump(config, json_file, indent=4)

    # store dataframes
    for name, dataframe in results.items():
        dataframe.to_csv(os.path.join(dir, name + ".csv"))

    print(f"Experiment stored in {dir}")
