import os
import random
import string


def config_uglify(experiment):
    experiment_alias = os.path.relpath(experiment, "experiments")
    experiment_alias = os.path.splitext(experiment_alias)[0]
    experiment_alias = "dps_exp_" + experiment_alias.replace("/", "_")
    random_string = "".join(random.choice(string.hexdigits) for _ in range(8))
    experiment_alias += "_" + random_string
    experiment_alias = experiment_alias.lower()
    return experiment_alias
