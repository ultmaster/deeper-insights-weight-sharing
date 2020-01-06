import multiprocessing
import os
import subprocess
from argparse import ArgumentParser

from submit_utils import config_uglify
from utils import get_gpu_map


def worker(config, name, key, val):
    selected_gpu = multiprocessing.current_process()._identity[0] - 1
    command = ["python", "search.py", "--name", "{}_{}_{}".format(name, key, val),
               "--config_file", config, "--{}".format(key), str(val)]
    print("Worker: Selecting GPU {}, Command: {}".format(selected_gpu, " ".join(command)))
    env = {}
    env.update(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(selected_gpu)})
    process = subprocess.run(command, env=env)
    print(process)


parser = ArgumentParser()
parser.add_argument("--config_file", required=True)
parser.add_argument("--tune_key", required=True)
parser.add_argument("--tune_range", required=True)
args = parser.parse_args()

if args.tune_key == "seed":
    tune_range = list(range(int(args.tune_range)))
elif args.tune_key == "epochs":
    tune_range = list(map(int, args.tune_range.split(",")))
elif args.tune_key == "step_order":
    tune_range = list(args.tune_range.split(","))
else:
    raise NotImplementedError

config_files = [args.config_file]
if os.path.isdir(args.config_file):
    config_files = []
    for rt, dirs, files in os.walk(args.config_file):
        for file in files:
            if file.endswith("yml") or file.endswith("yaml"):
                config_files.append(os.path.join(rt, file))
print("Found config files: {}".format(config_files))

config_and_names = [(config, "v100_" + config_uglify(config)) for config in config_files]

gpu_map = get_gpu_map()
with multiprocessing.Pool(len(gpu_map)) as pool:
    pool.starmap(worker, [(config, name, args.tune_key, val)
                          for val in tune_range for config, name in config_and_names])
