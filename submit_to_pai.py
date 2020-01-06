import copy
import os
from argparse import ArgumentParser

from openpaisdk.core import ClusterList

from submit_utils import config_uglify

parser = ArgumentParser()
parser.add_argument("experiment", type=str)
parser.add_argument("--alias", default="cluster", type=str)
parser.add_argument("--virtualcluster", default=None, type=str)
parser.add_argument("--nodes", default=4, type=int)
parser.add_argument("--repeat", default=4, type=int)
parser.add_argument("--key", default="seed", type=str)
parser.add_argument("--finetune", default=False, action="store_true")

args = parser.parse_args()

# list experiments
if os.path.isfile(args.experiment):
    experiments = [args.experiment]
elif os.path.isdir(args.experiment):
    experiments = []
    for root, dirs, files in os.walk(args.experiment):
        for file in files:
            file_path = os.path.join(root, file)
            experiments.append(file_path)
else:
    raise ValueError("Requested files or directories do not exist.")

# job config
command_base = "git clone -b dev-renew https://github.com/repo_address && " \
               "cd repo && bash pai/{}.sh --name test --config_file {}"
config_base = {
    "jobName": "",
    "image": "ultmaster/nni:darts",
    "virtualCluster": "nni",
    "taskRoles": [
        {
            "name": "main",
            "taskNumber": 0,
            "cpuNumber": 4,
            "gpuNumber": 1,
            "memoryMB": 16384,
            "shmMB": 16384,
            "command": "",
            "portList": [
                {
                    "label": "tensorboard",
                    "beginAt": 0,
                    "portNumber": 1
                }
            ]
        }
    ],
    "jobEnvs": {
    }
}

clusters = ClusterList().load()
client = clusters.get_client(args.alias)
for experiment in experiments:
    # submit one by one
    if args.virtualcluster:
        virtualcluster = args.virtualcluster
    else:
        resources = client.available_resources()
        virtualcluster = max(resources.keys(), key=lambda k: resources[k]["GPUs"])

    experiment_alias = config_uglify(experiment) + "_" + args.key
    print("Submitting job: {}, as {}".format(experiment, experiment_alias))
    config = copy.deepcopy(config_base)
    config["taskRoles"][0]["taskNumber"] = args.nodes
    config["virtualCluster"] = virtualcluster
    if args.finetune:
        config["jobEnvs"]["FINETUNE"] = 1
    for i in range(args.repeat):
        config["jobName"] = experiment_alias + ("_%02d" % i)
        command = command_base.format("finetune" if args.finetune else "run", experiment)
        if args.key == "step_order_one":
            command += " --step_order one_{}".format(i)
        elif args.key == "step_order_every":
            command += " --step_order every_{}".format(i)
        elif args.key == "seed":
            command += " --seed {}".format(i)
        config["taskRoles"][0]["command"] = command
        client.rest_api_submit(config)
