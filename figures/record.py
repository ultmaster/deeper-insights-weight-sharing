import os

import numpy as np
import pandas as pd
import re
from typing import List, Iterable

EXPECTED_SUBGRAPH_NUMBER = 64


def convert_subgraph_label_to_index(label):
    ret = 0
    for t in label:
        ret = ret * 4 + (int(t) - 1)
    return ret


def convert_subgraph_index_to_label(index):
    ret = ""
    for i in range(3):
        ret = str(index % 4 + 1) + ret
        index //= 4
    return ret


def sorted_dedup(iterable: Iterable):
    return sorted(list(set(iterable)))


class ValidationSingleRecord:
    def __init__(self, epoch, step, subgraph_val, subgraph_trn, acc, dense_validation):
        self.epoch = epoch
        self.step = step
        self.acc = acc
        self.subgraph_val = subgraph_val  # subgraph now validating
        self.subgraph_trn = subgraph_trn  # subgraph just trained
        self.dense_validation = dense_validation

    def __str__(self):
        return "Ep={}, Step={}, Acc={:.4f}, Trn={}, Val={}".format(self.epoch, self.step, self.acc,
                                                                   self.subgraph_trn, self.subgraph_val)

    def __repr__(self):
        return "SingleRecord({})".format(str(self))


class Record:
    def __init__(self, logger, log_files, format=None):
        self.logger = logger
        self.records: List[ValidationSingleRecord] = []
        self.grouping_info = {}
        self.finetune_step = 0
        for idx, file in enumerate(log_files):
            if format == "finetune":
                subgraphs = self.process_finetune_file(file)
                self.finetune_step = int(os.path.basename(file).split("_")[1])
            else:
                subgraphs = self.process_file(file)
            if not subgraphs:
                self.logger.error("No subgraph found in {}".format(file))
            for subgraph in subgraphs:
                if subgraph in self.grouping_info:
                    raise ValueError("Subgraph {} has been trained twice in different groups.".format(subgraph))
                self.grouping_info[subgraph] = idx

    @property
    def groups(self):
        return sorted_dedup(self.grouping_info.values())

    @property
    def columns(self):
        return sorted(list(self.grouping_info.keys()))

    @property
    def group_number(self):
        return len(self.groups)

    @property
    def grouping_numpy(self):
        ret = np.full(EXPECTED_SUBGRAPH_NUMBER, -1, dtype=np.int)
        for g, v in self.grouping_info.items():
            ret[g] = v
        return ret

    def filter_records(self, by, cutoff):
        assert by in ["epochs", "steps"], "by not in epochs and steps"
        ret = dict()
        if by == "epochs":
            for d in self.records:
                if d.step > cutoff > 0:
                    continue
                k = (d.epoch, d.subgraph_val)
                if k not in ret or ret[k].step < d.step:
                    ret[k] = d
        elif by == "steps":
            for d in self.records:
                if d.step > cutoff > 0:
                    continue
                if d.dense_validation:
                    k = (d.step, d.subgraph_val)
                    if k not in ret or not ret[k].dense_validation:
                        ret[k] = d
        steps = sorted_dedup([k for k, _ in ret.keys()])
        if len(steps) <= 2:
            raise ValueError("Too few steps for analysis")
        return steps, ret

    def validation_acc_dataframe(self, by, cutoff=0):
        steps, raw_data = self.filter_records(by, cutoff)
        ret_as_dict = dict()
        for column in self.columns:
            ret_as_dict[column] = [raw_data[step, column].acc for step in steps]
        return pd.DataFrame(ret_as_dict, index=steps, columns=self.columns)

    def grouping_subgraph_training_dataframe(self, by, cutoff=0):
        steps, raw_data = self.filter_records(by, cutoff)
        trn_subgraph_by_group = dict()
        for column in self.columns:
            trn_subgraph_by_group[self.grouping_info[column]] = [raw_data[step, column].subgraph_trn for step in steps]
        return pd.DataFrame(trn_subgraph_by_group, index=steps,
                            columns=sorted_dedup(self.grouping_info.values()))

    def process_finetune_file(self, file_path):
        name_regex = r"Subgraph name: (\d+)"
        current_epoch, current_step = 1, -1
        training_step_regex = r"Train: \[\s*(\d+)/(\d+)\] Step (\d+)/(\d+)"
        validation_regex = r"Valid: \[\s*(\d+)/\s*\d+\] Final Prec@1 ([\d.]+)%"
        with open(file_path, "r") as fp:
            for line in fp.readlines():
                grp = re.search(name_regex, line)
                if grp is not None:
                    subgraph_name = convert_subgraph_label_to_index(grp.group(1))
                grp = re.search(training_step_regex, line)
                if grp is not None:
                    current_epoch = int(grp.group(1))
                    current_step = (int(grp.group(1)) - 1) * (int(grp.group(4)) + 1) + int(grp.group(3)) + 1
                grp = re.search(validation_regex, line)
                if grp is not None:
                    current_epoch = int(grp.group(1))
                    acc = float(grp.group(2)) / 100
                    self.records.append(ValidationSingleRecord(current_epoch, current_step, subgraph_name,
                                                               subgraph_name, acc, True))
        return [subgraph_name]

    def process_file(self, file_path):
        current_epoch, current_step, current_archit = 1, -1, 0
        ret = []
        parameters = {}
        archit_order, subgraphs = None, set()
        valid_ep_regex = r"Valid: \[\s*(\d+)/\s*(\d+)\]"
        training_step_regex = r"Train: \[\s*(\d+)/(\d+)\] Step (\d+)/(\d+)"
        training_archit_regex = r"Archit: (\d+)"
        top_acc_label = "Valid: Top accuracy: "
        dense_validation_regex = "Entering dense validation steps"
        parameter_status = -1
        with open(file_path, "r") as fp:
            for line in fp.readlines():
                if parameter_status < 0:
                    if line.find("Parameter") != -1:
                        parameter_status = 0
                    continue
                elif parameter_status == 0:
                    grp = re.search(r"([A-Z0-9_]+)=(.*)", line)
                    if grp is not None:
                        parameters[grp.group(1).lower()] = grp.group(2)
                    else:
                        parameter_status = 1
                        try:
                            archit_order = eval(parameters["designated_subgraph"])
                            if archit_order is None:
                                raise ValueError
                        except:
                            self.logger.warning("Designated subgraphs not found, assuming default")
                            archit_order = list(range(EXPECTED_SUBGRAPH_NUMBER))
                        subgraphs = set(archit_order)
                    continue

                grp = re.search(valid_ep_regex, line)
                if grp is not None:
                    current_epoch = int(grp.group(1))
                grp = re.search(training_step_regex, line)
                if grp is not None:
                    current_step = (int(grp.group(1)) - 1) * (int(grp.group(4)) + 1) + int(grp.group(3)) + 1
                    grp = re.search(training_archit_regex, line)
                    if grp is not None:
                        current_archit = convert_subgraph_label_to_index(grp.group(1))
                        if current_archit != archit_order[(current_step - 1) % len(archit_order)] and not (
                                "step_order" in parameters and
                                any(parameters["step_order"].startswith(k) for k in ["one", "every"])):
                            raise ValueError("Unexpected architecture found for step {}".format(current_step))
                        else:
                            archit_order[(current_step - 1) % len(archit_order)] = current_archit
                    current_archit = archit_order[(current_step - 1) % len(archit_order)]  # deal with missing info case
                grp = re.search(dense_validation_regex, line)
                if grp is not None:
                    parameter_status = 2
                if line.find(top_acc_label) != -1:
                    acc = eval(line[line.find(top_acc_label) + len(top_acc_label):])
                    for t, v in acc:
                        ret.append(ValidationSingleRecord(current_epoch, current_step,
                                                          convert_subgraph_label_to_index(t),
                                                          current_archit, v, parameter_status > 1))
        self.records.extend(ret)
        return subgraphs


def record_factory(logger, log_dir, expected_total_subgraphs):
    # return a main record and possibly a few finetune records
    main_logs = []
    finetune_records = []
    for rt, dirs, files in os.walk(log_dir):
        # process for each directory
        finetune_record = False
        logs = []
        for file in files:
            if not file.endswith(".log"):
                continue
            if file.startswith("finetune_"):
                finetune_record = True
            file_path = os.path.join(rt, file)
            logs.append(file_path)
        if finetune_record:
            finetune_records.append(Record(logger, logs, format="finetune"))
        else:
            main_logs.extend(logs)
    logger.info("Found {} finetune records".format(len(finetune_records)))
    main_record = Record(logger, main_logs)
    for r in finetune_records:
        assert len(r.columns) == expected_total_subgraphs, \
            "Finetune record of length {} != {}".format(len(r.columns), expected_total_subgraphs)
    assert len(main_record.columns) == expected_total_subgraphs, \
        "Main record of length {} != {}".format(len(main_record.columns), expected_total_subgraphs)
    return main_record, finetune_records
