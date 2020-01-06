import logging
import os
import traceback
from argparse import ArgumentParser
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from record import Record, record_factory, EXPECTED_SUBGRAPH_NUMBER, convert_subgraph_index_to_label
from visualize import boxplot, lineplot, heatmap, scatterplot, MultiPageContext, errorbar


def rankdata_greater(row):
    return stats.rankdata(-row, method="ordinal")


def get_consecutive_rank_tau(df):
    ret = np.zeros((len(df) - 1,))
    for i in range(1, len(df)):
        ret[i - 1], _ = stats.kendalltau(df.iloc[i - 1], df.iloc[i])
    return ret


def get_tau_curves_by_groups(df, gt, group_table, groups):
    return {cur: get_tau_along_epochs(df, gt, np.where(group_table == cur)[0]) for cur in groups}


def get_tau_along_epochs(df, gt, group):
    return np.array([stats.kendalltau(row[group].values, gt[group])[0] for _, row in df.iterrows()])


def get_tau_along_epochs_combining_best_groups(df, gt, group_table, groups, universe):
    tau_curves_by_groups = get_tau_curves_by_groups(df, gt, group_table, groups)
    ref_gt_acc = np.zeros((len(df), EXPECTED_SUBGRAPH_NUMBER))
    for cur in groups:
        # for each group, enumerate the epochs from the most obedient to most rebellious
        for i, loc in enumerate(np.argsort(-tau_curves_by_groups[cur])):
            group_mask = np.where(group_table == cur)[0]
            ref_gt_acc[i][group_mask] = df[group_mask].iloc[loc]
    ref_gt_acc_tau = np.array([stats.kendalltau(acc[universe], gt[universe])[0] for acc in ref_gt_acc])
    return ref_gt_acc, ref_gt_acc_tau


def get_top_k_acc_rank(acc_table, acc_gt):
    gt_rank = rankdata_greater(acc_gt)
    idx = np.stack([np.argsort(-row) for row in acc_table])
    top_acc = np.maximum.accumulate(acc_gt[idx], 1)
    top_rank = np.minimum.accumulate(gt_rank[idx], 1)
    return top_acc, top_rank


def report_mean_std_max_min(analysis_dir, logger, name, arr):
    np.savetxt(os.path.join(analysis_dir, "METRICS-{}.txt".format(name)),
               np.array([np.mean(arr), np.std(arr), np.max(arr), np.min(arr)]))
    logger.info("{}: mean={:.4f}, std={:.4f}, max={:.4f}, min={:.4f}".format(name, np.mean(arr), np.std(arr),
                                                                             np.max(arr), np.min(arr)))


def stack_with_index(index, row):
    return np.stack([index, row]).T


def plot_top_k_variance_chart(filepath, index, top_acc, top_rank, gt_acc, topk):
    gt_acc_index = np.argsort(-gt_acc)
    curves = []
    for k in topk:
        curves.append(stack_with_index(index, np.array([gt_acc[gt_acc_index[k - 1]]] * top_acc.shape[0])))
        curves.append(stack_with_index(index, top_acc[:, k - 1]))
    lineplot(curves, filepath=filepath + "_acc")

    curves = []
    for k in topk:
        curves.append(stack_with_index(index, np.array([k] * top_acc.shape[0])))
        curves.append(stack_with_index(index, top_rank[:, k - 1]))
    lineplot(curves, filepath=filepath + "_rank", inverse_y=True)


def pipeline_for_single_instance(logger, analysis_dir, main: Record, finetune: List[Record], by: str, gt: np.ndarray):
    logger.info("Analysing results for {}".format(analysis_dir))
    main_df = main.validation_acc_dataframe(by)
    main_archit = main.grouping_subgraph_training_dataframe(by)
    main_grouping = main.grouping_numpy

    os.makedirs(analysis_dir, exist_ok=True)

    # Save raw data
    main_df.to_csv(os.path.join(analysis_dir, "val_acc_all_epochs.csv"), index=True)
    np.savetxt(os.path.join(analysis_dir, "group_info.txt"), main_grouping, "%d")

    # correlation between subgraphs
    corr_matrix = main_df.corr().values
    heatmap(corr_matrix, filepath=os.path.join(analysis_dir, "corr_heatmap"))
    np.savetxt(os.path.join(analysis_dir, "corr_heatmap.txt"), corr_matrix)

    # Consecutive tau (single)
    consecutive_taus = get_consecutive_rank_tau(main_df)
    lineplot([np.array(list(zip(main_df.index[1:], consecutive_taus)))],
             filepath=os.path.join(analysis_dir, "consecutive_tau_single"))

    # GT rank (for color reference)
    gt_rank = rankdata_greater(gt)
    gt_rank_color = 1 - gt_rank / EXPECTED_SUBGRAPH_NUMBER
    # in some cases, it could be a subset of 64 subgraphs; process this later

    # Acc variance (lineplot)
    acc_curves = [np.array(list(zip(main_df.index, main_df[i]))) for i in main_df.columns]
    subgraph_markers = [[] for _ in range(EXPECTED_SUBGRAPH_NUMBER)]
    if len(main.groups) != len(main.columns):  # hide it for ground truth
        for i, (_, row) in enumerate(main_archit.iterrows()):
            for k in filter(lambda k: k >= 0, row.values):
                subgraph_markers[k].append(i)
    else:
        logger.info("Markers hidden because groups == columns")

    lineplot(acc_curves, filepath=os.path.join(analysis_dir, "acc_curve_along_epochs"),
             color=[gt_rank_color[i] for i in main_df.columns], alpha=0.7,
             markers=[subgraph_markers[i] for i in main_df.columns],
             fmt=["-D"] * len(acc_curves))

    # Rank version of df
    df_rank = main_df.apply(rankdata_greater, axis=1, result_type="expand")
    df_rank.columns = main_df.columns

    # Rank variance (lineplot)
    rank_curves = [np.array(list(zip(df_rank.index, df_rank[i]))) for i in df_rank.columns]
    lineplot(rank_curves, filepath=os.path.join(analysis_dir, "rank_curve_along_epochs"),
             color=[gt_rank_color[i] for i in df_rank.columns], alpha=0.7, inverse_y=True, markers=subgraph_markers)

    # Rank variance for top-5 subgraphs found at half and end
    # recalculate for original order
    for loc in [len(main_df) // 2, len(main_df) - 1]:
        selected_rank_curves = [rank_curves[i] for i in np.argsort(-main_df.iloc[loc])[:5]]
        lineplot(selected_rank_curves, inverse_y=True,
                 filepath=os.path.join(analysis_dir, "rank_curves_along_epochs_for_ep{}".format(main_df.index[loc])))

    # Rank variance (boxplot), sorted by the final rank
    boxplot(sorted(df_rank.values.T, key=lambda d: d[-1]),
            filepath=os.path.join(analysis_dir, "rank_boxplot_along_epochs_sorted_final_rank"),
            inverse_y=True)

    gt_order = np.argsort(-gt)

    # Group info
    np.savetxt(os.path.join(analysis_dir, "group_info_sorted_gt.txt"), main_grouping[gt_order], "%d")

    # Rank variance (boxplot), sorted by ground truth
    boxplot([df_rank[i] for i in gt_order if i in df_rank.columns], inverse_y=True,
            filepath=os.path.join(analysis_dir, "rank_boxplot_along_epochs_sorted_gt_rank"))
    boxplot([df_rank[i][-10:] for i in gt_order if i in df_rank.columns], inverse_y=True,
            filepath=os.path.join(analysis_dir, "rank_boxplot_along_epochs_sorted_gt_rank_last_10"))

    # Tau every epoch
    gt_tau_data = get_tau_along_epochs(main_df, gt, main.columns)
    report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window", gt_tau_data)
    lineplot([stack_with_index(main_df.index, gt_tau_data)],
             filepath=os.path.join(analysis_dir, "tau_curve_along_epochs"))

    if finetune:
        # Finetune curves
        for data in finetune:
            try:
                finetune_step = data.finetune_step
                if by == "epochs":
                    finetune_step //= 196
                half_length = len(main_df.loc[main_df.index <= finetune_step])
                finetune_df = data.validation_acc_dataframe(by, cutoff=finetune_step).iloc[:half_length]
                if finetune_step < min(main_df.index) - 1 or finetune_step > max(main_df.index) + 1:
                    continue
                finetune_df.index += finetune_step
                finetune_curves = [np.array([[finetune_step, main_df.loc[finetune_step, i]]] +
                                            list(zip(finetune_df.index, finetune_df[i])))
                                   for i in main_df.columns]
                finetune_tau_curve = get_tau_along_epochs(finetune_df, gt, data.columns)
                finetune_colors = [gt_rank_color[i] for i in finetune_df.columns]
                logger.info("Finetune step {}, found {} finetune curves".format(finetune_step, len(finetune_curves)))
                lineplot([c[:half_length] for c in acc_curves] + finetune_curves,
                         filepath=os.path.join(analysis_dir,
                                               "acc_curve_along_epochs_finetune_{}".format(finetune_step)),
                         color=[gt_rank_color[i] for i in main_df.columns] + finetune_colors, alpha=0.7,
                         fmt=["-"] * len(acc_curves) + [":"] * len(finetune_curves))
                lineplot([stack_with_index(main_df.index, gt_tau_data)[:half_length],
                          np.concatenate((np.array([[finetune_step, gt_tau_data[half_length - 1]]]),
                                         stack_with_index(finetune_df.index, finetune_tau_curve)))],
                         filepath=os.path.join(analysis_dir,
                                               "tau_curve_along_epochs_finetune_{}".format(finetune_step)),
                         color=["tab:blue", "tab:blue"], alpha=1, fmt=["-", ":"])
            except ValueError:
                pass

    # Tau every epoch group by groups
    grouping_info_backup = main.grouping_info.copy()
    divide_group = main.group_number == 1 and len(main.columns) == 64
    for partition_file in [None] + list(os.listdir("assets")):
        suffix = ""
        if partition_file is not None:
            if not partition_file.startswith("partition"):
                continue
            if not divide_group:
                continue
            suffix = "_" + os.path.splitext(partition_file)[0]
            # regrouping
            main.grouping_info = {idx: g for idx, g in enumerate(np.loadtxt(os.path.join("assets", partition_file),
                                                                            dtype=np.int))}

        tau_curves_by_groups = get_tau_curves_by_groups(main_df, gt, main.grouping_numpy, main.groups)
        tau_curves_by_groups_mean = [np.mean(tau_curves_by_groups[cur]) for cur in main.groups]
        tau_curves_by_groups_std = [np.std(tau_curves_by_groups[cur]) for cur in main.groups]
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-By-Groups-Mean{}".format(suffix),
                                np.array(tau_curves_by_groups_mean))
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-By-Groups-Std{}".format(suffix),
                                np.array(tau_curves_by_groups_std))
        tau_curves_by_groups_for_plt = [stack_with_index(main_df.index, tau_curves_by_groups[cur])
                                        for cur in main.groups]

        pd.DataFrame(tau_curves_by_groups, columns=main.groups, index=main_df.index).to_csv(
            os.path.join(analysis_dir, "tau_curves_by_groups{}.csv".format(suffix))
        )
        lineplot(tau_curves_by_groups_for_plt,
                 filepath=os.path.join(analysis_dir, "tau_curves_by_groups{}".format(suffix)))

        # Acc curves (by group)
        with MultiPageContext(os.path.join(analysis_dir, "acc_curve_along_epochs_group_each{}".format(suffix))) as pdf:
            for g in range(main.group_number):
                subgraphs = np.where(main.grouping_numpy == g)[0]
                gt_rank_group = [gt_rank_color[i] for i in subgraphs]
                subgraph_names = list(map(convert_subgraph_index_to_label, subgraphs))
                subgraph_names_ranks = ["{} (Rank {})".format(name, gt_rank[i])
                                        for name, i in zip(subgraph_names, subgraphs)]
                # cannot leverage acc_curves, because it's a list, this can be a subset, which cannot be used as index
                lineplot([np.array(list(zip(main_df.index, main_df[i]))) for i in subgraphs] +
                         [stack_with_index(main_df.index, [gt[i]] * len(main_df.index)) for i in subgraphs],
                         context=pdf, color=gt_rank_group * 2, alpha=0.8, labels=subgraph_names_ranks,
                         fmt=["-D"] * len(subgraphs) + ["--"] * len(subgraphs),
                         markers=[subgraph_markers[i] for i in subgraphs] + [[]] * len(subgraphs),
                         title="Group {}, Subgraph {} -- {}".format(g, "/".join(map(str, subgraphs)),
                                                                    "/".join(subgraph_names)))

    main.grouping_info = grouping_info_backup

    # Tau among steps
    for k in (10, 64):
        max_tau_calc = min(k, len(main_df))
        tau_correlation = np.zeros((max_tau_calc, max_tau_calc))
        for i in range(max_tau_calc):
            for j in range(max_tau_calc):
                tau_correlation[i][j] = stats.kendalltau(main_df.iloc[-i - 1], main_df.iloc[-j - 1])[0]
        heatmap(tau_correlation, filepath=os.path.join(analysis_dir, "tau_correlation_last_{}".format(k)))
        np.savetxt(os.path.join(analysis_dir, "tau_correlation_last_{}.txt".format(k)), tau_correlation)
        tau_correlation = tau_correlation[np.triu_indices_from(tau_correlation, k=1)]
        report_mean_std_max_min(analysis_dir, logger, "Tau-as-Corr-Last-{}".format(k), tau_correlation)

    # Calculate best tau and log
    ref_gt_acc, ref_gt_acc_tau = get_tau_along_epochs_combining_best_groups(main_df, gt, main_grouping, main.groups,
                                                                            main.columns)
    pd.DataFrame(ref_gt_acc).to_csv(os.path.join(analysis_dir,
                                                 "acc_epochs_combining_different_epochs_sorted_gt.csv"))
    lineplot([stack_with_index(np.arange(len(ref_gt_acc_tau)), ref_gt_acc_tau)],
             filepath=os.path.join(analysis_dir, "tau_curve_epochs_sorted_combining_different_epochs"))

    # Show subgraph for each batch
    scatterplot([stack_with_index(main_archit.index, main_archit[col]) for col in main_archit.columns],
                filepath=os.path.join(analysis_dir, "subgraph_id_for_each_batch_validated"))

    # Substituted with ground truth rank
    scatterplot([stack_with_index(main_archit.index, gt_rank[main_archit[col]]) for col in main_archit.columns],
                filepath=os.path.join(analysis_dir, "subgraph_rank_for_each_batch_validated"),
                inverse_y=True)

    # Top-K-Rank
    top_acc, top_rank = get_top_k_acc_rank(main_df.values, gt)
    plot_top_k_variance_chart(os.path.join(analysis_dir, "top_k_along_epochs"), main_df.index,
                              top_acc, top_rank, gt, (1, 3))

    # Observe last window (for diff. epochs)
    for k in (10, 64,):
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window-Last-{}".format(k), gt_tau_data[-k:])
        for v in (1, 3):
            report_mean_std_max_min(analysis_dir, logger, "Top-{}-Rank-Last-{}".format(v, k), top_rank[-k:, v - 1])


def pipeline_for_inter_instance(logger, analysis_dir, data, by, gt):
    logger.info("Analysing results for {}".format(analysis_dir))

    data_as_df = [d.validation_acc_dataframe(by) for d in data]
    os.makedirs(analysis_dir, exist_ok=True)
    subgraphs = data[0].columns
    for d in data:
        assert d.columns == subgraphs

    final_acc = np.zeros((len(data), len(subgraphs)))
    for i, df in enumerate(data_as_df):
        final_acc[i] = df.iloc[-1]

    # Consecutive tau (multi)
    lineplot([np.array(list(zip(df.index[1:], get_consecutive_rank_tau(df)))) for df in data_as_df],
             filepath=os.path.join(analysis_dir, "taus_consecutive_epochs"))

    # Final acc distribution
    boxplot(final_acc, filepath=os.path.join(analysis_dir, "final_acc"))

    # Final rank distribution
    final_rank = np.stack([rankdata_greater(row) for row in final_acc])
    boxplot(final_rank, filepath=os.path.join(analysis_dir, "final_rank_boxplot"), inverse_y=True)

    # GT-Tau
    gt_tau = np.array([stats.kendalltau(row, gt[subgraphs])[0] for row in final_acc])
    np.savetxt(os.path.join(analysis_dir, "inst_gt_tau.txt"), gt_tau)
    report_mean_std_max_min(analysis_dir, logger, "GT-Tau", gt_tau)

    # Tau every epoch
    tau_data = [get_tau_along_epochs(df, gt, subgraphs) for df in data_as_df]
    tau_data_mean_over_instances = np.mean(np.stack(tau_data, axis=0), axis=0)
    report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window", np.concatenate(tau_data))
    tau_curves = [stack_with_index(df.index, tau_d) for df, tau_d in zip(data_as_df, tau_data)]
    lineplot(tau_curves, filepath=os.path.join(analysis_dir, "tau_curve_along_epochs"))
    for k in (10, 64):
        tau_data_clip = [t[-k:] for t in tau_data]
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window-Last-{}-Mean".format(k),
                                np.array([np.mean(t) for t in tau_data_clip]))
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window-Last-{}-Std".format(k),
                                np.array([np.std(t) for t in tau_data_clip]))
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window-Last-{}-Max".format(k),
                                np.array([np.max(t) for t in tau_data_clip]))
        report_mean_std_max_min(analysis_dir, logger, "GT-Tau-In-Window-Last-{}-Min".format(k),
                                np.array([np.min(t) for t in tau_data_clip]))

        acc_data = [np.mean(df.iloc[-k:].values, axis=0) for df in data_as_df]
        report_mean_std_max_min(analysis_dir, logger, "Acc-Mean-In-Window-Last-{}-Mean".format(k),
                                np.array([np.mean(x) for x in acc_data]))
        report_mean_std_max_min(analysis_dir, logger, "Acc-Mean-In-Window-Last-{}-Std".format(k),
                                np.array([np.std(x) for x in acc_data]))

    # S-Tau (last 5 epochs)
    s_tau = np.zeros((min(map(lambda d: len(d), data_as_df)), len(data), len(data)))
    for k in range(len(s_tau)):
        for i, table1 in enumerate(data_as_df):
            for j, table2 in enumerate(data_as_df):
                s_tau[k][i][j], _ = stats.kendalltau(table1.iloc[k], table2.iloc[k])
    np.savetxt(os.path.join(analysis_dir, "inter_inst_s_tau.txt"), s_tau[-1])
    heatmap(s_tau[0], filepath=os.path.join(analysis_dir, "inter_inst_last_s_tau_heatmap"), figsize=(10, 10))
    if len(data) > 1:
        upper = np.triu_indices_from(s_tau[0], k=1)
        report_mean_std_max_min(analysis_dir, logger, "S-Tau-Last", s_tau[-1][upper])
        s_tau_mean = np.mean(s_tau[:, upper[0], upper[1]], axis=1)
        s_tau_std = np.std(s_tau[:, upper[0], upper[1]], axis=1)
        report_mean_std_max_min(analysis_dir, logger, "S-Tau-Min", s_tau[np.argmin(s_tau_mean)][upper])
        s_tau_errorbar = np.stack([np.arange(len(s_tau)), s_tau_mean, s_tau_std], axis=1)
        errorbar([s_tau_errorbar], filepath=os.path.join(analysis_dir, "inter_inst_s_tau_curve"))

        # S-Tau (without variance)
        lineplot([s_tau_errorbar[:, :2]], fmt=["-o"],
                 filepath=os.path.join(analysis_dir, "inter_inst_s_tau_curve_along_epochs_without_var"))

        # Compare with GT-Tau
        lineplot(tau_curves + [s_tau_errorbar], fmt=["-"] * len(tau_curves) + [":"],
                 filepath=os.path.join(analysis_dir, "tau_curve_along_epochs_compare_to_s_tau"))

        lineplot([np.stack([np.arange(len(tau_data_mean_over_instances)), tau_data_mean_over_instances], axis=1)] +
                 [s_tau_errorbar], fmt=["-", ":"],
                 filepath=os.path.join(analysis_dir, "tau_curve_along_epochs_mean_compare_to_s_tau"))

    # Final rank dist (sorted by GT)
    gt_rank = sorted(np.arange(len(subgraphs)), key=lambda i: gt[subgraphs[i]], reverse=True)
    final_rank_resorted = final_rank[:, gt_rank]
    boxplot(final_rank_resorted, filepath=os.path.join(analysis_dir, "final_rank_boxplot_sorted_gt"),
            inverse_y=True)

    # Tau sorted
    ref_gt_acc_taus = []
    for df, raw in zip(data_as_df, data):
        _, ref_gt_acc_tau = get_tau_along_epochs_combining_best_groups(df, gt, raw.grouping_numpy, raw.groups,
                                                                       subgraphs)
        ref_gt_acc_taus.append(stack_with_index(np.arange(len(ref_gt_acc_tau)), ref_gt_acc_tau))
    lineplot(ref_gt_acc_taus, filepath=os.path.join(analysis_dir, "tau_curves_sorted_combining_different_epochs"))

    # Top-K-Rank
    top_acc, top_rank = get_top_k_acc_rank(final_acc, gt)
    topk = (1, 3)
    for k in topk:
        report_mean_std_max_min(analysis_dir, logger, "Top-{}-Acc".format(k), top_acc[:, k - 1])
        report_mean_std_max_min(analysis_dir, logger, "Top-{}-Rank".format(k), top_rank[:, k - 1])
    plot_top_k_variance_chart(os.path.join(analysis_dir, "inst_top_k"), np.arange(len(top_acc)),
                              top_acc, top_rank, gt, topk)

    # Average final acc
    avg_acc = np.mean(final_acc, axis=0)
    np.savetxt(os.path.join(analysis_dir, "average_final_acc.txt"), avg_acc)
    std_acc = np.std(final_acc, axis=0)
    np.savetxt(os.path.join(analysis_dir, "std_final_acc.txt"), std_acc)


def pipeline(keyword, gt, expected_total_subgraphs):
    root_dir = "data"
    analysis_dir = os.path.join("analysis", keyword)
    os.makedirs(analysis_dir, exist_ok=True)

    logger = logging.getLogger("analysis")
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(os.path.join(analysis_dir, "run.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    data = []
    for folder in sorted(os.listdir(root_dir)):
        if keyword not in folder:
            continue
        logger.info("Processing logs from \"{}\"".format(folder))
        try:
            main_record, finetune_records = record_factory(logger, os.path.join(root_dir, folder),
                                                           expected_total_subgraphs)
            if not args.skip_single:
                for by in ["epochs", "steps"]:
                    analysis_dir_for_df = os.path.join(analysis_dir, by + "_" + folder.split("_")[-1])
                    try:
                        pipeline_for_single_instance(logger, analysis_dir_for_df, main_record, finetune_records, by, gt)
                    except ValueError as e:
                        logger.warning(e)
            data.append(main_record)
        except (AssertionError, ValueError) as e:
            logger.warning(e)

    logger.info("Found {} instances for {}".format(len(data), keyword))
    if not data:
        return

    for by in ["epochs", "steps"]:
        try:
            analysis_dir_for_inter = os.path.join(analysis_dir, by + "_inter")
            pipeline_for_inter_instance(logger, analysis_dir_for_inter, data, by, gt)
        except ValueError as e:
            logger.warning(e)
            traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("keyword")
    parser.add_argument("--gt", "--ground-truth", default="assets/gt_acc.txt", type=str)
    parser.add_argument("--expected-total-subgraphs", type=int, default=64)
    parser.add_argument("--skip-single", default=False, action="store_true")
    args = parser.parse_args()
    ground_truth = np.loadtxt(args.gt)
    pipeline(args.keyword, ground_truth, args.expected_total_subgraphs)
