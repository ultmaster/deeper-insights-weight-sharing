from visualize import replay
import matplotlib

del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()


if __name__ == "__main__":
    replay(["analysis/final_share_all_ep200_45766df6/epochs_inter/REPLAY_inter_inst_s_tau_curve_along_epochs_without_var.pkl",
            "analysis/final_share_all_ep200_f829c584_step_order_every/epochs_inter/REPLAY_inter_inst_s_tau_curve_along_epochs_without_var.pkl",
            "analysis/final_group_stable_partition_g64_c42fdedd/epochs_inter/REPLAY_inter_inst_s_tau_curve_along_epochs_without_var.pkl"],
           figsize=(10, 7), fontsize=25, margins=[0, 0.1], cutoff=100, x_offset=1, filepath="outputs/diff_order_s_tau_every_epochs",
           xlabel="Epochs", ylabel="S-Tau", labels=["Diff. seeds", "Diff. orders (shuffle)", "Ground truth"],
           fmt=["-o", "-.D", "--"])

    replay("analysis/final_share_all_ep200_f829c584_step_order_every/epochs_inter/REPLAY_final_rank_boxplot_sorted_gt.pkl",
           figsize=(8, 4), fontsize=20, filepath="outputs/final_rank_boxplot_10_instances",
           xticklabels=[str(i) if i % 5 == 1 else "" for i in range(1, 65)],
           xlabel="Subgraphs", ylabel="Rank")
    replay("analysis/final_share_all_ep200_f829c584_step_order_every/epochs_01/REPLAY_rank_boxplot_along_epochs_sorted_gt_rank_last_10.pkl",
           figsize=(8, 4), fontsize=20, filepath="outputs/final_rank_boxplot_10_epochs",
           xticklabels=[str(i) if i % 5 == 1 else "" for i in range(1, 65)],
           xlabel="Subgraphs", ylabel="Rank")

    replay("analysis/share_all_finetune_ep100_bba7/steps_01/REPLAY_acc_curve_along_epochs_finetune_19599.pkl",
           figsize=(10, 6), fontsize=25, filepath="outputs/accuracy_finetune_ep100",
           xlabel="Mini-batches", ylabel="Accuracy", margins=[0, 0.05])
    replay("analysis/share_all_finetune_ep200_afc8/steps_03/REPLAY_acc_curve_along_epochs_finetune_39199.pkl",
           figsize=(10, 6), fontsize=25, filepath="outputs/accuracy_finetune_ep200",
           xlabel="Mini-batches", ylabel="Accuracy", margins=[0, 0.05])
    replay("analysis/share_all_finetune_ep100_bba7/steps_01/REPLAY_tau_curve_along_epochs_finetune_19599.pkl",
           figsize=(10, 4), fontsize=25, filepath="outputs/gt_tau_finetune_ep100",
           xlabel="Mini-batches", ylabel="GT-Tau", color=["k", "k"], margins=[0, 0.05])
    replay("analysis/share_all_finetune_ep200_afc8/steps_03/REPLAY_tau_curve_along_epochs_finetune_39199.pkl",
           figsize=(10, 4), fontsize=25, filepath="outputs/gt_tau_finetune_ep200",
           xlabel="Mini-batches", ylabel="GT-Tau", color=["k", "k"], margins=[0, 0.05])

    replay("analysis/final_share_all_ep200_f829c584_step_order_every/steps_03/REPLAY_acc_curve_along_epochs.pkl",
           figsize=(10, 6), fontsize=25, filepath="outputs/accuracy_last_batches_shuffled",
           xlabel="Mini-batches", ylabel="Accuracy", margins=[0, 0.05], markersize=7)
    replay("analysis/final_share_all_ep200_45766df6/steps_02/REPLAY_acc_curve_along_epochs.pkl",
           figsize=(10, 6), fontsize=25, filepath="outputs/accuracy_last_batches_ordered",
           xlabel="Mini-batches", ylabel="Accuracy", margins=[0, 0.05], markersize=7)
    replay("analysis/final_group_stable_partition_g16_565ce648/steps_09/REPLAY_acc_curve_along_epochs_group_each.pkl",
           figsize=(10, 6), fontsize=25, page=8, filepath="outputs/accuracy_g16_p8",
           xlabel="Mini-batches", ylabel="Accuracy", title="", margins=[0, 0.1], markersize=7, data_cutoff=4,
           color=[.17, 0., .36, .5], fmt=["--D", "-o", "-.s", "-p"],
           legend_loc="lower right", legend_labelspacing=0.1, legend_borderpad=0.2, legend_borderaxespad=0.2)
    replay("analysis/final_group_stable_partition_g1_508c3fba/steps_00/REPLAY_acc_curve_along_epochs_group_each_partition_stable_16.pkl",
           figsize=(10, 6), fontsize=25, page=8, filepath="outputs/accuracy_g16_p8_m1",
           xlabel="Mini-batches", ylabel="Accuracy", title="", margins=[0, 0.1], markersize=7, data_cutoff=4,
           color=[.17, 0., .36, .5], fmt=["--D", "-o", "-.s", "-p"],
           legend_loc="lower right", legend_labelspacing=0.1, legend_borderpad=0.2, legend_borderaxespad=0.2)

    replay("analysis/final_share_all_ep2000_7bdd44dc/steps_02/REPLAY_acc_curve_along_epochs.pkl",
           figsize=(15, 6), fontsize=20, filepath="outputs/accuracy_last_batches_ordered_ep2000",
           xlabel="Mini-batches", ylabel="Accuracy", margins=[0, 0.05])
    replay("analysis/final_partial_sharing_hp_bn1e-1_5fde412c/steps_inter/REPLAY_tau_curve_along_epochs.pkl",
           figsize=(10, 5), fontsize=25, filepath="outputs/gt_tau_bn1e-1",
           xlabel="Mini-batches", ylabel="GT-Tau", margins=[0, 0.05])
    replay("analysis/final_partial_sharing_hp_bn9e-1_d37fa71f/steps_inter/REPLAY_tau_curve_along_epochs.pkl",
           figsize=(10, 5), fontsize=25, filepath="outputs/gt_tau_bn9e-1",
           xlabel="Mini-batches", ylabel="GT-Tau", margins=[0, 0.05])
    replay("analysis/final_partial_sharing_hp_sgd_momentum0_dbf22dd3/steps_02/REPLAY_acc_curve_along_epochs.pkl",
           figsize=(15, 6), fontsize=20, filepath="outputs/accuracy_last_batches_ordered_sgd_momentum0",
           xlabel="Mini-batches", ylabel="Accuracy", margins=[0, 0.05])
