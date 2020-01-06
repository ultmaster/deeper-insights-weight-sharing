@echo off

echo "============================================"

echo "Table 1, S-Tau"
python typesetting.py ^
    --metric S-Tau-Last ^
    --show-max-min ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_inter ^
    --folder analysis/final_share_all_ep200_fcd7bca5_step_order_one/epochs_inter ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_inter ^
    --folder analysis/final_group_stable_partition_g64_c42fdedd/epochs_inter

echo "Table 2, GT-Tau"
python typesetting.py ^
    --metric GT-Tau ^
    --metric Top-1-Rank ^
    --metric Top-3-Rank ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_inter ^
    --folder analysis/final_share_all_ep200_fcd7bca5_step_order_one/epochs_inter ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_inter ^
    --folder analysis/final_group_stable_partition_g64_c42fdedd/epochs_inter

echo "In text, S-Tau, different epochs"
python typesetting.py ^
    --metric Tau-as-Corr-Last-10 ^
    --metric GT-Tau-In-Window-Last-10 ^
    --metric Top-1-Rank-Last-10 ^
    --metric Top-3-Rank-Last-10 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_00 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_01 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_02 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_03 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_04 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_05 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_06 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_07 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_08 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_09 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_00 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_01 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_02 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_03 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_04 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_05 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_06 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_07 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_08 ^
    --folder analysis/final_share_all_ep200_45766df6/epochs_09

echo "In table, different epochs"
python typesetting.py ^
    --show-max-min ^
    --metric Tau-as-Corr-Last-10 ^
    --metric GT-Tau-In-Window-Last-10 ^
    --metric Top-1-Rank-Last-10 ^
    --metric Top-3-Rank-Last-10 ^
    --folder analysis/final_share_all_ep200_f829c584_step_order_every/epochs_09

echo "Group sharing data"
python typesetting.py ^
    --metric GT-Tau-In-Window-Last-64-Mean ^
    --metric GT-Tau-In-Window-Last-64-Std ^
    --folder analysis/final_group_stable_partition_g1_508c3fba/steps_inter ^
    --folder analysis/final_group_stable_partition_g2_ed5ff6d1/steps_inter ^
    --folder analysis/final_group_stable_partition_g4_d8918f36/steps_inter ^
    --folder analysis/final_group_stable_partition_g8_6480e0bd/steps_inter ^
    --folder analysis/final_group_stable_partition_g16_565ce648/steps_inter ^
    --folder analysis/final_group_stable_partition_g32_e7d2dea0/steps_inter ^
    --folder analysis/final_group_stable_partition_g64_c42fdedd/steps_inter

echo "Group similarity"
python typesetting.py ^
    --metric GT-Tau-In-Window-Last-64-Mean ^
    --metric GT-Tau-In-Window-Last-64-Std ^
    --folder analysis/final_group_sim_g2_3e397f16/steps_inter ^
    --folder analysis/final_group_sim_g4_6eedc0b0/steps_inter ^
    --folder analysis/final_group_sim_g8_fd508967/steps_inter ^
    --folder analysis/final_group_sim_g16_0cba3e44/steps_inter ^
    --folder analysis/final_group_sim_g32_a82f750a/steps_inter

echo "Prefix sharing data"
python typesetting.py ^
    --metric GT-Tau-In-Window-Last-64-Mean ^
    --metric GT-Tau-In-Window-Last-64-Std ^
    --metric Acc-Mean-In-Window-Last-64-Mean ^
    --metric Acc-Mean-In-Window-Last-64-Std ^
    --folder analysis/final_partial_sharing_prefix_k0_9fa72ebb/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_k1_5aa22e49/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_k2_d97518fe/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_k3_ac66d7f4/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_k4_d6bf931c/steps_inter

echo "Prefix sharing data EP200"
python typesetting.py ^
    --metric GT-Tau-In-Window-Last-64-Mean ^
    --metric GT-Tau-In-Window-Last-64-Std ^
    --metric Acc-Mean-In-Window-Last-64-Mean ^
    --metric Acc-Mean-In-Window-Last-64-Std ^
    --folder analysis/final_partial_sharing_prefix_ep200_k0_4bfe58ad/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_ep200_k1_da1dcc44/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_ep200_k2_ce6cc93d/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_ep200_k3_bf28dc5a/steps_inter ^
    --folder analysis/final_partial_sharing_prefix_ep200_k4_accaaacc/steps_inter
