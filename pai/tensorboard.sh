set -x

export TENSORBOARD_PORT=PAI_PORT_LIST_main_${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX}_tensorboard
while true; do
timeout -sHUP 5m tensorboard --logdir outputs --port ${!TENSORBOARD_PORT};
done
