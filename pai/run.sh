set -x

sed -i 's/[# ]*Port.*/Port '$PAI_CONTAINER_SSH_PORT'/' /etc/ssh/sshd_config
service ssh restart
mkdir --parents /mnt/data
hdfs-mount <IP_ADDRESS> /mnt/data &
bash pai/tensorboard.sh &
mkdir --parents outputs checkpoints
python search.py "$@"
zip -r outputs.zip outputs checkpoints
mkdir --parents /mnt/data/v_yugzh/dps/${PAI_JOB_NAME}
cp outputs.zip /mnt/data/v_yugzh/dps/${PAI_JOB_NAME}/${PAI_CURRENT_TASK_ROLE_NAME}_${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX}.zip
