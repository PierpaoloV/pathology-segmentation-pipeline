#!/bin/bash 

project_name="project_name"
config_path="/home/user/${project_name}/network_configuration.yaml"
data_path=$1
albumentation_path="/home/user/${project_name}/albumentations_pytorch.yaml"
logging_path="/home/user/${project_name}/logging"

#Insert your wandb token here
wandb login 

python3 /home/user/source/code/pytorch_exp_run.py \
--project_name=${project_name} \
--data_path=${data_path} \
--config_path=${config_path} \
--alb_config_path=${albumentation_path} \
--output_path=${logging_path}   