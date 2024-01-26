current_time=`date +%Y%m%d-%H%M%S`
model_path="generate_label/fakenet/model/${current_time}_"$1"/"
config_path="generate_label/fakenet/config/"$1".yaml"

python generate_label/fakenet/train.py --output_folder_name ${model_path} --config ${config_path}
