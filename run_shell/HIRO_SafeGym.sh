echo "`date '+%Y%m%d %H:%M:%S'`"
python -u clean_hiro/hiro_pytorch/main_sg.py --train --model_save_freq 1000 --num_episode 10000 --steps_per_epoch 30000 --env Safexp-PointGoal0-v0 --exp_name "Safexp-PointGoal0-v0-para0"

