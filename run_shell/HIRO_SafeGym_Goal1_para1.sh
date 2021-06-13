echo "`date '+%Y%m%d %H:%M:%S'`"
python -u clean_hiro/hiro_pytorch/main_sg.py --train --batch_size 64 --train_freq 20 --buffer_freq 20 --model_save_freq 1000 --num_episode 10000 --steps_per_epoch 30000 --env Safexp-PointGoal1-v0 --exp_name "Safexp-PointGoal1-v0-para1"
