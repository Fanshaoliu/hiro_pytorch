echo "`date '+%Y%m%d %H:%M:%S'`"
echo "collect expert data of Safexp-PointGoal1-v0"
python -u clean_hiro/hiro_pytorch/main_sg.py --collect_data --env Safexp-PointGoal1-v0 --eval_epochs 10