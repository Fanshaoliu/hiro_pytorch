# 1.HIRO Ant, baseline of HRL
#echo "HIRO AntMaze, baseline of HRL"
#echo "`date '+%Y%m%d %H:%M:%S'`"
#nohup sh run_shell/HIRO_AntMaze.sh > shell_output/HIRO_AntMaze.txt 2>&1 &
#
#echo "HIRO AntPush, baseline of HRL"
#echo "`date '+%Y%m%d %H:%M:%S'`"
#nohup sh run_shell/HIRO_AntPush.sh > shell_output/HIRO_AntPush.txt 2>&1 &

# 2.HIRO SafeGym Goal0, baseline of HRL
#echo "HIRO SafeGym Goal0, baseline of HRL"
#echo "`date '+%Y%m%d %H:%M:%S'`"
#nohup sh run_shell/HIRO_SafeGym_Goal0.sh > shell_output/HIRO_SafeGym.txt 2>&1 &

# 3.HIRO SafeGym Goal1, baseline of HRL
echo "HIRO SafeGym Goal1, baseline of HRL"
echo "`date '+%Y%m%d %H:%M:%S'`"
nohup sh run_shell/HIRO_SafeGym_Goal1_para1.sh > shell_output/HIRO_SafeGym_Goal1_para1.txt 2>&1 &
nohup sh run_shell/HIRO_SafeGym_Goal1_para2.sh > shell_output/HIRO_SafeGym_Goal1_para2.txt 2>&1 &