# 1.HIRO Ant, baseline of HRL
#echo "`date '+%Y%m%d %H:%M:%S'`"
#nohup sh run_shell/HIRO_AntMaze.sh > shell_output/HIRO_AntMaze.txt 2>&1 &
#
#echo "`date '+%Y%m%d %H:%M:%S'`"
#nohup sh run_shell/HIRO_AntPush.sh > shell_output/HIRO_AntPush.txt 2>&1 &

# 2.HIRO SafeGym, baseline of HRL
echo "HIRO SafeGym, baseline of HRL"
echo "`date '+%Y%m%d %H:%M:%S'`"
nohup sh run_shell/HIRO_SafeGym.sh > shell_output/HIRO_SafeGym.txt 2>&1 &