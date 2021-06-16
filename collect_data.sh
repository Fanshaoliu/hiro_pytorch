echo "`date '+%Y%m%d %H:%M:%S'`"
nohup sh run_shell/collect_data.sh > shell_output/collect_data.txt 2>&1 &