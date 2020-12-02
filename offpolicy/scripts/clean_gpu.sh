ps -A -ostat,ppid,pid,cmd | grep -E "train|multiprocessing" | grep -v grep | awk '$2==1 {print $3}' | xargs kill -9
