ps -A -ostat,ppid,pid,cmd | grep -e'^[Zz]' | awk '{print $2}' | xargs kill -9
