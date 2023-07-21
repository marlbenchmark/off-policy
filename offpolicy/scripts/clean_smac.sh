ps -ef | grep StarCraftII | grep -v grep | awk '{print $2}' | xargs kill -9
