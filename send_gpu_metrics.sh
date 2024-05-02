while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print "gpu.memory.usage:"$1"|g"}' | nc -w 1 -u graphite 8125
    sleep 5
done