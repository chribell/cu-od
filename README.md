# CUDA outlier detection

This is a naive brute force approach for finding outliers using the GPU.

# Compile

```
mkdir release && cd release
cmake .. -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=61 # for Compute Capability 6.1
make -j 4
```

# Execution 

```
cd release
./cu_od --input ../tao.txt -c 575000 -d 3 -t 1.9 -w 10000 -s 500 -k 50 --weights ../weights.txt 
```

# Example (CUREX - DoS attack)
1. Compile the project
2. Place the compiled binary inside the curex folder
3. Modify the python binary path in run.sh 
4. `./run.sh 100 100 20 0.005`
