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
./cu_od --input tao.txt -c 575000 -d 3 -t 1.9 -w 10000 -s 500 -k 50
```
