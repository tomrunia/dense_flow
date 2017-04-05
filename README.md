# Dense Optical Flow

This is a sample code for extracting dense flow field given a video. This repository is forked from https://github.com/wanglimin/dense_flow and includes visualization changes from https://github.com/vadimkantorov/mpegflow.
For more information see the original repository from wanglimin.

# Usage

`./denseFlow_gpu -f test.avi -o ./tmp/rgb_output_dir/ -i tmp/image -b 20 -t 1 -d 0 -s 1 -c 0`

Additional parameters :

`
-o tmp/rgbOutputDir  [path of where to store RGB images]
-c 1                 [applyColorMap, true/false]
`

# Troubleshooting

If you get the problem:

```
cannot find -lopencv_dep_cudart
```

add the following flag to `cmake`: `-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF`
See here: https://github.com/opencv/opencv/issues/6542
