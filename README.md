# Dense Optical Flow

This is a sample code for extracting dense flow field given a video. This repository is forked from https://github.com/wanglimin/dense_flow and includes visualization changes from https://github.com/vadimkantorov/mpegflow.
For more information see the original repository from wanglimin.

**Usage:**

`./denseFlow_gpu -f test.avi -o ./tmp/rgb_output_dir/ -i tmp/image -b 20 -t 1 -d 0 -s 1 -c 1`

Additional parameters compared to the original repository are:

```
-o tmp/rgbOutputDir     [path of where to store RGB images]
-c 1                    [applyColorMap, true/false]
```
