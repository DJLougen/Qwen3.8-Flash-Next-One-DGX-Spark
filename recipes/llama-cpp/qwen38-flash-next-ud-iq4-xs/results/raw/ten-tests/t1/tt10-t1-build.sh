#!/bin/bash
# tt10 T1 build: clone main tree build config into tt10-t1 clone (cmake re-run, NOT cp -a
# of a live build dir), then build llama-server only.
set -e
cd /home/djl/llama.cpp-tt10-t1
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /tmp/tt10-t1-cmake.log 2>&1
cmake --build build -j20 --target llama-server > /tmp/tt10-t1-build.log 2>&1
echo BUILD_DONE
ls -la build/bin/llama-server
