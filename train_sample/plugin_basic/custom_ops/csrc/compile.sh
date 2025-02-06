#!/bin/sh

hbdk4_path=$(python3 -c 'import hbdk4;print(hbdk4.__path__[-1])')
include_path=${hbdk4_path}/runtime/x86_64_unknown_linux_gnu/nash/include

echo Using include path: ${include_path}

g++ -I${include_path} -fPIC --std=c++17 matmul.cpp -static-libstdc++ -Wl,--exclude-libs=libstdc++.a --shared -o custom_matmul.so
