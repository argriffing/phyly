#!/bin/sh

autoreconf --install

args="--prefix=/home/username/.local \
CFLAGS='-march=native -O3 -g -Wall -Wextra'"

echo
echo "----------------------------------------------------------------"
echo "Initialized build system. For a common configuration please run:"
echo "----------------------------------------------------------------"
echo
echo "./configure $args"
echo
