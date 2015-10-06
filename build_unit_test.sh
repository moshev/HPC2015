#!/bin/bash

if [[ -z ${CXX} ]]; then
	CXX=clang++-3.7
fi

SRC=(
	main.cpp
	test_virtual/test_virtual.cpp
	test_virtual/handmade_virtual.cpp
	test_inline/test_inline_impl.cpp
	test_pointer_alias/test_pointer_alias.cpp
	test_pointer_alias/test_pointer_alias_impl.cpp
)
BUILD_CMD=${CXX}" "${SRC[@]}" -std=c++14 -I. -lpthread -O3 -fno-rtti -fstrict-aliasing -ffast-math -march=native -mtune=native"
echo $BUILD_CMD
$BUILD_CMD
