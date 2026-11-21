#!/usr/bin/env bash
# -D flags of this directory's CMakeLists.txt configuration (v9_adapted), for host builds
sed -n '/target_compile_definitions/,/)/p' "$(dirname "$0")/../CMakeLists.txt" | grep -v target_compile_definitions | tr -d ')' | sed 's/^ *//' | grep . | sed 's/^/-D/' | tr '\n' ' '
