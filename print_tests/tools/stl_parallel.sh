#!/bin/sh
# 부품 STEP 여러 개를 STL 로 동시에 바꾼다(부품마다 따로 프로세스, step2stl.py 그대로).
# 사용: sh print_tests/tools/stl_parallel.sh <STEP 폴더> <STL 폴더> <이름 패턴>...
#   예: sh print_tests/tools/stl_parallel.sh out20/print_parts out20/print_parts_stl 'R_*' 'L_*'
# 결과 줄(열린 모서리 · 부피 차)은 <STL 폴더>/stl_parallel.log 에 모인다. 동시 개수는 JOBS(기본 16).
set -eu
src=$1; dst=$2; shift 2
cd "$(dirname "$0")/../.."
log="$dst/stl_parallel.log"
: > "$log"
for pat in "$@"; do ls "$src"/$pat.stp; done | sort -u | \
  xargs -P "${JOBS:-16}" -I{} sh -c 'n=$(basename {} .stp); LD_LIBRARY_PATH=sim/tmp/glu uv run -q --no-project --python 3.12 --with gmsh --with numpy python -I print_tests/tools/step2stl.py {} '"$dst"'/$n.stl 2>&1 | tail -1' >> "$log"
sort "$log"
echo "부품 $(wc -l < "$log")개, 열린 모서리 있는 것 $(grep -vc 'open_edges 0 ' "$log" || true)개"
