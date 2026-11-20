#!/usr/bin/env bash
# Runs inside WSL. Remaining test flights, one house per call; a house that hangs is skipped after 25 min.
D=/mnt/c/Users/ninad/Desktop/EECS116/Final_project/data
one() { timeout -k 10 1500 bash $D/agile.sh "$1" 8 "$2" "ProcTHOR-Train-$3" "$4" || echo "=== SKIPPED house $3 (hang or error)" | tee -a $D/agile_log.txt; }
one test_med     51 122 med
one test_hard    53 140 hard
one test_hard    53 141 hard
one test_xfast   57 160 xfast
one test_xfast   57 161 xfast
one test_pubfast 59 1   hard
echo "=== gen_test done $(date)" | tee -a $D/agile_log.txt
