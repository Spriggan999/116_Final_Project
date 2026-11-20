#!/usr/bin/env bash
# Runs inside WSL. Training flights for the learned front end: 24 houses x 8 flights, 4 speed tiers.
# Houses 20-43 only (not 1 = public, not 120-161 = test, not 192-199 = held-out / dev).
# One house per call; a house that hangs is skipped after 25 min.
D=/mnt/c/Users/ninad/Desktop/EECS116/Final_project/data
one() { timeout -k 10 1500 bash $D/agile.sh "$1" 8 "$2" "ProcTHOR-Train-$3" "$4" || echo "=== SKIPPED house $3 (hang or error)" | tee -a $D/agile_log.txt; }
for h in 20 21 22 23 24 25; do one train_med   41 $h med;   done
for h in 26 27 28 29 30 31; do one train_hard  42 $h hard;  done
for h in 32 33 34 35 36 37; do one train_xfast 43 $h xfast; done
export WP=120
for h in 38 39 40 41 42 43; do one train_slow  44 $h "0.4 1.0 1.0 3.0 1.0 0.3 0.8"; done
echo "=== gen_train done $(date)" | tee -a $D/agile_log.txt
