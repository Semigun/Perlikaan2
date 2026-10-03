#!/bin/bash
# Run candidate-vs-incumbent head-to-heads for several parameter changes.
SNAP=research/snapshots/frozen.py
mkdir -p research/snapshots && cp perudo/bots/orakel.py $SNAP   # freeze the bot under test
for P in "$@"; do
  echo "### candidate $P vs default" 
  python3 research/h2h.py --a $SNAP --b $SNAP --pa "$P" --games 150 --chunks 8
done
