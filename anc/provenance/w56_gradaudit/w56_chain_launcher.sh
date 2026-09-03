#!/usr/bin/env bash
# Start the W5.6 queue only after the W1.7 extra-seeds queue fully drains.
while pgrep -f w17_e1_extra_seeds.sh >/dev/null 2>&1; do sleep 60; done
sleep 30  # let the last paired report flush
exec bash ~/w56_gradaudit_queue.sh
