#!/usr/bin/env bash
# Chain axis-2 behind the axis-1 sweep container; never run two training at once.
while docker ps --filter name=w24-sweep --format "{{.Names}}" | grep -q w24-sweep; do sleep 60; done
sleep 20
exec bash ~/w24_axis2_queue.sh
