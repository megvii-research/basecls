#!/bin/bash
# Used to make a clean replication of model zoo inside doc folder

if [ $# -ne 2 ]; then
   echo ./sync_zoo.sh SRC DST
   exit 1
fi

for PTN in `cat ../.gitignore`; do
  EXC="${EXC:+$EXC }--exclude $PTN"
done

echo $EXC  # for debug

rsync -arv $1/ $2/ $EXC
