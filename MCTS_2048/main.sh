#!/bin/bash

datestr=$(date +%Y%m%d%H%M)
mkdir -p ./test_output/$datestr
cp ./constants.py ./test_output/$datestr/
echo -n "Please input the comment:"
read comment
echo $comment > ./test_output/$datestr/comment.txt
python -u ./main.py |tee ./test_output/$datestr/result.log
echo -n "It'over! Is there thoughts?->"
read thoughts
echo $thoughts > ./test_output/$datestr/summary.txt
