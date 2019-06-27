#!/bin/bash

# Obtain the date and copy the constants.py
datestr=$(date +%Y%m%d%H%M)
mkdir -p ./test_output/$datestr
cp ./constants.py ./test_output/$datestr/
cp -r ./Model_Resnet/ ./test_output/$datestr/
mv ./test_output/$datestr/Model_Resnet ./test_output/$datestr/old_model


# Write the comment file.
echo -n "Please input the comment:"
read comment
echo $comment > ./test_output/$datestr/nohup_comment.txt

# Run the program and write the log file 
nohup python ./main.py > ./test_output/$datestr/result.log &

cp -r ./Model_Resnet/ ./test_output/$datestr/
mv ./test_output/$datestr/Model_Resnet ./test_output/$datestr/new_model
