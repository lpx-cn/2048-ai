#!/bin/bash

# Obtain the date and copy the constants.py and the best_model.h5
datestr=$(date +%Y%m%d%H%M)
mkdir -p ./test_output/$datestr
cp ./constants.py ./test_output/$datestr/
cp -r ./Model_Resnet/ .test_output/$datestr/
mv .test_output/$datestr/Model_Resnet .test_output/$datestr/old_model

# Write the comment file.
echo -n "Please input the comment:"
read comment
echo $comment > ./test_output/$datestr/comment.txt

# Run the program and write the log file 
python -u ./main.py |tee ./test_output/$datestr/result.log

cp -r ./Model_Resnet/ .test_output/$datestr/
mv .test_output/$datestr/Model_Resnet .test_output/$datestr/new_model

# Write the summary file
echo -n "It'over! Is there thoughts?->"
read thoughts
echo $thoughts > ./test_output/$datestr/summary.txt

# Rename the result.log 
echo -n "Rename the result:"
read result_name
mv ./test_output/$datestr/result.log ./test_output/$datestr/$result_name.log
