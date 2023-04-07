trap '{ echo "Hey, you pressed Ctrl-C.  Time to quit." ; exit 1; }' INT

search_dir="/datadisk/shuo/CodeReview/code_review_intent/test_data_split_by_intent/"
for entry in "$search_dir"/*
do if [[ "$entry" == *-test.jsonl ]]; then
        bash test-msg.sh "$entry"
    fi
done