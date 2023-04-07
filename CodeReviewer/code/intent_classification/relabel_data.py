import os
import json
import numpy as np
from contextlib import ExitStack

INCLUDED_INTENTS = set([
    "EXCEPTIONHANDLING_LOG_ERROR",
    "CODE_REFACTOR_RENAME",
    "CODE_CLEANUP",
    "CODE_FORMAT_LINES",
    "UNIT_TESTING",
    "NULL_HANDLING",
    "STRING_OPERATIONS",
    "CONFIG",
    "STATIC_CONSTANT_IMPORT",
    "DATETIME_TIME_DURATION",
    "FILES_PATHS",
    #"CODE_REFACTOR_MOVE",
    "THREAD_LOCK_SLEEP",
    #"TYPES_OBJECT",
    "CHECK_VERIFY_LOGIC",
    #"CODE_DOCUMENTATION",
    #"TYPO_REMOVE_DELETE_NIT"
])

OTHER_CLASS = "OTHERS"

DOWN_SAMPLING_PROB = 0.25

def relabel_data(original_data_file_path, relabeled_data_file_path, down_sampling_others):
    """ Keep intent labels that are in INCLUDED_INTENTS, for other intents, label them as OTHER_CLASS. """
    total_count = 0
    non_others_count = 0

    with ExitStack() as stack:
        original_data = stack.enter_context(open(original_data_file_path, 'r', encoding="utf-8"))
        relabeled_data = stack.enter_context(open(relabeled_data_file_path, 'w', encoding="utf-8"))

        for line in original_data:
            dp_dict = json.loads(line)
            if dp_dict['cmt_label'] in INCLUDED_INTENTS:
                relabeled_data.write(line)
                non_others_count += 1
                total_count += 1
            else:
                dp_dict['cmt_label'] = OTHER_CLASS
                if down_sampling_others:
                    # down sample OTHERS class
                    if np.random.uniform() < DOWN_SAMPLING_PROB:
                        relabeled_data.write(json.dumps(dp_dict))
                        relabeled_data.write("\n")
                        total_count += 1
                else:
                    relabeled_data.write(json.dumps(dp_dict))
                    relabeled_data.write("\n")
                    total_count += 1

    return total_count, non_others_count


if __name__ == "__main__":
    original_data_folder = "/datadisk/shuo/CodeReview/code_review_intent/data_with_intent/"
    relabeled_data_folder = os.path.join("/datadisk/shuo/CodeReview/code_review_intent/", "data_with_intent_"+str(len(INCLUDED_INTENTS)+1)+"_classes_test")
    if not os.path.exists(relabeled_data_folder):
        os.mkdir(relabeled_data_folder)

    original_train_data = os.path.join(original_data_folder, "msg-train.jsonl")
    relabeled_train_data = os.path.join(relabeled_data_folder, "msg-train.jsonl")
    train_total_count, train_non_others_count = relabel_data(original_train_data, relabeled_train_data, down_sampling_others=True)

    original_val_data = os.path.join(original_data_folder, "msg-valid.jsonl")
    relabeled_val_data = os.path.join(relabeled_data_folder, "msg-valid.jsonl")
    val_total_count, val_non_others_count = relabel_data(original_val_data, relabeled_val_data, down_sampling_others=True)

    original_test_data = os.path.join(original_data_folder, "msg-test.jsonl")
    relabeled_test_data = os.path.join(relabeled_data_folder, "msg-test.jsonl")
    test_total_count, test_non_others_count = relabel_data(original_test_data, relabeled_test_data, down_sampling_others=False)

    print("{:.2%} data is covered by non-other labels in training and validation sets." \
        .format((train_non_others_count + val_non_others_count)*1.0/(train_total_count + val_total_count)))
    print("{:.2%} data is covered by non-other labels in test set.".format(test_non_others_count*1.0/test_total_count))