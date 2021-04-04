from colorama import Fore
from colorama import Style

from sklearn.model_selection import StratifiedKFold
import Helpers
import DecisionTreeLearning
import time


def compare(X, Y):
    acc_rate = 0
    for i in range(len(X)):
        if X[i] == Y[i]:
            acc_rate += 1
    return round(acc_rate / len(X), 2)


def testing(examples, titles, n_splits):
    data = DecisionTreeLearning.delete_attribute(examples, DecisionTreeLearning.get_outcome_pos(examples))
    target = DecisionTreeLearning.get_outcome_list(examples)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True)
    gen_acc_test = []
    gen_acc_train = []

    for train_index, test_index in folds.split(data, target):
        pred_test = []
        pred_train = []
        true_test = [target[i] for i in test_index]
        true_train = [target[i] for i in train_index]
        #print(train_index, test_index)
        examples_train = examples[train_index]
        examples_test = data[test_index]

        fit_tree = DecisionTreeLearning.decision_tree_learning(examples_train, titles)

        ##TESTING ON THE TEST SET##
        for exs in examples_test:
            pred_test.append(fit_tree(titles, exs))
        acc_test = compare(pred_test, true_test)
        gen_acc_test.append(acc_test)
        print(f"Expected output on the test set: {pred_test}")
        print(f'True output: {true_test}')
        print(f"{Fore.CYAN}Accuracy on the test is: {acc_test * 100}%{Style.RESET_ALL}")

        ##TESTING ON THE TRAINING SET##
        examples_train_no_target = data[train_index]
        for exs in examples_train_no_target:
            pred_train.append(fit_tree(titles, exs))
        acc_train = compare(pred_train, true_train)
        gen_acc_train.append(acc_train)
        print(f"Expected output on the training set: {pred_train}")
        print(f'True output: {true_train}')
        print(f"{Fore.CYAN}Accuracy on the training set is: {acc_train * 100}%{Style.RESET_ALL}")
        print("  ")
        print("   ###########################################################################################   ")
        print("  ")

    avg_test = round(sum(gen_acc_test) / len(gen_acc_test), 3)
    avg_train = round(sum(gen_acc_train) / len(gen_acc_train), 3)


    print(f'FINAL: THE AVERAGE ACCURACY ON THE TEST SET IS {avg_test}')
    print(f'THE AVERAGE ACCURACY ON THE TRAINING SET IS {avg_train}')


if __name__ == '__main__':

    titles, examples = Helpers.read_from_csv_cat("tic_tac_toe.csv")

    #titles, examples = Helpers.read_from_csv_num("breast_cancer.csv")

    #titles, examples = Helpers.read_from_csv_mixed("heart_disease_cleveland.csv")

    start_time = time.time()
    testing(examples, titles, n_splits=10)
    end_time = time.time()
    print(f'TIME: {end_time - start_time}')
