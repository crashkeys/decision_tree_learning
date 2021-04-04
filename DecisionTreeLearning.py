import math
import numpy
from itertools import groupby

import DecisionLeaf
import DecisionNode


def get_outcome_pos(examples_list):
    outcome_pos = len(examples_list[0]) - 1
    return outcome_pos


def get_outcome_list(examples_list):
    outcomes = []
    for example in examples_list:
        outcomes.append(example[get_outcome_pos(examples_list)])
    return outcomes


def get_all_different_values(examples_list, attribute):
    """All possible values for an attribute with no repetition"""
    possible_values = []
    for example in examples_list:
        if possible_values.count(example[attribute]) == 0:
            possible_values.append(example[attribute])
    return possible_values


def get_all_values(examples_list, attribute):
    """All answers for an attribute"""
    all_values = []
    for example in examples_list:
        all_values.append(example[attribute])
    return all_values


def outcomes_per_value(examples_list, attribute, value):
    """All the outcomes for a possible distinct value of an attribute"""
    outcomes_per_value = []
    for example in examples_list:
        if example[attribute] == value:
            outcomes_per_value.append(example[get_outcome_pos(examples_list)])
    return outcomes_per_value


def get_all_possible_outcomes(examples_list):
    return get_all_different_values(examples_list, get_outcome_pos(examples_list))


#### IMPORTANCE #####

def boolean_entropy(q):
    if q == 0 or q == 1:
        return 0
    else:
        return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


def entropy_goal(examples_list):
    all_outcomes = get_outcome_list(examples_list)
    possible_outcomes = get_all_possible_outcomes(examples_list)
    l = len(examples_list)
    count = 0
    for o in all_outcomes:
        if o == possible_outcomes[0]:
            count += 1
    b = boolean_entropy(count / l)
    return b


def get_remainder(examples_list, attribute):
    remainder = 0
    different_values = get_all_different_values(examples_list, attribute)
    for value in different_values:
        p = 0
        n = 0
        outcomes = outcomes_per_value(examples_list, attribute, value)
        possible_outcomes = get_all_possible_outcomes(examples_list)
        for o in outcomes:
            if o == possible_outcomes[0]:
                p += 1
            else:
                n += 1
        prob = (p + n) / len(examples_list)
        b = boolean_entropy(p / (p + n))
        remainder += prob * b
    return remainder


####### FOR CONTINUOUS VALUES ######

def is_numeric(examples_list, attribute):
    return isinstance(examples_list[0][attribute], int) or isinstance(examples_list[0][attribute], float)


def find_threshold(examples_list, attribute):
    """Return the array of all possible candidate thresholds"""
    values = get_all_values(examples_list, attribute)
    thresholds = []
    out_pos = get_outcome_pos(examples_list)
    labels = examples_list[:, out_pos]
    combined = [list(a) for a in zip(values, labels)]
    combined.sort(key=lambda test_list: test_list[0])
    values = [combined[i][0] for i in range(len(combined))]
    labels = [combined[i][1] for i in range(len(combined))]

    def compress_groups(vls):
        """Combines duplicate (values, labels) to single (value, label)."""
        val0, lab0 = next(vls[1])
        if all(lab == lab0 for val, lab in vls[1]):
            return val0, lab0
        return val0, -1 #case2_label

    vl_combined = [(v,l) for v,l in zip(values, labels)]
    vl_groups = groupby(vl_combined, lambda vc: vc[0])
    vl_groups = map(lambda vl_group: compress_groups(vl_group), vl_groups)

    prev_value, prev_label = next(vl_groups)

    for curr_value, curr_label in vl_groups:
        if prev_label == -1 or curr_label != prev_label:
            thresholds.append( (prev_value + curr_value)/2 )

        prev_value, prev_label = curr_value, curr_label

    return list(set(thresholds))


def split_example(examples_list, attribute):
    """Choose best threshold to keep"""
    entrs = []
    thresholds = find_threshold(examples_list, attribute)
    for th in thresholds:
        sections = []
        entropy = 0
        sec_1 = [exs for exs in examples_list if exs[attribute] <= th]
        sec_2 = [exs for exs in examples_list if exs[attribute] > th]
        sections.append(sec_1)
        sections.append(sec_2)
        for sec in sections:
            entropy += (len(sec)/len(examples_list)) * (entropy_goal(sec))
        entrs.append(entropy)
    if len(thresholds) == 0:
        return None, 1
    index_array = numpy.argsort(entrs)
    best_threshold = thresholds[index_array[0]]
    return best_threshold, entrs[index_array[0]]


#### CHOOSE BEST ATTRIBUTE ###


def information_gain(examples_list, attribute):
    b = entropy_goal(examples_list)
    if is_numeric(examples_list, attribute):
        th, remainder = split_example(examples_list, attribute)
    else:
        remainder = get_remainder(examples_list, attribute)
    return b - remainder


def choose_best_attribute(examples_list, titles):
    gains = []
    for i in range(len(titles)-1):
        gain = information_gain(examples_list, i)
        gains.append(gain)
    best_attribute = gains.index(max(gains))
    gains.sort()
    gains.reverse()
    print(f'NODE---------> Best attribute is {titles[best_attribute]} with a gain of {gains[0]}')
    return best_attribute, titles[best_attribute]


#### DELETE ####

def delete_attribute(examples_list, attribute):
    examples_list = numpy.delete(examples_list, attribute, axis=1)
    return examples_list


def delete_title(titles, attribute):
    titles = numpy.delete(titles, attribute)
    return titles


def delete_rows(examples_list, attribute, v, th):
    if is_numeric(examples_list, attribute):
        if v == f'<= {th}':
            examples_list = [
                exs for exs in examples_list if exs[attribute] <= th
            ]
        else:
            examples_list = [
                exs for exs in examples_list if exs[attribute] > th
            ]
    else: #for categoric attributes
        examples_list = [
            exs for exs in examples_list if exs[attribute] == v
        ]

    return examples_list


def uncertainty(examples_list):
    """If all examples left are the same but have different outputs"""
    unc = False
    examples = delete_attribute(examples_list, get_outcome_pos(examples_list))
    outputs = numpy.array(get_outcome_list(examples_list))
    if (examples == examples[0]).all() and (outputs != outputs[0]).any():
        unc = True
    return unc


def plurality_value(examples_list):
    """Selects the most common output in a set of examples"""
    all_outcomes = get_outcome_list(examples_list)
    possible_outcomes = get_all_possible_outcomes(examples_list)
    p = 0
    n = 0
    for o in all_outcomes:
        if o == possible_outcomes[0]:
            p += 1
        else:
            n += 1
    if p > n:
        return DecisionLeaf.DecisionLeaf(possible_outcomes[0])
    else:
        return DecisionLeaf.DecisionLeaf(possible_outcomes[1])


####### DECISION TREE LEARNING #############

def decision_tree_learning(examples_list, titles, parent_examples=[]):
    if len(examples_list) == 0 or uncertainty(examples_list):
        return plurality_value(parent_examples)
    elif entropy_goal(examples_list) == 0: #if all same classification
        classification = DecisionLeaf.DecisionLeaf(examples_list[0][get_outcome_pos(examples_list)])
        classification.display()
        return classification
    elif len(titles) == 0:
        return plurality_value(examples_list)
    else:
        best_attr, best_attr_name = choose_best_attribute(examples_list, titles)
        if is_numeric(examples_list, best_attr): #CONTINUOUS
            vals = []
            th, entr = split_example(examples_list, best_attr)
            vals.append(f'<= {th}')
            vals.append(f'> {th}')
        else:  #CATEGORICAL
            th = None
            vals = get_all_different_values(examples_list, best_attr)
        tree = DecisionNode.DecisionNode(best_attr, best_attr_name)
        for v in vals:
            new_examples_list = delete_rows(examples_list, best_attr, v, th)
            new_examples_list = delete_attribute(new_examples_list, best_attr)
            new_titles = delete_title(titles, best_attr)
            tree.display(best_attr_name, v, th)
            subtree = decision_tree_learning(new_examples_list, new_titles, examples_list)
            tree.add(v, subtree)
        return tree














