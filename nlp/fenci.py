# -*- coding: utf-8 -*-
dictionary = {"计算机语言学": 0.5, "课程": 0.5, "意思": 0.5}
s1 = "计算机语言学课程有意思"


def MaximumForwardMatching(s1):
    s2 = ""
    begin = 0
    end = len(s1)
    attributes = {'singleDictWord': 0, 'nonDictWord': 0, "numOfWords": 0}
    while begin != end:
        while end - begin > 1 and s1[begin: end] not in dictionary:
            end -= 1
        if end-begin == 1:
            if s1[begin: end] in dictionary:
                attributes['singleDictWord'] += 1
            else:
                attributes['nonDictWord'] += 1
        attributes['numOfWords'] += 1
        s2 = s2 + s1[begin: end] + "/"
        begin = end
        end = len(s1)
    print(s2)
    return s2, attributes

def MaximumBackwardMatching(s1):
    s2 = ""
    begin, end = 0, len(s1)
    attributes = {'singleDictWord': 0, 'nonDictWord': 0, "numOfWords": 0}
    while begin != end:
        while end - begin > 1 and s1[begin: end] not in dictionary:
            begin += 1
        if end-begin == 1:
            if s1[begin: end] in dictionary:
                attributes['singleDictWord'] += 1
            else:
                attributes['nonDictWord'] += 1
        attributes['numOfWords'] += 1
        s2 = s1[begin: end] + "/" + s2
        end = begin
        begin = 0
    print(s2)
    return s2, attributes


dictionary = dict(zip(["我们", "在", "在野", "生动", "野生动物园", "园", "玩"], [0.5]*7))
s1 = "我们在野生动物园玩"
f_s2, f_attributes = MaximumForwardMatching(s1)
b_s2, b_attributes = MaximumBackwardMatching(s1)
print(f_attributes, b_attributes)
f = 0
for attr in f_attributes:
    if f_attributes[attr] <= b_attributes[attr]:
        f += 1
if f >= 2:
    print("forward", f_s2)
else:
    print("backward", b_s2)

