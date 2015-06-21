import unit
import numpy as np

mapping = "abcdefghijklmnopqrstuvwxyz ?.,#@"

def serialize_text(string):
    r = []
    string = string.lower()
    n = len(mapping)
    for x in string:
        q = [0] * (n)
        if x in mapping:
            s = mapping.index(x)
        else:
            s= n+1
        q[s] = 1.0
        r.append(q)
    return r

def set_to_character(set):
    q = set.argsort()[0]
    try:
        g = mapping[q]
    except:
        print q
        print set
        print set.argsort()
    return g

def outputs_to_text(outputs):
    m = ""
    for x in outputs:
        m = m + set_to_character(x)
    return m

def feed_text(string, unit):
    d = serialize_text(string)
    r = unit.run(d)
    g = outputs_to_text(r)
    return score_outputs(string, g), g, r

def score_outputs(input_string, output_string): #EACH CHAR INPUTTED LEADS TO PREDICTED NEXT CHAR, so counts wont match
    score = 0
    for i, x in enumerate(output_string[0:len(output_string)-1]): #omit last character outputted
        m = input_string[i+1]
        if m == x:
            score = score + 1
    return float(score)/float(len(input_string)-1)

def character_unit():
    input_n = len(mapping)
    output_n = input_n
    hidden_width = 100
    hidden_depth = 3
    return unit.Unit(input_n, output_n, hidden_width, hidden_depth)

def test():
    unit = character_unit()
    t =    "the man walked the dog to the park"
    s, message, raw_outputs = feed_text(t, unit)
    print unit.backpropragate(raw_outputs, serialize_text(t[0:len(t)-1]))
    return s, message

def try_many():
    r=True
    best = 0
    while r:
        s, m = test()
        if s>best:
            best =s
            print best
            print m
        if s>0.7:
            r=False
