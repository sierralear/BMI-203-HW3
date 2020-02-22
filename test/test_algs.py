import numpy as np
from smith_waterman import algs

def test_roc():
    return None

def test_smithwaterman():
    BLOSUM50, BLOSUM50_i = load_sim_matrix("BLOSUM50")
    

    #test that I get perfect alignment (alignment is the same as my sequence if sequences are the same)
    perfect_seq = "ARN"
    perfect_m = matrix(perfect_seq, perfect_seq, BLOSUM50, BLOSUM50_i, 2, 3)
    string_perfect, n_ = traceback(perfect_m, perfect_seq)
    assert string_perfect == perfect_seq

    #test that I get no alignment when my sequences are completely different
    bad_seq = "GZZ"
    bad_m = matrix(perfect_seq, bad_seq, BLOSUM50, BLOSUM50_i, 2, 3)
    string_bad, n_ = traceback(bad_m, bad_seq)
    assert string_bad == ""

def test_scoring():
    return None
