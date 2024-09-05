import pytest
import sequencelib as sq
import random
import re
import numpy as np
import collections
import itertools
import math
from collections import Counter
from math import log


# Note: I could use fixtures to make the testing code much shorter (sharing a few instances
# of DNA sequences for many test functions etc instead of setting up new test data for each
# function), but I find it simpler to understand the tests this way, with repetitiveness

# Note 2: I am here explicitly testing concrete subclasses instead of base classes (that
# are not meant to be instantiated). This also causes some duplication (for some methods
# they are tested in multiple derived classes). This is closer to testing actual use
# and follows principle to test behaviour instead of implementation details


###################################################################################################
###################################################################################################

# Tests for loose functions

###################################################################################################
###################################################################################################

class Test_find_seqtype:

    seqlen = 75

    def test_DNA_noambig(self):
        DNA_string = "".join(random.choices(list(sq.Const.DNA), k=self.seqlen))
        assert sq.find_seqtype(DNA_string) == "DNA"
        assert sq.find_seqtype(list(DNA_string)) == "DNA"
        assert sq.find_seqtype(set(DNA_string)) == "DNA"

    def test_DNA_ambig(self):
        DNA_string_ambig = "".join(random.choices(list(sq.Const.DNA_maxambig), k=self.seqlen))
        assert sq.find_seqtype(DNA_string_ambig) == "DNA"
        assert sq.find_seqtype(list(DNA_string_ambig)) == "DNA"
        assert sq.find_seqtype(set(DNA_string_ambig)) == "DNA"

    def test_Protein_noambig(self):
        Protein_string = "".join(random.choices(list(sq.Const.Protein), k=self.seqlen))
        assert sq.find_seqtype(Protein_string) == "protein"
        assert sq.find_seqtype(list(Protein_string)) == "protein"
        assert sq.find_seqtype(set(Protein_string)) == "protein"

    def test_Protein_ambig(self):
        Protein_string_ambig = "".join(random.choices(list(sq.Const.Protein_maxambig), k=self.seqlen))
        assert sq.find_seqtype(Protein_string_ambig) == "protein"
        assert sq.find_seqtype(list(Protein_string_ambig)) == "protein"
        assert sq.find_seqtype(set(Protein_string_ambig)) == "protein"

    def test_ASCII(self):
        ASCII_string = "".join(random.choices(list(sq.Const.ASCII), k=self.seqlen))
        assert sq.find_seqtype(ASCII_string) == "ASCII"
        assert sq.find_seqtype(list(ASCII_string)) == "ASCII"
        assert sq.find_seqtype(set(ASCII_string)) == "ASCII"

    def test_Standard(self):
        Standard_string = "".join(random.choices(list(sq.Const.Standard), k=self.seqlen))
        assert sq.find_seqtype(Standard_string) == "standard"
        assert sq.find_seqtype(list(Standard_string)) == "standard"
        assert sq.find_seqtype(set(Standard_string)) == "standard"

    def test_unrecognized_raises(self):
        ASCII_string = "".join(random.choices(list(sq.Const.ASCII), k=self.seqlen))
        unknown = ASCII_string + "ØÆÅ=)&%#"
        with pytest.raises(sq.SeqError):
            sq.find_seqtype(unknown)

###################################################################################################

class Test_seqtype_attributes:

    def test_DNA(self):
        assert (sq.seqtype_attributes("DNA")
                == (set("ACGTURYMKWSBDHVN"), set("URYMKWSBDHVN")))

    def test_Protein(self):
        assert (sq.seqtype_attributes("protein")
                == (set("ACDEFGHIKLMNPQRSTVWYBZX"), set("BXZ")))

    def test_ASCII(self):
        assert (sq.seqtype_attributes("ASCII")
                == (set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,._"), set()))

    def test_Standard(self):
        assert (sq.seqtype_attributes("standard")
                == (set("0123456789"), set()))

    def test_unknown_raises(self):
        with pytest.raises(sq.SeqError, match = r"Unknown sequence type: normannisk"):
            sq.seqtype_attributes("normannisk")

###################################################################################################

class Test_indices:

    def test_singlesubstring(self):
        inputstring = "AAAAAAAAAA AAAAAAAA AAAAAAA Here AAAAAAAA AAAAAA"
        assert sq.indices(inputstring, "Here") == set([28])

    def test_triplesubstring(self):
        inputstring = "AAAAAAAAAA Here AAAAAAAA AAAAAAA Here AAAAAAAA AAAHereAAA"
        assert sq.indices(inputstring, "Here") == set([11,33,50])

    def test_overlapping(self):
        inputstring = "AAAAAAAAAA hehehehe AAAAAAA hehe AAAAA"
        assert sq.indices(inputstring, "hehe") == set([11,13,15,28])

###################################################################################################

class Test_remove_comments:

    def test_unnested_1chardelim(self):
        input = "This sentence [which is an example] contains one comment"
        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == "This sentence  contains one comment")

    def test_unnested__1chardelim_multiline(self):
        input = """This sentence [which is an example of a string with
                a multiline un-nested comment] contains one comment"""
        expexted_output = """This sentence  contains one comment"""
        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == expexted_output)

    def test_nested(self):
        input = "This sentence [which is an example [or is it?]] contains nested comments"
        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == "This sentence  contains nested comments")

    def test_nested__1chardelim_multiline(self):
        input = """This sentence [which is also an example] is far more complicated.
        [or is it [and here 'it' refers to the sentence]]. It contains nested
        comments [like[this]] and newlines. It also contains nested comments
        spread over multiple lines like this: [I am not sure this type of comment
        will ever appear in actual sequences [it might in trees though]]. The end"""

        expexted_output = """This sentence  is far more complicated.
        . It contains nested
        comments  and newlines. It also contains nested comments
        spread over multiple lines like this: . The end"""

        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == expexted_output)

    def test_unnested_multichardelim(self):
        input = "This sentence <B>which is an example<E> contains one comment"
        assert (sq.remove_comments(input, leftdelim="<B>", rightdelim="<E>")
                == "This sentence  contains one comment")

    def test_nested__multichardelim_multiline(self):
        input = """This sentence <com>which is also an example</com> is far more complicated.
        <com>or is it <com>and here 'it' refers to the sentence</com></com>. It contains nested
        comments <com>like<com>this</com></com> and newlines. It also contains nested comments
        spread over multiple lines like this: <com>I am not sure this type of comment
        will ever appear in actual sequences <com>it might in trees though</com></com>. The end"""

        expexted_output = """This sentence  is far more complicated.
        . It contains nested
        comments  and newlines. It also contains nested comments
        spread over multiple lines like this: . The end"""

        assert (sq.remove_comments(input, leftdelim="<com>", rightdelim="</com>")
                == expexted_output)

###################################################################################################

class Test_make_sparseencoder:

    def test_DNA_encoder(self):
        DNAencoder = sq.make_sparseencoder("ACGT")
        input = "AACGTX"
        output = DNAencoder(input)
        expected_output = np.array([
                                    1,0,0,0,
                                    1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0,
                                    0,0,0,1,
                                    0,0,0,0
                                    ])

        # pytest note: assert a == b does not work for numpy arrays:
        # https://github.com/pytest-dev/pytest/issues/5347
        # Use numpy's own testing setup instead: np.testing.assert_array_equal(a,b)
        #assert output.dtype == expected_output.dtype
        np.testing.assert_array_equal(output, expected_output)

###################################################################################################
###################################################################################################

# Tests for class DNA_sequence

###################################################################################################

class Test_init_DNA:

    seqlen = 180

    def test_attribute_assignment(self):
        name = "seq1"
        seq = "".join(random.choices("acgt", k=self.seqlen))
        annot = "".join(random.choices("ICP", k=self.seqlen))
        comments = "This sequence is randomly generated"
        dnaseq = sq.DNA_sequence(name=name, seq=seq, annotation=annot, comments=comments)
        assert dnaseq.name == name
        assert dnaseq.seq == seq.upper()
        assert dnaseq.comments == comments
        assert dnaseq.annotation == annot

    def test_degapping(self):
        seq = "aaaaa-----ccccc-----ggggg-----ttttt"
        dnaseq = sq.DNA_sequence(name="seq1", seq=seq, degap=True)
        assert dnaseq.seq == seq.upper().replace("-","")

    def test_check_alphabet_raise(self):
        seq = "".join(random.choices("acgtæøå", k=self.seqlen))
        with pytest.raises(sq.SeqError, match = r"Unknown symbols in sequence s1: .*"):
            dnaseq = sq.DNA_sequence(name="s1", seq=seq, check_alphabet=True)

    def test_check_alphabet_not_raise(self):
        # Implicitly tests for not raising: function returns None, which counts as passing
        seq = "".join(random.choices(list(sq.Const.DNA_maxambig), k=self.seqlen))
        dnaseq = sq.DNA_sequence(name="s1", seq=seq, check_alphabet=True)

###################################################################################################

class Test_eq_DNA:

    seqlen = 180

    def test_identical_withgaps(self):
        seq1 = "".join(random.choices("acgtn-", k=self.seqlen))
        dnaseq1 = sq.DNA_sequence("s1", seq1)
        dnaseq2 = sq.DNA_sequence("s2", seq1)
        assert dnaseq1 == dnaseq2

    def test_different(self):
        seq1 = "".join(random.choices("acgt", k=self.seqlen))
        seq2 = seq1[1:] + "n" # last seqlen-1 chars + one "n"
        dnaseq1 = sq.DNA_sequence("s1", seq1)
        dnaseq2 = sq.DNA_sequence("s2", seq2)
        assert dnaseq1 != dnaseq2

###################################################################################################

class Test_len_DNA:

    def test_5_lengths(self):
        for i in range(5):
            seqlen = random.randint(50, 350)
            seq1 = "".join(random.choices("acgtn-", k=seqlen))
            dnaseq = sq.DNA_sequence("s1", seq1)
            assert len(dnaseq) == seqlen

###################################################################################################

class Test_getitem_DNA:

    seqlen = 180

    def test_indexing(self):
        seq = "".join(random.choices("acgtn-", k=self.seqlen))
        dnaseq = sq.DNA_sequence("s1", seq)
        for i in random.choices(range(self.seqlen), k=10):
            assert dnaseq[i] == seq[i].upper()

    def test_slicing(self):
        seq = "".join(random.choices("acgtn-", k=self.seqlen))
        dnaseq = sq.DNA_sequence("s1", seq)
        for i in random.choices(range(self.seqlen-10), k=10):
            assert dnaseq[i:(i+8)] == seq[i:(i+8)].upper()

###################################################################################################

class Test_setitem_DNA:

    seqlen = 180

    def test_setsingle(self):
        for i in random.choices(range(self.seqlen), k=10):
            seqlist = random.choices("acg", k=self.seqlen)  # Note: no T
            seq = "".join(seqlist)
            dnaseq = sq.DNA_sequence("s1", seq)
            dnaseq[i] = "t"
            assert dnaseq[i] == "T"

###################################################################################################

class Test_str_DNA:

    seqlen = 180

    def test_fastastring(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        dnaseq = sq.DNA_sequence("s1", seq)
        output = "{}".format(dnaseq)
        expected_output = (
                            ">s1\n"
                            + "{}\n".format(seq[:60])
                            + "{}\n".format(seq[60:120])
                            + "{}".format(seq[120:180])
                        )
        assert output == expected_output

###################################################################################################

class Test_copy_DNA:

    seqlen = 180

    def test_seq_annot_comments(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        annot = "".join(random.choices("IPC", k=self.seqlen))
        comments = "This sequence will be copied"
        name = "origseq"
        dnaseq = sq.DNA_sequence(name, seq, annot, comments)
        dnaseq_copy = dnaseq.copy_seqobject()
        assert dnaseq == dnaseq_copy
        assert dnaseq.seq == dnaseq_copy.seq
        assert dnaseq.name == dnaseq_copy.name
        assert dnaseq.annotation == dnaseq_copy.annotation
        assert dnaseq.comments == dnaseq_copy.comments

###################################################################################################

class Test_rename_DNA:

    def test_changename(self):
        seq = "".join(random.choices("ACGTN-", k=50))
        dnaseq = sq.DNA_sequence("s1", seq)
        dnaseq.rename("newseqname")
        assert dnaseq.name == "newseqname"

###################################################################################################

class Test_subseq_DNA:

    def test_seq_annot_slice(self):
        seq = "".join(random.choices("ACGTN-", k=50))
        annot = "".join(random.choices("IPC", k=50))
        name = "mainseq"
        dnaseq = sq.DNA_sequence(name, seq, annot)
        subseq = dnaseq.subseq(start=10, stop=20, slicesyntax=True, rename=True)
        assert subseq.name == name + "_10_20"
        assert subseq.seq == seq[10:20]
        assert subseq.annotation == annot[10:20]
        assert subseq.seqtype == "DNA"

    def test_seq_notslice(self):
        seq = "AAAAACCCCCGGGGGTTTTT"
        name = "mainseq"
        dnaseq = sq.DNA_sequence(name, seq)
        subseq = dnaseq.subseq(start=6, stop=10, slicesyntax=False, rename=True)
        assert subseq.seq == "CCCCC"
        assert len(subseq.seq) == 5

    def test_toolong_subseq(self):
        seq = "AAAAACCCCC"
        name = "mainseq"
        dnaseq = sq.DNA_sequence(name, seq)
        exp_error_msg = re.escape("Requested subsequence (5 to 15) exceeds sequence length (10)")
        with pytest.raises(sq.SeqError, match = exp_error_msg):
             subseq = dnaseq.subseq(start=5, stop=15, slicesyntax=True, rename=True)

###################################################################################################

class Test_subseqpos_DNA:

    seqlen = 50

    def test_seq_annot_pos(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        annot = "".join(random.choices("IPC", k=self.seqlen))
        name = "mainseq"
        dnaseq = sq.DNA_sequence(name, seq, annot)
        poslist = random.choices(range(self.seqlen), k=10)
        subseqpos = dnaseq.subseqpos(poslist, namesuffix="_selected")
        assert subseqpos.name == name + "_selected"
        assert subseqpos.seq == "".join([seq[i] for i in poslist])
        assert subseqpos.annotation == "".join([annot[i] for i in poslist])
        assert subseqpos.seqtype == "DNA"

###################################################################################################

class Test_appendseq_DNA:

    seqlen = 180

    def test_seqs_annots_comments(self):
        seq1 = "".join(random.choices("ACGTN-", k=self.seqlen))
        seq2 = "".join(random.choices("ACGTN-", k=self.seqlen))
        name1 = "s1"
        name2 = "s2"
        annot1 = "".join(random.choices("IPC", k=self.seqlen))
        annot2 = "".join(random.choices("IPC", k=self.seqlen))
        com1 = "First gene"
        com2 = "Second gene"
        dnaseq1 = sq.DNA_sequence(name1, seq1, annot1, com1)
        dnaseq2 = sq.DNA_sequence(name2, seq2, annot2, com2)
        dnaseq3 = dnaseq1.appendseq(dnaseq2)
        assert dnaseq3.name == name1
        assert dnaseq3.seq == seq1 + seq2
        assert dnaseq3.annotation == annot1 + annot2
        assert dnaseq3.comments == com1 + " " + com2

###################################################################################################

class Test_prependseq_DNA:

    seqlen = 180

    def test_seqs_annots_comments(self):
        seq1 = "".join(random.choices("ACGTN-", k=self.seqlen))
        seq2 = "".join(random.choices("ACGTN-", k=self.seqlen))
        name1 = "s1"
        name2 = "s2"
        annot1 = "".join(random.choices("IPC", k=self.seqlen))
        annot2 = "".join(random.choices("IPC", k=self.seqlen))
        com1 = "First gene"
        com2 = "Second gene"
        dnaseq1 = sq.DNA_sequence(name1, seq1, annot1, com1)
        dnaseq2 = sq.DNA_sequence(name2, seq2, annot2, com2)
        dnaseq3 = dnaseq1.prependseq(dnaseq2)
        assert dnaseq3.name == name1
        assert dnaseq3.seq == seq2 + seq1
        assert dnaseq3.annotation == annot2 + annot1
        assert dnaseq3.comments == com2 + " " + com1

###################################################################################################

class Test_windows_DNA:

    # Python note: should add logic to original method (and tests here) for annotation and comments

    seqlen = 120

    def test_nooverhang_step1(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        name = "s1"
        dnaseq = sq.DNA_sequence(name, seq)
        wsize = 34
        window_iterator = dnaseq.windows(wsize=wsize, rename=True)
        windowlist = list(window_iterator)
        assert len(windowlist) == self.seqlen - wsize + 1
        for i, windowseq in enumerate(windowlist):
            assert windowseq.seqtype == "DNA"
            start = i
            stop = start + wsize
            assert windowseq.seq == seq[start:stop]

    def test_nooverhang_step5(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        name = "s1"
        dnaseq = sq.DNA_sequence(name, seq)
        wsize = 27
        stepsize = 7
        window_iterator = dnaseq.windows(wsize=wsize, stepsize=stepsize, rename=True)
        windowlist = list(window_iterator)
        assert len(windowlist) == (self.seqlen - 1 + stepsize - wsize) // stepsize
        for i, windowseq in enumerate(windowlist):
            assert windowseq.seqtype == "DNA"
            start = i * stepsize
            stop = start + wsize
            assert windowseq.seq == seq[start:stop]

    def test_loverhang_step1(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        name = "s1"
        dnaseq = sq.DNA_sequence(name, seq)
        wsize = 18
        l_overhang = 9
        window_iterator = dnaseq.windows(wsize=wsize, l_overhang=l_overhang, rename=True)
        windowlist = list(window_iterator)
        assert len(windowlist) == self.seqlen + l_overhang - wsize + 1
        for i, windowseq in enumerate(windowlist):
            assert windowseq.seqtype == "DNA"
            start = i - l_overhang
            stop = start + wsize
            if start >= 0:
                assert windowseq.seq == seq[start:stop]
            else:
                assert windowseq.seq[-stop:] == seq[:stop]
                assert windowseq.seq[:-stop] == "X" * (wsize - stop)

    def test_roverhang_step1(self):
        pass

###################################################################################################

class Test_remgaps_DNA:

    def test_remgaps(self):
        dnaseq = sq.DNA_sequence("s1", "AAAAA--CCCCC--GGGGG")
        dnaseq.remgaps()
        assert dnaseq.seq == "AAAAACCCCCGGGGG"

###################################################################################################

class Test_shuffle_DNA:

    seqlen = 120

    def test_composition_type(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        name = "s1"
        dnaseq1 = sq.DNA_sequence(name, seq)
        dnaseq2 = dnaseq1.shuffle()
        assert dnaseq2.seqtype == "DNA"
        assert dnaseq1.seq != dnaseq2.seq
        assert collections.Counter(dnaseq1.seq) == collections.Counter(dnaseq2.seq)

###################################################################################################

class Test_indexfilter_DNA:

    seqlen = 120

    def test_composition_type(self):
        seq = "".join(random.choices("ACGTN-", k=self.seqlen))
        name = "s1"
        dnaseq1 = sq.DNA_sequence(name, seq)

###################################################################################################

class Test_seqdiff_DNA:

    seqlen = 150

    def test_twoseqs_zeroindex(self):
        seq = "".join(random.choices("ACG", k=self.seqlen)) # Note: No T letters
        dnaseq1 = sq.DNA_sequence("s1", seq)
        dnaseq2 = dnaseq1.copy_seqobject()
        mutpos = random.choices(range(len(seq)), k=20)
        for i in mutpos:
            dnaseq2[i] = "T"
        seqdifflist = dnaseq1.seqdiff(dnaseq2)
        for pos,nuc1,nuc2 in seqdifflist:
            assert pos in mutpos
            assert dnaseq1[pos] == nuc1
            assert dnaseq2[pos] == nuc2
            assert nuc2 == "T"
        allpos_inresults = [i for i,n1,n2 in seqdifflist]
        assert set(allpos_inresults) == set(mutpos)

    def test_twoseqs_notzeroindex(self):
        seq = "".join(random.choices("ACG", k=self.seqlen)) # Note: No T letters
        dnaseq1 = sq.DNA_sequence("s1", seq)
        dnaseq2 = dnaseq1.copy_seqobject()
        mutpos = random.choices(range(1,len(seq)+1), k=20)
        for i in mutpos:
            dnaseq2[i-1] = "T"
        seqdifflist = dnaseq1.seqdiff(dnaseq2, zeroindex=False)
        for pos,nuc1,nuc2 in seqdifflist:
            assert pos in mutpos
            assert dnaseq1[pos-1] == nuc1
            assert dnaseq2[pos-1] == nuc2
            assert nuc2 == "T"
        allpos_inresults = [i for i,n1,n2 in seqdifflist]
        assert set(allpos_inresults) == set(mutpos)

###################################################################################################

class Test_hamming_DNA:

    seqlen = 150

    def test_10_random_pairs(self):
        for i in range(10):
            seq = "".join(random.choices("ACG-", k=self.seqlen)) # Note: No T letters
            dnaseq1 = sq.DNA_sequence("s1", seq)
            dnaseq2 = dnaseq1.copy_seqobject()
            nmut = random.randint(1,self.seqlen)
            mutpos = random.sample(range(len(seq)), k=nmut)      # No replacement
            for j in mutpos:
                dnaseq2[j] = "T"
            hammingdist = dnaseq1.hamming(dnaseq2)
            assert hammingdist == nmut

###################################################################################################

class Test_hamming_ignoregaps_DNA:

    seqlen = 150

    def test_10_random_pairs(self):
        for i in range(10):
            seq = "".join(random.choices("ACG-", k=self.seqlen)) # Note: No T letters
            dnaseq1 = sq.DNA_sequence("s1", seq)
            dnaseq2 = dnaseq1.copy_seqobject()
            nmut = random.randint(1,self.seqlen)
            mutpos = random.sample(range(len(seq)), k=nmut)
            ngaps = 0
            for j in mutpos:
                if dnaseq1[j] == "-":
                    ngaps += 1
                dnaseq2[j] = "T"
            hammingdist = dnaseq1.hamming_ignoregaps(dnaseq2)
            assert hammingdist == nmut - ngaps

###################################################################################################

class Test_pdist_DNA:

    seqlen = 150

    def test_10_random_pairs(self):
        for i in range(10):
            seq = "".join(random.choices("ACG-", k=self.seqlen)) # Note: No T letters
            dnaseq1 = sq.DNA_sequence("s1", seq)
            dnaseq2 = dnaseq1.copy_seqobject()
            nmut = random.randint(1,self.seqlen)
            mutpos = random.sample(range(len(seq)), k=nmut)     # No replacement
            for j in mutpos:
                dnaseq2[j] = "T"
            pdist = dnaseq1.pdist(dnaseq2)
            assert pdist == nmut / self.seqlen


###################################################################################################

class Test_pdist_ignoregaps_DNA:

    seqlen = 150

    def test_10_random_pairs(self):
        for i in range(10):
            seq = "".join(random.choices("ACG-", k=self.seqlen)) # Note: No T letters
            dnaseq1 = sq.DNA_sequence("s1", seq)
            dnaseq2 = dnaseq1.copy_seqobject()
            nmut = random.randint(1,self.seqlen)
            mutpos = random.sample(range(len(seq)), k=nmut)
            ngaps = 0
            for j in mutpos:
                if dnaseq1[j] == "-":
                    ngaps += 1
                dnaseq2[j] = "T"
            pdist = dnaseq1.pdist_ignoregaps(dnaseq2)
            assert pdist == (nmut - ngaps) / (self.seqlen - ngaps)

###################################################################################################

class Test_residuecounts_DNA:

    maxnuc = 100

    def test_oneseq(self):
        nA,nC, nG, nT = random.choices(range(self.maxnuc),k=4)
        seq = "A"*nA + "C"*nC + "G"*nG + "T"*nT

        dnaseq = sq.DNA_sequence("s1", seq)
        rescounts = dnaseq.residuecounts()
        assert rescounts["A"] == nA
        assert rescounts["C"] == nC
        assert rescounts["G"] == nG
        assert rescounts["T"] == nT

###################################################################################################

class Test_composition_DNA:

    maxnuc = 100

    def test_oneseq_countgaps(self):
        nA,nC, nG, nT, ngap = random.choices(range(1,self.maxnuc),k=5)
        seq = "A"*nA + "C"*nC + "G"*nG + "T"*nT + "-"*ngap
        seqlen = len(seq)
        seq = "".join(random.sample(seq, seqlen)) #Shuffle
        dnaseq = sq.DNA_sequence("s1", seq)
        comp = dnaseq.composition(ignoregaps=False)
        assert comp["A"] == (nA, nA/seqlen)
        assert comp["C"] == (nC, nC/seqlen)
        assert comp["G"] == (nG, nG/seqlen)
        assert comp["T"] == (nT, nT/seqlen)
        assert comp["-"] == (ngap, ngap/seqlen)

    def test_oneseq_ignoregaps(self):
        nA,nC, nG, nT, ngap = random.choices(range(1,self.maxnuc),k=5)
        seq = "A"*nA + "C"*nC + "G"*nG + "T"*nT + "-"*ngap
        seqlen = len(seq)
        seqlen_nogaps = seqlen - ngap
        seq = "".join(random.sample(seq, seqlen)) #Shuffle
        dnaseq = sq.DNA_sequence("s1", seq)
        comp = dnaseq.composition(ignoregaps=True)
        assert comp["A"] == (nA, nA/seqlen_nogaps)
        assert comp["C"] == (nC, nC/seqlen_nogaps)
        assert comp["G"] == (nG, nG/seqlen_nogaps)
        assert comp["T"] == (nT, nT/seqlen_nogaps)
        with pytest.raises(KeyError):
            notindict = comp["-"]

###################################################################################################

class Test_findgaps_DNA:

    def test_single_gap(self):
        seq = "AAAAA---CCCCC"
        dnaseq = sq.DNA_sequence("s1", seq)
        gaps = dnaseq.findgaps()
        assert gaps == [(5, 7)]

    def test_multiple_gaps(self):
        seq = "AA--CC----GG---TT"
        dnaseq = sq.DNA_sequence("s1", seq)
        gaps = dnaseq.findgaps()
        assert gaps == [(2, 3), (6, 9), (12, 14)]

    def test_no_gaps(self):
        seq = "AAAAACCCCCGGGGG"
        dnaseq = sq.DNA_sequence("s1", seq)
        gaps = dnaseq.findgaps()
        assert gaps == []

    def test_gap_at_start(self):
        seq = "---AAACCCGGG"
        dnaseq = sq.DNA_sequence("s1", seq)
        gaps = dnaseq.findgaps()
        assert gaps == [(0, 2)]

    def test_gap_at_end(self):
        seq = "AAACCCGGG---"
        dnaseq = sq.DNA_sequence("s1", seq)
        gaps = dnaseq.findgaps()
        assert gaps == [(9, 11)]

    def test_entirely_gaps(self):
        seq = "--------"
        dnaseq = sq.DNA_sequence("s1", seq)
        gaps = dnaseq.findgaps()
        assert gaps == [(0, 7)]

###################################################################################################

class Test_fasta_DNA:

    def test_len200_widthdefault_comments(self):
        seq = "".join(random.choices("ACGT", k=200))
        comments = "These are comments"
        dnaseq = sq.DNA_sequence("s1", seq, comments=comments)
        output = dnaseq.fasta()
        expected_output = (">s1 " + comments + "\n"
                            + seq[:60] + "\n"
                            + seq[60:120] + "\n"
                            + seq[120:180] + "\n"
                            + seq[180:200]
                        )
        assert output == expected_output

    def test_len200_width80_nocomments(self):
        seq = "".join(random.choices("ACGT", k=200))
        comments = "These are comments"
        dnaseq = sq.DNA_sequence("s1", seq, comments=comments)
        output = dnaseq.fasta(width=80,nocomments=True)
        expected_output = (">s1\n"
                            + seq[:80] + "\n"
                            + seq[80:160] + "\n"
                            + seq[160:200]
                        )
        assert output == expected_output

###################################################################################################

class Test_how_DNA:

    def test_len200_comments(self):
        seq = "".join(random.choices("ACGT", k=200))
        annot = "".join(random.choices("IC", k=200))
        comments = "These are comments"
        dnaseq = sq.DNA_sequence("s1", seq, annot, comments)
        output = dnaseq.how()
        expected_output = ("   200 s1 " + comments + "\n"
                            + seq[:80] + "\n"
                            + seq[80:160] + "\n"
                            + seq[160:200] + "\n"
                            + annot[:80] + "\n"
                            + annot[80:160] + "\n"
                            + annot[160:200]
                        )
        assert output == expected_output

###################################################################################################

class Test_gapencoded_DNA:

    def test_simpleseq(self):
        seq = ""
        for i in range(10):
            seq += "".join(random.choices(list(sq.Const.DNA_maxambig), k=5))
            seq += "-"*5
        dnaseq = sq.DNA_sequence("s1", seq)
        output = dnaseq.gapencoded()
        expected_output = "0000011111" * 10
        assert output == expected_output

###################################################################################################

class Test_tab_DNA:

    def test_len200_annot_comments(self):
        seq = "".join(random.choices("ACGT", k=200))
        annot = "".join(random.choices("IC", k=200))
        comments = "These are comments"
        dnaseq = sq.DNA_sequence("s1", seq, annot, comments)
        output = dnaseq.tab()
        expected_output = "s1" + "\t" + seq + "\t" + annot + "\t" + comments
        assert output == expected_output

    def test_len200_noannot_comments(self):
        seq = "".join(random.choices("ACGT", k=200))
        comments = "These are comments"
        dnaseq = sq.DNA_sequence("s1", seq, comments=comments)
        output = dnaseq.tab()
        expected_output = "s1" + "\t" + seq + "\t" + "\t" + comments
        assert output == expected_output

###################################################################################################

class Test_raw_DNA:

    def test_len200(self):
        seq = "".join(random.choices("ACGT", k=200))
        annot = "".join(random.choices("IC", k=200))
        comments = "These are comments"
        dnaseq = sq.DNA_sequence("s1", seq, annot, comments)
        output = dnaseq.raw()
        expected_output = seq
        assert output == expected_output

###################################################################################################

class Test_revcomp_DNA:

    def test_simple_sequence(self):
        seq = "ATGC"
        dnaseq = sq.DNA_sequence("s1", seq)
        revcomp_dnaseq = dnaseq.revcomp()
        assert revcomp_dnaseq.seq == "GCAT"
        assert revcomp_dnaseq.name == "s1_revcomp"

    def test_sequence_with_ambiguous_bases(self):
        seq = "ATGCRYSWKMBDHVN"
        dnaseq = sq.DNA_sequence("s2", seq)
        revcomp_dnaseq = dnaseq.revcomp()
        assert revcomp_dnaseq.seq == "NBDHVKMWSRYGCAT"
        assert revcomp_dnaseq.name == "s2_revcomp"

    def test_empty_sequence(self):
        seq = ""
        dnaseq = sq.DNA_sequence("s3", seq)
        revcomp_dnaseq = dnaseq.revcomp()
        assert revcomp_dnaseq.seq == ""
        assert revcomp_dnaseq.name == "s3_revcomp"

###################################################################################################

class Test_translate_DNA:

    def test_translate_reading_frame_1(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        dnaseq = sq.DNA_sequence("s1", seq)
        protein_seq = dnaseq.translate()
        assert protein_seq.seq == "MAIVMGR*KGAR*"
        assert isinstance(protein_seq, sq.Protein_sequence)

    def test_translate_reading_frame_2(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        dnaseq = sq.DNA_sequence("s2", seq)
        protein_seq = dnaseq.translate(reading_frame=2)
        assert protein_seq.seq == "WPL*WAAERVPD"
        assert isinstance(protein_seq, sq.Protein_sequence)

    def test_translate_reading_frame_3(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        dnaseq = sq.DNA_sequence("s3", seq)
        protein_seq = dnaseq.translate(reading_frame=3)
        assert protein_seq.seq == "GHCNGPLKGCPI"
        assert isinstance(protein_seq, sq.Protein_sequence)

    def test_translate_with_ambiguous_bases(self):
        seq = "ATGNNNCGT"
        dnaseq = sq.DNA_sequence("s4", seq)
        protein_seq = dnaseq.translate()
        assert protein_seq.seq == "MXR"
        assert isinstance(protein_seq, sq.Protein_sequence)

    def test_translate_with_short_sequence(self):
        seq = "ATG"
        dnaseq = sq.DNA_sequence("s5", seq)
        protein_seq = dnaseq.translate()
        assert protein_seq.seq == "M"
        assert isinstance(protein_seq, sq.Protein_sequence)

    def test_translate_empty_sequence(self):
        seq = ""
        dnaseq = sq.DNA_sequence("s6", seq)
        protein_seq = dnaseq.translate()
        assert protein_seq.seq == ""
        assert isinstance(protein_seq, sq.Protein_sequence)

###################################################################################################
###################################################################################################

# Tests for class Protein_sequence
# maybe also test all or some base class methods for this?

###################################################################################################

class Test_init_Protein:

    seqlen = 100

    def test_initialization(self):
        name = "protein1"
        seq = "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=self.seqlen))
        annot = "".join(random.choices("IHP", k=self.seqlen))
        comments = "Protein sequence example"
        protein_seq = sq.Protein_sequence(name=name, seq=seq, annotation=annot, comments=comments)
        assert protein_seq.name == name
        assert protein_seq.seq == seq.upper()
        assert protein_seq.annotation == annot
        assert protein_seq.comments == comments
        assert protein_seq.seqtype == "protein"

###################################################################################################
###################################################################################################

# Tests for class Protein_sequence
# maybe also test all or some base class methods for this?

###################################################################################################

class Test_ASCII_sequence:

    seqlen = 50

    def test_initialization(self):
        name = "ascii1"
        seq = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=self.seqlen))
        annot = "".join(random.choices("AB", k=self.seqlen))
        comments = "ASCII sequence example"
        ascii_seq = sq.ASCII_sequence(name=name, seq=seq, annotation=annot, comments=comments)
        assert ascii_seq.name == name
        assert ascii_seq.seq == seq.upper()
        assert ascii_seq.annotation == annot
        assert ascii_seq.comments == comments
        assert ascii_seq.seqtype == "ASCII"
        
###################################################################################################
###################################################################################################

# Tests for class Restriction_sequence
# maybe also test all or some base class methods for this?

###################################################################################################

class Test_Restriction_sequence:

    seqlen = 20

    def test_initialization(self):
        name = "restriction1"
        seq = "".join(random.choices("01", k=self.seqlen))
        annot = "".join(random.choices("01", k=self.seqlen))
        comments = "Restriction sequence example"
        restriction_seq = sq.Restriction_sequence(name=name, seq=seq, annotation=annot, comments=comments)
        assert restriction_seq.name == name
        assert restriction_seq.seq == seq.upper()
        assert restriction_seq.annotation == annot
        assert restriction_seq.comments == comments
        assert restriction_seq.seqtype == "restriction"

###################################################################################################
###################################################################################################

# Tests for class Standard_sequence
# maybe also test all or some base class methods for this?

###################################################################################################

class Test_Standard_sequence:

    seqlen = 20

    def test_initialization(self):
        name = "standard1"
        seq = "".join(random.choices("ABCDEFGH", k=self.seqlen))
        annot = "".join(random.choices("12345678", k=self.seqlen))
        comments = "Standard sequence example"
        standard_seq = sq.Standard_sequence(name=name, seq=seq, annotation=annot, comments=comments)
        assert standard_seq.name == name
        assert standard_seq.seq == seq.upper()
        assert standard_seq.annotation == annot
        assert standard_seq.comments == comments
        assert standard_seq.seqtype == "standard"

###################################################################################################
###################################################################################################

# Tests for class Mixed_sequence
# maybe also test all or some base class methods for this?

###################################################################################################

class Test_Mixed_sequence:

    seqlen = 20

    def test_initialization(self):
        name = "mixed1"
        seq = "".join(random.choices("ACGT01-", k=self.seqlen))
        annot = "".join(random.choices("ACGT01-", k=self.seqlen))
        comments = "Mixed sequence example"
        mixed_seq = sq.Mixed_sequence(name=name, seq=seq, annotation=annot, comments=comments)
        assert mixed_seq.name == name
        assert mixed_seq.seq == seq.upper()
        assert mixed_seq.annotation == annot
        assert mixed_seq.comments == comments
        assert mixed_seq.seqtype == "mixed"

###################################################################################################
###################################################################################################

# Test Classes for Contig Methods

###################################################################################################

class Test_Contig_init:

    def test_initialization(self):
        seq = sq.DNA_sequence(name="read1", seq="ATGCGT")
        contig = sq.Contig(seq)
        assert contig.name == "contig_0001"
        assert contig.assembly.seq == seq.seq
        assert contig.readdict["read1"].startpos == 0
        assert contig.readdict["read1"].stoppos == len(seq.seq)

###################################################################################################

class Test_Contig_findoverlap:

    def test_full_overlap(self):
        seq1 = sq.DNA_sequence(name="read1", seq="ATGCGT")
        seq2 = sq.DNA_sequence(name="read2", seq="GCGT")
        contig1 = sq.Contig(seq1)
        contig2 = sq.Contig(seq2)
        overlap = contig1.findoverlap(contig2, minoverlap=2)
        assert overlap == (2, 6, 0, 4, 4)  # seq2 fully overlaps at the end of seq1

    def test_partial_overlap(self):
        seq1 = sq.DNA_sequence(name="read1", seq="ATGCGT")
        seq2 = sq.DNA_sequence(name="read2", seq="CGTGA")
        contig1 = sq.Contig(seq1)
        contig2 = sq.Contig(seq2)
        overlap = contig1.findoverlap(contig2, minoverlap=3)
        assert overlap == (3, 6, 0, 3, 3)  # Partial overlap of "CGT"

    def test_no_overlap(self):
        seq1 = sq.DNA_sequence(name="read1", seq="ATGCGT")
        seq2 = sq.DNA_sequence(name="read2", seq="TTTT")
        contig1 = sq.Contig(seq1)
        contig2 = sq.Contig(seq2)
        overlap = contig1.findoverlap(contig2, minoverlap=2)
        assert overlap is None  # No overlap found

###################################################################################################

class Test_Contig_merge:

    def test_merge_with_overlap(self):
        seq1 = sq.DNA_sequence(name="read1", seq="ATGCGT")
        seq2 = sq.DNA_sequence(name="read2", seq="GCGTAA")
        contig1 = sq.Contig(seq1)
        contig2 = sq.Contig(seq2)
        overlap = contig1.findoverlap(contig2, minoverlap=2)
        contig1.merge(contig2, overlap)
        assert contig1.assembly.seq == "ATGCGTAA"  # Sequences are merged with overlapping part
        assert len(contig1.readdict) == 2  # Contains both reads

    def test_merge_no_overlap(self):
        seq1 = sq.DNA_sequence(name="read1", seq="ATGCGT")
        seq2 = sq.DNA_sequence(name="read2", seq="TTTAAA")
        contig1 = sq.Contig(seq1)
        contig2 = sq.Contig(seq2)
        overlap = contig1.findoverlap(contig2, minoverlap=2)
        assert overlap is None  # No overlap, merge should not be performed

###################################################################################################

class Test_Contig_regions:
    pass
    
    # To be written
    
###################################################################################################
###################################################################################################

# Test Classes for Read_assembler Methods

###################################################################################################
# TBD. NOte: potential issue with class level Contig counter

###################################################################################################
###################################################################################################

# Test Code for Seq_set

###################################################################################################

class Test_Seq_set_init:

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        seq_set = sq.Seq_set()  
        assert seq_set.name == "sequences"  # Default name from Sequences_base
        assert seq_set.seqtype is None
        assert seq_set.seqdict == {}
        assert seq_set.seqnamelist == []
        assert seq_set.alignment is False
        assert seq_set.seqpos2alignpos_cache == {}
        assert seq_set.alignpos2seqpos_cache == {}
        assert seq_set.alphabet is None
        assert seq_set.ambigsymbols is None

    def test_initialization_with_name_and_seqtype(self):
        """Test initialization with specific name and seqtype."""
        name = "my_seq_set"
        seqtype = "DNA"
        seq_set = sq.Seq_set(name=name, seqtype=seqtype)
        assert seq_set.name == name
        assert seq_set.seqtype == seqtype
        assert seq_set.seqdict == {}
        assert seq_set.seqnamelist == []
        assert seq_set.alignment is False
        assert seq_set.seqpos2alignpos_cache == {}
        assert seq_set.alignpos2seqpos_cache == {}
        # Assuming seqtype_attributes function returns the expected alphabet and ambigsymbols for DNA
        expected_alphabet, expected_ambigsymbols = sq.seqtype_attributes(seqtype)
        assert seq_set.alphabet == expected_alphabet
        assert seq_set.ambigsymbols == expected_ambigsymbols

    def test_initialization_with_seqlist(self):
        """Test initialization with a provided list of sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)
        assert len(seq_set.seqdict) == 2
        assert "seq1" in seq_set.seqdict
        assert "seq2" in seq_set.seqdict
        assert seq_set.seqdict["seq1"] == seq1
        assert seq_set.seqdict["seq2"] == seq2
        assert seq_set.seqnamelist == ["seq1", "seq2"]
        assert seq_set.alignment is False
        assert seq_set.seqpos2alignpos_cache == {}
        assert seq_set.alignpos2seqpos_cache == {}

    def test_initialization_with_empty_seqlist(self):
        """Test initialization with an empty sequence list."""
        seqlist = []
        seq_set = sq.Seq_set(seqlist=seqlist)
        assert seq_set.seqdict == {}
        assert seq_set.seqnamelist == []
        assert seq_set.alignment is False
        assert seq_set.seqpos2alignpos_cache == {}
        assert seq_set.alignpos2seqpos_cache == {}

###################################################################################################

class Test_Seq_set_remgaps:

    def test_remgaps_with_gaps(self):
        """Test the remgaps method when sequences contain gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-T-C-G")
        seq2 = sq.DNA_sequence(name="seq2", seq="GG--TA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)
        
        # Ensure sequences initially have gaps
        assert seq_set.seqdict["seq1"].seq == "A-T-C-G"
        assert seq_set.seqdict["seq2"].seq == "GG--TA"

        # Apply remgaps
        seq_set.remgaps()

        # Check if gaps have been removed
        assert seq_set.seqdict["seq1"].seq == "ATCG"
        assert seq_set.seqdict["seq2"].seq == "GGTA"

    def test_remgaps_without_gaps(self):
        """Test the remgaps method when sequences do not contain gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Ensure sequences initially have no gaps
        assert seq_set.seqdict["seq1"].seq == "ATCG"
        assert seq_set.seqdict["seq2"].seq == "GGTA"

        # Apply remgaps
        seq_set.remgaps()

        # Check if sequences remain unchanged
        assert seq_set.seqdict["seq1"].seq == "ATCG"
        assert seq_set.seqdict["seq2"].seq == "GGTA"

    def test_remgaps_mixed_content(self):
        """Test the remgaps method with a mix of sequences with and without gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="C--G-TA")
        seqlist = [seq1, seq2, seq3]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Ensure sequences initially have mixed content
        assert seq_set.seqdict["seq1"].seq == "A-TCG"
        assert seq_set.seqdict["seq2"].seq == "GGTA"
        assert seq_set.seqdict["seq3"].seq == "C--G-TA"

        # Apply remgaps
        seq_set.remgaps()

        # Check if gaps have been removed where necessary
        assert seq_set.seqdict["seq1"].seq == "ATCG"
        assert seq_set.seqdict["seq2"].seq == "GGTA"
        assert seq_set.seqdict["seq3"].seq == "CGTA"

    def test_remgaps_empty_sequences(self):
        """Test the remgaps method when sequences are empty."""
        seq1 = sq.DNA_sequence(name="seq1", seq="")
        seq2 = sq.DNA_sequence(name="seq2", seq="---")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Ensure sequences are initially empty or only have gaps
        assert seq_set.seqdict["seq1"].seq == ""
        assert seq_set.seqdict["seq2"].seq == "---"

        # Apply remgaps
        seq_set.remgaps()

        # Check if sequences remain or become empty
        assert seq_set.seqdict["seq1"].seq == ""
        assert seq_set.seqdict["seq2"].seq == ""

###################################################################################################

class Test_Seq_set_len:

    def test_len_empty(self):
        """Test len() with an empty Seq_set."""
        seq_set = sq.Seq_set()
        assert len(seq_set) == 0

    def test_len_non_empty(self):
        """Test len() with a Seq_set containing sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)
        assert len(seq_set) == 2

###################################################################################################

class Test_Seq_set_getitem:

    def test_getitem_by_index(self):
        """Test accessing sequences by integer index."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)
        assert seq_set[0] == seq1
        assert seq_set[1] == seq2

    def test_getitem_by_slice(self):
        """Test accessing a subset of sequences using slicing."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)
        subset = seq_set[0:2]
        assert isinstance(subset, sq.Seq_set)
        assert len(subset) == 2
        assert subset[0] == seq1
        assert subset[1] == seq2

    def test_getitem_by_tuple(self):
        """Test accessing subsequence using tuple indexing."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seqlist = [seq1]
        seq_set = sq.Seq_set(seqlist=seqlist)
        subseq = seq_set[0, 1:3]
        assert subseq.seq == "TC"

    def test_getitem_invalid_index(self):
        """Test accessing with an invalid index type raises an error."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seqlist = [seq1]
        seq_set = sq.Seq_set(seqlist=seqlist)
        with pytest.raises(sq.SeqError):
            seq_set["invalid"]

###################################################################################################

class Test_Seq_set_setitem:

    def test_setitem_valid_index(self):
        """Test setting a Sequence object using a valid integer index."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seqlist = [seq1, seq2]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Ensure initial sequence is seq2
        assert seq_set[1] == seq2

        # Set seq3 at index 1
        seq_set[1] = seq3

        # Verify that seq3 was set correctly
        assert seq_set[1] == seq3

    def test_setitem_non_integer_index(self):
        """Test setting a Sequence object using a non-integer index raises SeqError."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seqlist = [seq1]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Attempt to set using a non-integer index
        with pytest.raises(sq.SeqError, match="A set of sequences must be set using an integer index"):
            seq_set["invalid_index"] = seq1

    def test_setitem_non_sequence_object(self):
        """Test setting a non-Sequence object raises ValueError."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seqlist = [seq1]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Attempt to set a non-sequence object at index 0
        with pytest.raises(ValueError, match="Assigned value must be a Sequence object"):
            seq_set[0] = "Not a sequence object"

    def test_setitem_index_out_of_range(self):
        """Test setting a Sequence object at an index out of range raises IndexError."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seqlist = [seq1]
        seq_set = sq.Seq_set(seqlist=seqlist)

        # Attempt to set a sequence at an index that is out of range
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        with pytest.raises(IndexError):
            seq_set[2] = seq2  # Index 2 is out of range

###################################################################################################

class Test_Seq_set_eq:

    def test_eq_identical_sets(self):
        """Test equality between two identical Seq_set objects."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq1, seq2]
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 == seq_set2  # Both sets contain the same sequences

    def test_eq_same_sequences_different_order(self):
        """Test equality between Seq_set objects with same sequences in different order."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq2, seq1]  # Different order
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 == seq_set2  # Order should not matter

    def test_eq_different_sequences(self):
        """Test inequality between Seq_set objects with different sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq1, seq3]  # Different sequences
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 != seq_set2  # Different content

    def test_eq_different_sizes(self):
        """Test inequality between Seq_set objects of different sizes."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist1 = [seq1]
        seqlist2 = [seq1, seq2]  # Different size
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 != seq_set2  # Different sizes

    def test_eq_same_size_no_match(self):
        """Test inequality between Seq_set objects of same size but no matching sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq4 = sq.DNA_sequence(name="seq4", seq="CCGG")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq3, seq4]  # No matching sequences
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 != seq_set2  # No matches

###################################################################################################

class Test_Seq_set_ne:

    def test_ne_identical_sets(self):
        """Test inequality between two identical Seq_set objects (should be False)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq1, seq2]
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert not (seq_set1 != seq_set2)  # Both sets contain the same sequences, so should be False

    def test_ne_same_sequences_different_order(self):
        """Test inequality between Seq_set objects with same sequences in different order (should be False)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq2, seq1]  # Different order
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert not (seq_set1 != seq_set2)  # Order should not matter, so should be False

    def test_ne_different_sequences(self):
        """Test inequality between Seq_set objects with different sequences (should be True)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq1, seq3]  # Different sequences
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 != seq_set2  # Different content, so should be True

    def test_ne_different_sizes(self):
        """Test inequality between Seq_set objects of different sizes (should be True)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seqlist1 = [seq1]
        seqlist2 = [seq1, seq2]  # Different size
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 != seq_set2  # Different sizes, so should be True

    def test_ne_same_size_no_match(self):
        """Test inequality between Seq_set objects of same size but no matching sequences (should be True)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq4 = sq.DNA_sequence(name="seq4", seq="CCGG")
        seqlist1 = [seq1, seq2]
        seqlist2 = [seq3, seq4]  # No matching sequences
        seq_set1 = sq.Seq_set(seqlist=seqlist1)
        seq_set2 = sq.Seq_set(seqlist=seqlist2)
        assert seq_set1 != seq_set2  # No matches, so should be True

###################################################################################################

class Test_Seq_set_str:

    def test_str_empty_set(self):
        """Test string representation of an empty Seq_set."""
        seq_set = sq.Seq_set()
        assert str(seq_set) == ""  # Empty set should return an empty string

    def test_str_single_sequence(self):
        """Test string representation of a Seq_set with a single sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])
        expected_output = ">seq1\nATCG"
        assert str(seq_set) == expected_output

    def test_str_multiple_sequences(self):
        """Test string representation of a Seq_set with multiple sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])
        expected_output = ">seq1\nATCG\n>seq2\nGGTA"
        assert str(seq_set) == expected_output

    def test_str_sequence_with_gaps(self):
        """Test string representation of a Seq_set with sequences containing gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-T-C-G")
        seq2 = sq.DNA_sequence(name="seq2", seq="GG--TA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])
        expected_output = ">seq1\nA-T-C-G\n>seq2\nGG--TA"
        assert str(seq_set) == expected_output

###################################################################################################

class Test_Seq_set_sortnames:

    def test_sortnames_default(self):
        """Test sorting sequence names in ascending order."""
        seq1 = sq.DNA_sequence(name="seqB", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seqA", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seqC", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Before sorting
        assert seq_set.seqnamelist == ["seqB", "seqA", "seqC"]

        # Sort names in ascending order
        seq_set.sortnames()

        # After sorting
        assert seq_set.seqnamelist == ["seqA", "seqB", "seqC"]

    def test_sortnames_reverse(self):
        """Test sorting sequence names in descending order."""
        seq1 = sq.DNA_sequence(name="seqB", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seqA", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seqC", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Before sorting
        assert seq_set.seqnamelist == ["seqB", "seqA", "seqC"]

        # Sort names in descending order
        seq_set.sortnames(reverse=True)

        # After sorting
        assert seq_set.seqnamelist == ["seqC", "seqB", "seqA"]

    def test_sortnames_empty_set(self):
        """Test sorting on an empty Seq_set."""
        seq_set = sq.Seq_set()
        seq_set.sortnames()  # Sorting an empty set should not cause any errors
        assert seq_set.seqnamelist == []

    def test_sortnames_single_sequence(self):
        """Test sorting on a Seq_set with a single sequence."""
        seq1 = sq.DNA_sequence(name="seqA", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])
        seq_set.sortnames()  # Sorting a single sequence should not change anything
        assert seq_set.seqnamelist == ["seqA"]

###################################################################################################

class Test_Seq_set_addseq:

    def test_addseq_new_sequence(self):
        """Test adding a new sequence to Seq_set."""
        seq_set = sq.Seq_set()
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")

        # Before adding, the set should be empty
        assert len(seq_set) == 0

        # Add new sequence
        seq_set.addseq(seq1)

        # After adding, the set should contain one sequence
        assert len(seq_set) == 1
        assert seq_set.seqnamelist == ["seq1"]
        assert seq_set.seqdict["seq1"] == seq1

    def test_addseq_duplicate_sequence_name_exception(self):
        """Test adding a duplicate sequence name raises an exception when silently_discard_dup_name is False."""
        seq_set = sq.Seq_set()
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq1", seq="GGTA")  # Duplicate name

        # Add first sequence
        seq_set.addseq(seq1)

        # Adding a sequence with a duplicate name should raise an exception
        with pytest.raises(sq.SeqError, match="Duplicate sequence names: seq1"):
            seq_set.addseq(seq2)

    def test_addseq_duplicate_sequence_name_silent(self):
        """Test adding a duplicate sequence name silently discards the sequence when silently_discard_dup_name is True."""
        seq_set = sq.Seq_set()
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq1", seq="GGTA")  # Duplicate name

        # Add first sequence
        seq_set.addseq(seq1)

        # Adding a sequence with a duplicate name should not raise an exception and should be silently discarded
        seq_set.addseq(seq2, silently_discard_dup_name=True)

        # The set should still only contain the first sequence
        assert len(seq_set) == 1
        assert seq_set.seqdict["seq1"] == seq1

    def test_addseq_set_seqtype_on_first_add(self):
        """Test setting seqtype, alphabet, and ambigsymbols when adding the first sequence."""
        seq_set = sq.Seq_set()
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")

        # Before adding, seqtype, alphabet, and ambigsymbols should be None
        assert seq_set.seqtype is None
        assert seq_set.alphabet is None
        assert seq_set.ambigsymbols is None

        # Add new sequence
        seq_set.addseq(seq1)

        # After adding the first sequence, seqtype, alphabet, and ambigsymbols should be set
        assert seq_set.seqtype == seq1.seqtype
        assert seq_set.alphabet == seq1.alphabet
        assert seq_set.ambigsymbols == seq1.ambigsymbols

    def test_addseq_type_consistency_check(self):
        """Test adding a sequence with a different seqtype raises an exception."""
        seq_set = sq.Seq_set()
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.Protein_sequence(name="seq2", seq="MKV")  # Different seqtype

        # Add first sequence (DNA)
        seq_set.addseq(seq1)

        # Adding a sequence with a different seqtype should raise an exception
        with pytest.raises(sq.SeqError, match="Mismatch between sequence types: DNA vs. protein"):
            seq_set.addseq(seq2)

###################################################################################################

class Test_Seq_set_addseqset:

    def test_addseqset_all_new_sequences(self):
        """Test adding all new sequences from another Seq_set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq4 = sq.DNA_sequence(name="seq4", seq="CCGG")
        
        seq_set1 = sq.Seq_set(seqlist=[seq1, seq2])
        seq_set2 = sq.Seq_set(seqlist=[seq3, seq4])

        # Add all sequences from seq_set2 to seq_set1
        seq_set1.addseqset(seq_set2)

        # Verify all sequences are added correctly
        assert len(seq_set1) == 4
        assert seq_set1.seqnamelist == ["seq1", "seq2", "seq3", "seq4"]

    def test_addseqset_with_duplicates_exception(self):
        """Test adding sequences with duplicates raises exception when silently_discard_dup_name is False."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq1", seq="TTAA")  # Duplicate name

        seq_set1 = sq.Seq_set(seqlist=[seq1, seq2])
        seq_set2 = sq.Seq_set(seqlist=[seq3])

        # Adding sequences with duplicates should raise an exception
        with pytest.raises(sq.SeqError, match="Duplicate sequence names: seq1"):
            seq_set1.addseqset(seq_set2)

    def test_addseqset_with_duplicates_silent(self):
        """Test adding sequences with duplicates silently discards them when silently_discard_dup_name is True."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq1", seq="TTAA")  # Duplicate name

        seq_set1 = sq.Seq_set(seqlist=[seq1, seq2])
        seq_set2 = sq.Seq_set(seqlist=[seq3])

        # Adding sequences with duplicates should not raise an exception when silently_discard_dup_name is True
        seq_set1.addseqset(seq_set2, silently_discard_dup_name=True)

        # Verify that duplicate was not added
        assert len(seq_set1) == 2
        assert seq_set1.seqnamelist == ["seq1", "seq2"]

###################################################################################################

class Test_Seq_set_remseq:

    def test_remseq_existing_sequence(self):
        """Test removing an existing sequence by name."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Before removal, the set should contain two sequences
        assert len(seq_set) == 2

        # Remove sequence by name
        seq_set.remseq("seq1")

        # After removal, the set should contain only one sequence
        assert len(seq_set) == 1
        assert "seq1" not in seq_set.seqnamelist
        assert "seq1" not in seq_set.seqdict
        assert seq_set.seqnamelist == ["seq2"]

    def test_remseq_nonexistent_sequence(self):
        """Test removing a non-existent sequence raises an exception."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempt to remove a sequence that does not exist should raise an exception
        with pytest.raises(sq.SeqError, match="No such sequence: seq2"):
            seq_set.remseq("seq2")

###################################################################################################

class Test_Seq_set_remseqs:

    def test_remseqs_existing_sequences(self):
        """Test removing multiple existing sequences by their names."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Before removal, the set should contain three sequences
        assert len(seq_set) == 3

        # Remove multiple sequences by their names
        seq_set.remseqs(["seq1", "seq3"])

        # After removal, the set should contain one sequence
        assert len(seq_set) == 1
        assert "seq1" not in seq_set.seqnamelist
        assert "seq3" not in seq_set.seqnamelist
        assert seq_set.seqnamelist == ["seq2"]

    def test_remseqs_some_nonexistent_sequences(self):
        """Test removing some existing and some non-existent sequences raises an exception."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Attempt to remove sequences where one does not exist should raise an exception
        with pytest.raises(sq.SeqError, match="No such sequence: seq3"):
            seq_set.remseqs(["seq1", "seq3"])

    def test_remseqs_all_nonexistent_sequences(self):
        """Test removing all non-existent sequences raises an exception."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempt to remove sequences that do not exist should raise an exception
        with pytest.raises(sq.SeqError, match="No such sequence: seq2"):
            seq_set.remseqs(["seq2", "seq3"])

    def test_remseqs_empty_namelist(self):
        """Test removing sequences with an empty namelist does nothing."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Removing with an empty namelist should not change anything
        seq_set.remseqs([])

        # The set should still contain the original sequences
        assert len(seq_set) == 2
        assert seq_set.seqnamelist == ["seq1", "seq2"]

###################################################################################################

class Test_Seq_set_changeseqname:

    def test_changeseqname_to_new_name(self):
        """Test changing a sequence name to a new unique name."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Before changing, the name should be "seq1"
        assert seq1.name == "seq1"
        assert "seq1" in seq_set.seqnamelist
        assert "seq1" in seq_set.seqdict

        # Change name to "seq2"
        seq_set.changeseqname("seq1", "seq2")

        # After changing, the name should be updated
        assert seq1.name == "seq2"
        assert "seq2" in seq_set.seqnamelist
        assert "seq2" in seq_set.seqdict
        assert "seq1" not in seq_set.seqnamelist
        assert "seq1" not in seq_set.seqdict

    def test_changeseqname_to_existing_name_without_fix(self):
        """Test changing a sequence name to an existing name without fixing duplicates (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Attempting to change "seq1" to "seq2" should raise an exception
        with pytest.raises(sq.SeqError):
            seq_set.changeseqname("seq1", "seq2")

    def test_changeseqname_to_existing_name_with_fix(self):
        """Test changing a sequence name to an existing name with fix_dupnames=True."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Change name to "seq2", which is already taken, but allow fixing duplicates
        seq_set.changeseqname("seq1", "seq2", fix_dupnames=True)

        # The name should have been changed to "seq2_2"
        assert seq1.name == "seq2_2"
        assert "seq2_2" in seq_set.seqnamelist
        assert "seq2_2" in seq_set.seqdict
        assert "seq1" not in seq_set.seqnamelist
        assert "seq1" not in seq_set.seqdict

    def test_changeseqname_nonexistent_oldname(self):
        """Test changing the name of a non-existent sequence (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempting to change a name that does not exist should raise an exception
        with pytest.raises(sq.SeqError, match="No such sequence: seq3"):
            seq_set.changeseqname("seq3", "seq4")

    def test_changeseqname_same_name(self):
        """Test changing a sequence name to the same name (should do nothing)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Change name to the same name "seq1"
        seq_set.changeseqname("seq1", "seq1")

        # The name should remain unchanged
        assert seq1.name == "seq1"
        assert "seq1" in seq_set.seqnamelist
        assert "seq1" in seq_set.seqdict

###################################################################################################

class Test_Seq_set_getseq:

    def test_getseq_existing_sequence(self):
        """Test retrieving an existing sequence by name."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Retrieve existing sequence
        result = seq_set.getseq("seq1")

        # Check that the returned object is the correct sequence
        assert result == seq1
        assert result.seq == "ATCG"
        assert result.name == "seq1"

    def test_getseq_nonexistent_sequence(self):
        """Test retrieving a non-existent sequence raises an exception."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempt to retrieve a sequence that does not exist
        with pytest.raises(sq.SeqError, match="No such sequence: seq2"):
            seq_set.getseq("seq2")

###################################################################################################

class Test_Seq_set_subset:

    def test_subset_with_existing_names(self):
        """Test creating a subset with existing sequence names."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Create a subset with specific names
        subset = seq_set.subset(["seq1", "seq3"])

        # Check the subset contains only the specified sequences
        assert len(subset) == 2
        assert subset.seqnamelist == ["seq1", "seq3"]
        assert subset.seqdict["seq1"] == seq1
        assert subset.seqdict["seq3"] == seq3
        assert "seq2" not in subset.seqnamelist
        assert "seq2" not in subset.seqdict

    def test_subset_with_nonexistent_name(self):
        """Test creating a subset with a name that does not exist in the sequence collection raises an exception."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Attempt to create a subset with a name that does not exist
        with pytest.raises(sq.SeqError, match="Requested subset contains names that are not in sequence collection: {'seq3'}"):
            seq_set.subset(["seq1", "seq3"])

    def test_subset_with_empty_namelist(self):
        """Test creating a subset with an empty namelist returns an empty Seq_set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Create a subset with an empty namelist
        subset = seq_set.subset([])

        # Check that the subset is empty
        assert len(subset) == 0
        assert subset.seqnamelist == []
        assert subset.seqdict == {}

###################################################################################################

class Test_Seq_set_subsample:

    def test_subsample_valid_size(self):
        """Test subsampling with a valid sample size."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq4 = sq.DNA_sequence(name="seq4", seq="CCGG")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3, seq4])

        # Subsample with a sample size of 2
        subsample = seq_set.subsample(2)

        # Check that the subsample contains exactly 2 sequences
        assert len(subsample) == 2
        assert set(subsample.seqnamelist).issubset(set(seq_set.seqnamelist))

    def test_subsample_full_size(self):
        """Test subsampling with a sample size equal to the full size of the set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Subsample with a sample size equal to the full size of the set
        subsample = seq_set.subsample(2)

        # Check that the subsample contains all sequences
        assert len(subsample) == 2
        assert set(subsample.seqnamelist) == set(seq_set.seqnamelist)

    def test_subsample_larger_than_set(self):
        """Test subsampling with a sample size larger than the total size of the set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempt to subsample with a size larger than the set should raise an exception
        with pytest.raises(sq.SeqError, match="Requested samplesize larger than full data set"):
            seq_set.subsample(2)

    def test_subsample_negative_size(self):
        """Test subsampling with a negative sample size."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempt to subsample with a negative size should raise an exception
        with pytest.raises(sq.SeqError, match="Requested samplesize is negative - must be positive integer"):
            seq_set.subsample(-1)

    def test_subsample_zero_size(self):
        """Test subsampling with a sample size of zero (should return an empty set)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Subsample with a sample size of zero
        subsample = seq_set.subsample(0)

        # Check that the subsample is empty
        assert len(subsample) == 0
        assert subsample.seqnamelist == []
        assert subsample.seqdict == {}

###################################################################################################

class Test_Seq_set_subseq:

    def test_subseq_valid_range_slicesyntax_true(self):
        """Test subseq with a valid range using slicesyntax=True (0-based indexing)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Extract subseq from index 2 to 6 using slicesyntax=True (0-based)
        subseq_set = seq_set.subseq(start=2, stop=6, slicesyntax=True)

        # Check the subsequences
        assert len(subseq_set) == 2
        assert subseq_set.getseq("seq1_2_6").seq == "CGGC"  # 0-based indexing; slice includes start and excludes stop
        assert subseq_set.getseq("seq2_2_6").seq == "TACC"  # 0-based indexing

    def test_subseq_valid_range_slicesyntax_false(self):
        """Test subseq with a valid range using slicesyntax=False (1-based inclusive indexing)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Extract subseq from index 2 to 6 using slicesyntax=False (1-based)
        subseq_set = seq_set.subseq(start=2, stop=6, slicesyntax=False)

        # Check the subsequences
        assert len(subseq_set) == 2
        assert subseq_set.getseq("seq1_2_6").seq == "TCGGC"  # 1-based indexing; includes both start and stop
        assert subseq_set.getseq("seq2_2_6").seq == "GTACC"  # 1-based indexing

    def test_subseq_start_stop_out_of_bounds(self):
        """Test subseq with start and stop indices out of bounds (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempt to extract subseq with out-of-bounds indices
        with pytest.raises(sq.SeqError):
            seq_set.subseq(start=10, stop=15)

    def test_subseq_with_rename_sequences_false(self):
        """Test subseq with renaming the sequences (rename=False)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Extract subseq and rename sequences
        subseq_set = seq_set.subseq(start=2, stop=6, slicesyntax=True, rename=False)

        # Check the renamed sequences
        assert len(subseq_set) == 2
        assert subseq_set.getseq("seq1").seq == "CGGC"  # 0-based, not renamed
        assert subseq_set.getseq("seq2").seq == "TACC"  # 0-based, not renamed

    def test_subseq_with_custom_aln_name(self):
        """Test subseq with a custom alignment name."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Extract subseq with a custom alignment name
        subseq_set = seq_set.subseq(start=2, stop=6, slicesyntax=True, aln_name="custom_name")

        # Check that the alignment name is set correctly
        assert subseq_set.name == "custom_name"

###################################################################################################

class Test_Seq_set_getnames:

    def test_getnames(self):
        """Test retrieving the list of sequence names from the set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Get names of all sequences in the set
        names = seq_set.getnames()

        # Check that the returned list of names is correct
        assert names == ["seq1", "seq2", "seq3"]
        assert isinstance(names, list)

    def test_getnames_empty_set(self):
        """Test retrieving names from an empty sequence set."""
        seq_set = sq.Seq_set()

        # Get names of all sequences in the empty set
        names = seq_set.getnames()

        # Check that the returned list of names is empty
        assert names == []
        assert isinstance(names, list)

###################################################################################################

class Test_Seq_set_range:

    def test_range_valid(self):
        """Test discarding symbols outside of a valid range in sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Apply range that is valid for all sequences
        seq_set.range(1, 4)

        # Check that sequences have been truncated correctly
        assert seq_set.getseq("seq1").seq == "TCG"
        assert seq_set.getseq("seq2").seq == "GTA"

    def test_range_invalid_range_order(self):
        """Test that an invalid range (where rangefrom > rangeto) raises an error."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Apply an invalid range where rangefrom > rangeto
        with pytest.raises(sq.SeqError, match="End-of-range index is higher than start-of-range index"):
            seq_set.range(5, 2)

    def test_range_exceeds_length(self):
        """Test that a range exceeding the sequence length raises an error."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Apply a range that exceeds the sequence length
        with pytest.raises(sq.SeqError, match="Range exceeds length of sequence seq1: 4"):
            seq_set.range(1, 10)

    def test_range_exact_length(self):
        """Test that applying a range equal to the sequence length keeps sequences unchanged."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Apply a range equal to the sequence length
        seq_set.range(0, 4)

        # Check that sequences are unchanged
        assert seq_set.getseq("seq1").seq == "ATCG"
        assert seq_set.getseq("seq2").seq == "GGTA"

###################################################################################################

class Test_Seq_set_removedupseqs:

    def test_removedupseqs_with_duplicates(self):
        """Test removing duplicate sequences from the set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCG")  # Duplicate of seq1
        seq3 = sq.DNA_sequence(name="seq3", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Remove duplicate sequences
        duplist = seq_set.removedupseqs()

        # Check that duplicates were removed correctly
        assert len(seq_set) == 2
        assert seq_set.getseq("seq1").seq == "ATCG"
        assert seq_set.getseq("seq3").seq == "GGTA"
        assert "seq2" not in seq_set.getnames()
        assert duplist == [["seq1", "seq2"]]  # List of duplicates

    def test_removedupseqs_no_duplicates(self):
        """Test removing duplicates when there are no duplicates in the set."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Remove duplicate sequences
        duplist = seq_set.removedupseqs()

        # Check that no duplicates were found
        assert len(seq_set) == 3
        assert "seq1" in seq_set.getnames()
        assert "seq2" in seq_set.getnames()
        assert "seq3" in seq_set.getnames()
        assert duplist == []  # No duplicates

    def test_removedupseqs_multiple_duplicates(self):
        """Test removing multiple sets of duplicate sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCG")  # Duplicate of seq1
        seq3 = sq.DNA_sequence(name="seq3", seq="GGTA")
        seq4 = sq.DNA_sequence(name="seq4", seq="GGTA")  # Duplicate of seq3
        seq5 = sq.DNA_sequence(name="seq5", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3, seq4, seq5])

        # Remove duplicate sequences
        duplist = seq_set.removedupseqs()

        # Check that all duplicates were removed correctly
        assert len(seq_set) == 3
        assert "seq2" not in seq_set.getnames()
        assert "seq4" not in seq_set.getnames()
        assert duplist == [["seq1", "seq2"], ["seq3", "seq4"]]  # List of duplicates

    def test_removedupseqs_all_identical(self):
        """Test removing duplicates when all sequences are identical."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCG")  # Duplicate of seq1
        seq3 = sq.DNA_sequence(name="seq3", seq="ATCG")  # Duplicate of seq1
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Remove duplicate sequences
        duplist = seq_set.removedupseqs()

        # Check that only one sequence is kept
        assert len(seq_set) == 1
        assert "seq1" in seq_set.getnames()
        assert "seq2" not in seq_set.getnames()
        assert "seq3" not in seq_set.getnames()
        assert duplist == [["seq1", "seq2", "seq3"]]  # All duplicates in one group

###################################################################################################

class Test_Seq_set_group_identical_seqs:

    def test_group_identical_seqs_with_duplicates(self):
        """Test grouping identical sequences in a set with duplicates."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCG")  # Duplicate of seq1
        seq3 = sq.DNA_sequence(name="seq3", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Group identical sequences
        grouplist = seq_set.group_identical_seqs()

        # Check that groups are identified correctly
        assert grouplist == [["seq1", "seq2"], ["seq3"]]

    def test_group_identical_seqs_no_duplicates(self):
        """Test grouping identical sequences when there are no duplicates."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Group identical sequences
        grouplist = seq_set.group_identical_seqs()

        # Check that each sequence is in its own group
        assert grouplist == [["seq1"], ["seq2"], ["seq3"]]

    def test_group_identical_seqs_multiple_duplicates(self):
        """Test grouping multiple sets of identical sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCG")  # Duplicate of seq1
        seq3 = sq.DNA_sequence(name="seq3", seq="GGTA")
        seq4 = sq.DNA_sequence(name="seq4", seq="GGTA")  # Duplicate of seq3
        seq5 = sq.DNA_sequence(name="seq5", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3, seq4, seq5])

        # Group identical sequences
        grouplist = seq_set.group_identical_seqs()

        # Check that all groups are identified correctly
        assert grouplist == [["seq1", "seq2"], ["seq3", "seq4"], ["seq5"]]

    def test_group_identical_seqs_all_identical(self):
        """Test grouping when all sequences are identical."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCG")  # Duplicate of seq1
        seq3 = sq.DNA_sequence(name="seq3", seq="ATCG")  # Duplicate of seq1
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Group identical sequences
        grouplist = seq_set.group_identical_seqs()

        # Check that all sequences are in a single group
        assert grouplist == [["seq1", "seq2", "seq3"]]

    def test_group_identical_seqs_empty_set(self):
        """Test grouping identical sequences in an empty set."""
        seq_set = sq.Seq_set()

        # Group identical sequences
        grouplist = seq_set.group_identical_seqs()

        # Check that the result is an empty list
        assert grouplist == []

###################################################################################################

class Test_Seq_set_residuecounts:

    def test_residuecounts_basic(self):
        """Test residue counts for a basic set of sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Calculate residue counts
        counts = seq_set.residuecounts()
        concat_seq = seq1.seq + seq2.seq + seq3.seq
        expected_counts = Counter(concat_seq)  
        assert counts == expected_counts

    def test_residuecounts_with_gaps(self):
        """Test residue counts for sequences with gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGT-A")
        seq3 = sq.DNA_sequence(name="seq3", seq="TT-AA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Calculate residue counts
        counts = seq_set.residuecounts()
        concat_seq = seq1.seq + seq2.seq + seq3.seq
        expected_counts = Counter(concat_seq)
        assert counts == expected_counts

    def test_residuecounts_empty_set(self):
        """Test residue counts for an empty set of sequences."""
        seq_set = sq.Seq_set()

        # Calculate residue counts
        counts = seq_set.residuecounts()

        # Check that the counts are an empty Counter
        assert counts == Counter()

    def test_residuecounts_single_sequence(self):
        """Test residue counts for a single sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Calculate residue counts
        counts = seq_set.residuecounts()

        # Check that the counts are correct
        expected_counts = Counter(seq1.seq)
        assert counts == expected_counts

###################################################################################################

class Test_Seq_set_composition:

    def test_composition_basic_ignore_gaps(self):
        """Test composition (counts and frequencies) for a basic set of sequences, ignoring gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Calculate composition
        composition = seq_set.composition(ignoregaps=True)

        # Check that the composition is correct
        total_residues = 12
        expected_composition = {
            "A": [4, 4 / total_residues],
            "T": [4, 4 / total_residues],
            "C": [1, 1 / total_residues],
            "G": [3, 3 / total_residues],
        }
        assert composition == expected_composition

    def test_composition_basic_include_gaps(self):
        """Test composition (counts and frequencies) for a basic set of sequences, including gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGT-A")
        seq3 = sq.DNA_sequence(name="seq3", seq="TT-AA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Calculate composition including gaps
        composition = seq_set.composition(ignoregaps=False)

        # Check that the composition is correct
        total_residues = 15
        expected_composition = {
            "A": [4, 4 / total_residues],
            "T": [4, 4 / total_residues],
            "C": [1, 1 / total_residues],
            "G": [3, 3 / total_residues],
            "-": [3, 3 / total_residues],
        }
        assert composition == expected_composition

    def test_composition_empty_set(self):
        """Test composition for an empty set of sequences."""
        seq_set = sq.Seq_set()

        # Calculate composition
        composition = seq_set.composition(ignoregaps=True)

        # Check that the composition is an empty dictionary
        assert composition == {}

    def test_composition_single_sequence_ignore_gaps(self):
        """Test composition for a single sequence, ignoring gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATC---GATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Calculate composition ignoring gaps
        composition = seq_set.composition(ignoregaps=True)

        # Check that the composition is correct
        total_residues = 8
        expected_composition = {
            "A": [2, 2 / total_residues],
            "T": [2, 2 / total_residues],
            "C": [2, 2 / total_residues],
            "G": [2, 2 / total_residues],
        }
        assert composition == expected_composition

    def test_composition_single_sequence_include_gaps(self):
        """Test composition for a single sequence, including gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGATC-G")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Calculate composition including gaps
        composition = seq_set.composition(ignoregaps=False)

        # Check that the composition is correct
        total_residues = 9
        expected_composition = {
            "A": [2, 2 / total_residues],
            "T": [2, 2 / total_residues],
            "C": [2, 2 / total_residues],
            "G": [2, 2 / total_residues],
            "-": [1, 1 / total_residues],
        }
        assert composition == expected_composition

###################################################################################################

class Test_Seq_set_clean_names:

    def test_clean_names_basic(self):
        """Test cleaning names with basic illegal characters."""
        seq1 = sq.DNA_sequence(name="seq1,abc", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2;def", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3(ghi)", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Clean names with default illegal characters
        seq_set.clean_names()

        # Check that names have been cleaned correctly
        assert seq_set.getnames() == ["seq1_abc", "seq2_def", "seq3_ghi_"]

    def test_clean_names_custom_illegal(self):
        """Test cleaning names with custom illegal characters."""
        seq1 = sq.DNA_sequence(name="seq1*abc", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2@def", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3%ghi", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Clean names with custom illegal characters
        seq_set.clean_names(illegal="*@%")

        # Check that names have been cleaned correctly
        assert seq_set.getnames() == ["seq1_abc", "seq2_def", "seq3_ghi"]

    def test_clean_names_no_illegal_characters(self):
        """Test cleaning names when there are no illegal characters."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Clean names with default illegal characters
        seq_set.clean_names()

        # Check that names have not changed
        assert seq_set.getnames() == ["seq1", "seq2", "seq3"]

    def test_clean_names_empty_illegal_string(self):
        """Test cleaning names when illegal string is empty."""
        seq1 = sq.DNA_sequence(name="seq1*abc", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2@def", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3%ghi", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Clean names with empty illegal string
        seq_set.clean_names(illegal="")

        # Check that names have not changed
        assert seq_set.getnames() == ["seq1*abc", "seq2@def", "seq3%ghi"]

###################################################################################################

class Test_Seq_set_rename_numbered:

    def test_rename_numbered_basic(self):
        """Test renaming sequences with a basic basename."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Rename sequences
        seq_set.rename_numbered(basename="sample")

        # Check that names have been renamed correctly
        assert seq_set.getnames() == ["sample_1", "sample_2", "sample_3"]

    def test_rename_numbered_with_namefile(self, tmp_path):
        """Test renaming sequences and writing the changes to a file."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Create a temporary file path
        namefile = tmp_path / "name_changes.txt"

        # Rename sequences and write to the file
        seq_set.rename_numbered(basename="sample", namefile=str(namefile))

        # Check that names have been renamed correctly
        assert seq_set.getnames() == ["sample_1", "sample_2", "sample_3"]

        # Verify contents of the namefile
        with open(namefile, "r") as f:
            lines = f.readlines()
            assert lines == ["sample_1\tseq1\n", "sample_2\tseq2\n", "sample_3\tseq3\n"]

    def test_rename_numbered_large_set(self):
        """Test renaming a large set of sequences to ensure proper zero-padding."""
        seq_set = sq.Seq_set(seqlist=[sq.DNA_sequence(name=f"seq{i}", seq="ATCG") for i in range(100)])

        # Rename sequences
        seq_set.rename_numbered(basename="sample")

        # Check that names have been renamed correctly
        expected_names = [f"sample_{str(i + 1).zfill(3)}" for i in range(100)]
        assert seq_set.getnames() == expected_names

    def test_rename_numbered_empty_set(self):
        """Test renaming an empty set of sequences."""
        seq_set = sq.Seq_set()

        # Rename sequences
        seq_set.rename_numbered(basename="sample")

        # Check that there are no names to rename
        assert seq_set.getnames() == []

    def test_rename_numbered_large_set(self):
        """Test renaming a large set of sequences to ensure proper zero-padding."""
        seq_set = sq.Seq_set(seqlist=[sq.DNA_sequence(name=f"seq{i}", seq="ATCG") for i in range(100)])

        # Rename sequences
        seq_set.rename_numbered(basename="sample")

        # Check that names have been renamed correctly with three-digit padding
        expected_names = [f"sample_{str(i + 1).zfill(3)}" for i in range(100)]
        assert seq_set.getnames() == expected_names

###################################################################################################

class Test_Seq_set_rename_regexp:

    def test_rename_regexp_basic(self):
        """Test renaming sequence names using a simple regex pattern."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Rename sequences using regex
        seq_set.rename_regexp(old_regex=r"s..", new_string="sample")

        # Check that names have been renamed correctly
        assert seq_set.getnames() == ["sample1", "sample2", "sample3"]

    def test_rename_regexp_no_match(self):
        """Test renaming sequence names with a regex pattern that matches nothing."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Rename sequences with a pattern that matches nothing
        seq_set.rename_regexp(old_regex=r"xyz", new_string="sample")

        # Check that names have not changed
        assert seq_set.getnames() == ["seq1", "seq2", "seq3"]

    def test_rename_regexp_name_clash(self):
        """Test renaming sequence names where a name clash occurs."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq_2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq_3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Attempt to rename sequences in a way that will cause a name clash
        with pytest.raises(sq.SeqError, match=r"Name clash during renaming by regular expression:"):
            seq_set.rename_regexp(old_regex=r"_\d", new_string="1")

    def test_rename_regexp_with_namefile(self, tmp_path):
        """Test renaming sequence names using regex and writing changes to a file."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Create a temporary file path
        namefile = tmp_path / "name_changes.txt"

        # Rename sequences using regex and write to the file
        seq_set.rename_regexp(old_regex=r"seq", new_string="sample", namefile=str(namefile))

        # Check that names have been renamed correctly
        assert seq_set.getnames() == ["sample1", "sample2", "sample3"]

        # Verify contents of the namefile
        with open(namefile, "r") as f:
            lines = f.readlines()
            assert lines == ["sample1\tseq1\n", "sample2\tseq2\n", "sample3\tseq3\n"]

###################################################################################################

class Test_Seq_set_transname:

    def test_transname_basic(self, tmp_path):
        """Test translating names using a namefile with oldname/newname pairs."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTAA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2, seq3])

        # Create a temporary namefile with oldname/newname pairs
        namefile = tmp_path / "name_changes.txt"
        with open(namefile, "w") as f:
            f.write("seq1\tsample1\n")
            f.write("seq2\tsample2\n")
            f.write("seq3\tsample3\n")

        # Translate names using the namefile
        seq_set.transname(namefile=str(namefile))

        # Check that names have been translated correctly
        assert seq_set.getnames() == ["sample1", "sample2", "sample3"]

    def test_transname_nonexistent_name(self, tmp_path):
        """Test translating names when a name in the namefile does not exist in the collection."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Create a temporary namefile with an oldname that does not exist in the collection
        namefile = tmp_path / "name_changes.txt"
        with open(namefile, "w") as f:
            f.write("nonexistent_seq\tnewname\n")

        # Attempt to translate names using the namefile
        with pytest.raises(sq.SeqError, match=r"No sequence with this name: nonexistent_seq"):
            seq_set.transname(namefile=str(namefile))

    def test_transname_partial_translation(self, tmp_path):
        """Test translating names where only some names are translated."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Create a temporary namefile with only one oldname/newname pair
        namefile = tmp_path / "name_changes.txt"
        with open(namefile, "w") as f:
            f.write("seq1\tnewseq1\n")

        # Translate names using the namefile
        seq_set.transname(namefile=str(namefile))

        # Check that names have been translated correctly
        assert seq_set.getnames() == ["newseq1", "seq2"]

###################################################################################################

class Test_Seq_set_revcomp:

    def test_revcomp_dna_sequences(self):
        """Test reverse complement on a collection of DNA sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Get the reverse complement sequence collection
        revcomp_seq_set = seq_set.revcomp()

        # Check that the reverse complements are correct
        assert revcomp_seq_set.getseq("seq1_revcomp").seq == "CGAT"
        assert revcomp_seq_set.getseq("seq2_revcomp").seq == "TACC"

    def test_revcomp_non_dna_sequences(self):
        """Test reverse complement on a non-DNA sequence collection (should raise an error)."""
        seq1 = sq.Protein_sequence(name="seq1", seq="MVK")
        seq_set = sq.Seq_set(seqlist=[seq1])

        # Attempting to reverse complement non-DNA sequences should raise an exception
        with pytest.raises(sq.SeqError, match=r"Attempt to reverse complement non-DNA. Sequence type is: protein"):
            seq_set.revcomp()

    def test_revcomp_empty_set(self):
        """Test reverse complement on an empty sequence collection."""
        seq_set = sq.Seq_set(seqtype="DNA")

        # Get the reverse complement sequence collection
        revcomp_seq_set = seq_set.revcomp()

        # Check that the result is still an empty set
        assert len(revcomp_seq_set) == 0

###################################################################################################

class Test_Seq_set_translate:

    def test_translate_dna_sequences(self):
        """Test translation of a collection of DNA sequences to protein sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATGCGT")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACC")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Translate the sequence collection to protein sequences
        prot_seq_set = seq_set.translate()

        # Check that the translated sequences are correct
        assert prot_seq_set.getseq("seq1").seq == "MR"
        assert prot_seq_set.getseq("seq2").seq == "GT"

    def test_translate_non_dna_sequences(self):
        """Test translation on a non-DNA sequence collection (should raise an error)."""
        seq1 = sq.Protein_sequence(name="seq1", seq="MVK")
        seq_set = sq.Seq_set(seqtype="protein", seqlist=[seq1])

        # Attempting to translate non-DNA sequences should raise an exception
        with pytest.raises(sq.SeqError, match=r"Attempt to translate non-DNA. Sequence type is: protein"):
            seq_set.translate()

    def test_translate_empty_set(self):
        """Test translation on an empty DNA sequence collection."""
        seq_set = sq.Seq_set(seqtype="DNA")

        # Translate the empty sequence collection
        prot_seq_set = seq_set.translate()

        # Check that the result is still an empty set
        assert len(prot_seq_set) == 0

    def test_translate_dna_sequences_different_frames(self):
        """Test translation of a collection of DNA sequences with different reading frames."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATGCGTTCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGTG")
        seq_set = sq.Seq_set(seqtype="DNA", seqlist=[seq1, seq2])

        # Translate the sequence collection to protein sequences using reading frame 2
        prot_seq_set_rf2 = seq_set.translate(reading_frame=2)

        # Check that the translated sequences are correct for reading frame 2
        assert prot_seq_set_rf2.getseq("seq1").seq == "CV"
        assert prot_seq_set_rf2.getseq("seq2").seq == "VP"

###################################################################################################

class Test_Seq_set_fasta:

    def test_fasta_output_basic(self):
        """Test fasta output for a basic set of DNA sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate FASTA format
        fasta_output = seq_set.fasta()

        # Check the generated FASTA format
        expected_output = (
            ">seq1\n"
            "ATCG\n"
            ">seq2\n"
            "GGTA"
        )
        assert fasta_output == expected_output

    def test_fasta_empty_set(self):
        """Test fasta output for an empty set of sequences (should raise an error)."""
        seq_set = sq.Seq_set()

        # Attempting to create FASTA format for an empty set should raise an exception
        with pytest.raises(sq.SeqError, match=r"No sequences in sequence set.  Can't create fasta"):
            seq_set.fasta()

    def test_fasta_output_with_comments(self):
        """Test fasta output including comments."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG", comments="This is seq1")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA", comments="This is seq2")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate FASTA format with comments
        fasta_output = seq_set.fasta(nocomments=False)

        # Check the generated FASTA format including comments
        expected_output = (
            ">seq1 This is seq1\n"
            "ATCG\n"
            ">seq2 This is seq2\n"
            "GGTA"
        )
        assert fasta_output == expected_output

    def test_fasta_output_without_comments(self):
        """Test fasta output excluding comments."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG", comments="This is seq1")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA", comments="This is seq2")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate FASTA format without comments
        fasta_output = seq_set.fasta(nocomments=True)

        # Check the generated FASTA format excluding comments
        expected_output = (
            ">seq1\n"
            "ATCG\n"
            ">seq2\n"
            "GGTA"
        )
        assert fasta_output == expected_output

    def test_fasta_output_custom_width(self):
        """Test fasta output with a custom line width."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGGTACC")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTATTAGC")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate FASTA format with a custom width of 4
        fasta_output = seq_set.fasta(width=4)

        # Check the generated FASTA format with custom width
        expected_output = (
            ">seq1\n"
            "ATCG\n"
            "GGTA\n"
            "CC\n"
            ">seq2\n"
            "GGTA\n"
            "TTAG\n"
            "C"
        )
        assert fasta_output == expected_output

###################################################################################################

class Test_Seq_set_how:

    def test_how_output_basic(self):
        """Test HOW format output for a basic set of DNA sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate HOW format
        how_output = seq_set.how()

        # Check the generated HOW format
        expected_output = (
            "     4 seq1\n"
            "ATCG\n"
            "....\n"
            "     4 seq2\n"
            "GGTA\n"
            "...."
        )
        assert how_output == expected_output

    def test_how_empty_set(self):
        """Test HOW output for an empty set of sequences (should raise an error)."""
        seq_set = sq.Seq_set()

        # Attempting to create HOW format for an empty set should raise an exception
        with pytest.raises(sq.SeqError, match=r"No sequences in sequence set.  Can't create HOW"):
            seq_set.how()

    def test_how_output_with_comments(self):
        """Test HOW output including comments."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG", comments="This is seq1")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA", comments="This is seq2")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate HOW format with comments
        how_output = seq_set.how(nocomments=False)

        # Check the generated HOW format including comments
        expected_output = (
            "     4 seq1 This is seq1\n"
            "ATCG\n"
            "....\n"
            "     4 seq2 This is seq2\n"
            "GGTA\n"
            "...."
        )
        assert how_output == expected_output

###################################################################################################

class Test_Seq_set_tab:

    def test_tab_output_basic(self):
        """Test TAB format output for a basic set of DNA sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate TAB format
        tab_output = seq_set.tab()

        # Check the generated TAB format
        expected_output = (
            "seq1\tATCG\t\t\n"
            "seq2\tGGTA\t\t"
        )
        assert tab_output == expected_output

    def test_tab_empty_set(self):
        """Test TAB output for an empty set of sequences (should raise an error)."""
        seq_set = sq.Seq_set()

        # Attempting to create TAB format for an empty set should raise an exception
        with pytest.raises(sq.SeqError, match=r"No sequences in sequence set.  Can't create TAB"):
            seq_set.tab()

    def test_tab_output_with_comments(self):
        """Test TAB output including comments."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG", comments="This is seq1")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA", comments="This is seq2")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate TAB format with comments
        tab_output = seq_set.tab(nocomments=False)

        # Check the generated TAB format including comments
        expected_output = (
            "seq1\tATCG\t\tThis is seq1\n"
            "seq2\tGGTA\t\tThis is seq2"
        )
        assert tab_output == expected_output

###################################################################################################

class Test_Seq_set_raw:

    def test_raw_output_basic(self):
        """Test RAW format output for a basic set of DNA sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq_set = sq.Seq_set(seqlist=[seq1, seq2])

        # Generate RAW format
        raw_output = seq_set.raw()

        # Check the generated RAW format
        expected_output = (
            "ATCG\n"
            "GGTA"
        )
        assert raw_output == expected_output

    def test_raw_empty_set(self):
        """Test RAW output for an empty set of sequences (should raise an error)."""
        seq_set = sq.Seq_set()

        # Attempting to create RAW format for an empty set should raise an exception
        with pytest.raises(sq.SeqError, match=r"No sequences in sequence set.  Can't create RAW"):
            seq_set.raw()

###################################################################################################
###################################################################################################

# Test classes for the Seq_alignment class

###################################################################################################
###################################################################################################

class Test_Seq_alignment_init:

    def test_default_initialization(self):
        """Test the default initialization of a Seq_alignment object."""

        # Instantiate a Seq_alignment object with default parameters
        alignment = sq.Seq_alignment()

        # Verify inherited attributes from Sequences_base
        assert alignment.name == "alignment"  # Default name
        assert alignment.seqdict == {}        # Should be an empty dictionary
        assert alignment.seqnamelist == []    # Should be an empty list
        assert alignment.seqtype is None      # Default seqtype should be None
        assert alignment.alphabet is None     # Default alphabet should be None
        assert alignment.ambigsymbols is None # Default ambiguity symbols should be None

        # Verify Seq_alignment specific attributes
        assert alignment.alignment is True    # Should be True indicating it is an alignment
        assert alignment.seqpos2alignpos_cache == {}  # Should be an empty dictionary
        assert alignment.alignpos2seqpos_cache == {}  # Should be an empty dictionary
        assert alignment.annotation is None   # Default annotation should be None
        assert alignment.partitions is None   # Default partitions should be None

    def test_custom_initialization(self):
        """Test the initialization of a Seq_alignment object with custom parameters."""

        # Define custom name and seqtype
        custom_name = "my_alignment"
        custom_seqtype = "DNA"

        # Instantiate a Seq_alignment object with custom parameters
        alignment = sq.Seq_alignment(name=custom_name, seqtype=custom_seqtype)

        # Verify inherited attributes from Sequences_base
        assert alignment.name == custom_name  # Should match custom name
        assert alignment.seqdict == {}        # Should be an empty dictionary
        assert alignment.seqnamelist == []    # Should be an empty list
        assert alignment.seqtype == custom_seqtype  # Should match custom seqtype
        assert alignment.alphabet is not None       # Check that alphabet is set correctly based on seqtype
        assert alignment.ambigsymbols is not None   # Check that ambiguity symbols are set correctly based on seqtype

        # Verify Seq_alignment specific attributes
        assert alignment.alignment is True    # Should be True indicating it is an alignment
        assert alignment.seqpos2alignpos_cache == {}  # Should be an empty dictionary
        assert alignment.alignpos2seqpos_cache == {}  # Should be an empty dictionary
        assert alignment.annotation is None   # Default annotation should be None
        assert alignment.partitions is None   # Default partitions should be None

###################################################################################################

class Test_Seq_alignment_copy_alignobject:

    def test_copy_attributes(self):
        """Test that copy_alignobject copies all attributes correctly."""

        # Create an initial Seq_alignment object with custom settings
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment = sq.Seq_alignment(name="test_alignment", seqtype="DNA")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.annotation = "Test annotation"
        alignment.partitions = [("test_alignment", 0, len(seq1), "DNA")]

        # Copy the alignment object using copy_alignobject
        copied_alignment = alignment.copy_alignobject()

        # Verify that all attributes are copied correctly
        assert copied_alignment.name == alignment.name
        assert copied_alignment.seqdict == alignment.seqdict
        assert copied_alignment.seqnamelist == alignment.seqnamelist
        assert copied_alignment.alphabet == alignment.alphabet
        assert copied_alignment.ambigsymbols == alignment.ambigsymbols
        assert copied_alignment.alignment == alignment.alignment
        assert copied_alignment.seqpos2alignpos_cache == alignment.seqpos2alignpos_cache
        assert copied_alignment.alignpos2seqpos_cache == alignment.alignpos2seqpos_cache
        assert copied_alignment.annotation == alignment.annotation
        assert copied_alignment.partitions == alignment.partitions

    def test_copy_independence(self):
        """Test that the copied object is independent of the original object."""

        # Create an initial Seq_alignment object with some sequences
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment(name="test_alignment", seqtype="DNA")
        alignment.addseq(seq1)
        alignment.annotation = "Test annotation"
        alignment.partitions = [("test_alignment", 0, len(seq1), "DNA")]

        # Copy the alignment object using copy_alignobject
        copied_alignment = alignment.copy_alignobject()

        # Modify the copied object
        copied_alignment.name = "copied_alignment"
        copied_alignment.seqnamelist.append("new_seq")
        copied_alignment.alphabet.add("X")
        copied_alignment.annotation = "Modified annotation"
        copied_alignment.partitions[0] = ("modified_alignment", 0, 4, "DNA")

        # Verify that the original object is not affected
        assert alignment.name == "test_alignment"
        assert "new_seq" not in alignment.seqnamelist
        assert "X" not in alignment.alphabet
        assert alignment.annotation == "Test annotation"
        assert alignment.partitions == [("test_alignment", 0, 4, "DNA")]

    def test_deepcopy_mutable_attributes(self):
        """Test that mutable attributes are deeply copied."""

        # Create an initial Seq_alignment object with sequences
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment(name="test_alignment", seqtype="DNA")
        alignment.addseq(seq1)

        # Copy the alignment object using copy_alignobject
        copied_alignment = alignment.copy_alignobject()

        # Modify mutable attributes in the copied object
        copied_alignment.seqdict["seq1"].seq = "TTTT"
        copied_alignment.seqnamelist[0] = "new_seq1"

        # Verify that the original object's mutable attributes are not affected
        assert alignment.seqdict["seq1"].seq == "ATCG"
        assert alignment.seqnamelist[0] == "seq1"

###################################################################################################

class Test_Seq_alignment_addseq:

    def test_addseq_first_sequence(self):
        """Test adding the first sequence to an empty Seq_alignment object."""

        # Create a DNA sequence
        seq = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment()

        # Add the first sequence to the alignment
        alignment.addseq(seq)

        # Verify that the sequence was added
        assert alignment.seqdict["seq1"] == seq
        assert alignment.seqnamelist == ["seq1"]

        # Check that partitions are set correctly
        assert alignment.partitions == [("alignment", 0, len(seq), seq.seqtype)]

    def test_addseq_subsequent_sequence_same_length(self):
        """Test adding subsequent sequences with the same length."""

        # Create DNA sequences of the same length
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment = sq.Seq_alignment()

        # Add sequences to the alignment
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Verify that both sequences were added
        assert alignment.seqdict["seq1"] == seq1
        assert alignment.seqdict["seq2"] == seq2
        assert alignment.seqnamelist == ["seq1", "seq2"]

        # Check that partitions were not changed after the first sequence
        assert alignment.partitions == [("alignment", 0, len(seq1), seq1.seqtype)]

    def test_addseq_subsequent_sequence_different_length(self):
        """Test adding a sequence of different length raises an error."""

        # Create DNA sequences of different lengths
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTAC")
        alignment = sq.Seq_alignment()

        # Add the first sequence to the alignment
        alignment.addseq(seq1)

        # Adding a sequence of a different length should raise an exception
        with pytest.raises(sq.SeqError, match=r"Not an alignment: these sequences have different lengths: seq2 and seq1"):
            alignment.addseq(seq2)

    def test_addseq_duplicate_name_with_silently_discard(self):
        """Test adding a sequence with a duplicate name with silently_discard_dup_name=True."""

        # Create DNA sequences
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq1", seq="GGTA")  # Same name as seq1
        alignment = sq.Seq_alignment()

        # Add the first sequence to the alignment
        alignment.addseq(seq1)

        # Add a sequence with the same name and silently discard duplicates
        alignment.addseq(seq2, silently_discard_dup_name=True)

        # Verify that the second sequence was not added (seq1 should remain unchanged)
        assert len(alignment.seqdict) == 1
        assert alignment.seqdict["seq1"] == seq1

    def test_addseq_duplicate_name_without_silently_discard(self):
        """Test adding a sequence with a duplicate name without silently_discard_dup_name."""

        # Create DNA sequences
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq1", seq="GGTA")  # Same name as seq1
        alignment = sq.Seq_alignment()

        # Add the first sequence to the alignment
        alignment.addseq(seq1)

        # Adding a sequence with the same name without silently discarding should raise an exception
        with pytest.raises(sq.SeqError, match=r"Duplicate sequence names: seq1"):
            alignment.addseq(seq2, silently_discard_dup_name=False)

###################################################################################################

class Test_Seq_alignment_appendalignment:
    """Test suite for the appendalignment method in Seq_alignment."""

    def test_appendalignment_basic(self):
        """Test basic functionality of appending an alignment to another with matching sequence names."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment1 = sq.Seq_alignment(name="alignment1", seqtype="DNA", seqlist=[seq1, seq2])

        seq3 = sq.DNA_sequence(name="seq1", seq="CCGG")
        seq4 = sq.DNA_sequence(name="seq2", seq="TTAA")
        alignment2 = sq.Seq_alignment(name="alignment2", seqtype="DNA", seqlist=[seq3, seq4])

        # Append alignment2 to alignment1
        appended_alignment = alignment1.appendalignment(alignment2)

        # Check that the sequences are concatenated correctly
        assert appended_alignment.getseq("seq1").seq == "ATCGCCGG"
        assert appended_alignment.getseq("seq2").seq == "GGTATTAA"
        # Ensure that the partitions were updated correctly
        assert appended_alignment.partitions == [("alignment1", 0, 4, "DNA"), ("alignment2", 4, 4, "DNA")]

    def test_appendalignment_empty_self(self):
        """Test appending to an empty alignment (should raise an error)."""
        alignment1 = sq.Seq_alignment(name="alignment1")
        seq3 = sq.DNA_sequence(name="seq1", seq="CCGG")
        seq4 = sq.DNA_sequence(name="seq2", seq="TTAA")
        alignment2 = sq.Seq_alignment(name="alignment2", seqlist=[seq3, seq4])

        # Attempt to append to an empty alignment should raise SeqError
        with pytest.raises(sq.SeqError, match="Can't append alignment to empty Seq_alignment object"):
            alignment1.appendalignment(alignment2)

    def test_appendalignment_mixed_seqtype(self):
        """Test appending alignments of different sequence types resulting in mixed sequence type."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment1 = sq.Seq_alignment(name="alignment1", seqlist=[seq1, seq2])

        seq3 = sq.Protein_sequence(name="seq1", seq="MKFL")
        seq4 = sq.Protein_sequence(name="seq2", seq="HISF")
        alignment2 = sq.Seq_alignment(name="alignment2", seqlist=[seq3, seq4])

        # Append alignments of different types
        appended_alignment = alignment1.appendalignment(alignment2)

        # Check that the resulting alignment is of mixed type
        assert appended_alignment.seqtype == "mixed"
        assert appended_alignment.getseq("seq1").seq == "ATCGMKFL"
        assert appended_alignment.getseq("seq2").seq == "GGTAHISF"
        # Ensure that the partitions were updated correctly
        assert appended_alignment.partitions == [("alignment1", 0, 4, "DNA"), ("alignment2", 4, 4, "protein")]

    def test_appendalignment_mismatched_names(self):
        """Test appending alignments where sequence names do not match (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment1 = sq.Seq_alignment(name="alignment1", seqlist=[seq1, seq2])

        seq3 = sq.DNA_sequence(name="seq3", seq="CCGG")  # Mismatched name
        seq4 = sq.DNA_sequence(name="seq4", seq="TTAA")  # Mismatched name
        alignment2 = sq.Seq_alignment(name="alignment2", seqlist=[seq3, seq4])

        # Attempt to append alignments with mismatched names should raise SeqError
        with pytest.raises(sq.SeqError, match="Sequences in files have different names. No match found for seq1"):
            alignment1.appendalignment(alignment2)

    def test_appendalignment_with_partitions(self):
        """Test appending alignments and checking partition information."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment1 = sq.Seq_alignment(name="alignment1", seqlist=[seq1])
        alignment1.partitions = [("alignment1", 0, 4, "DNA")]

        seq2 = sq.DNA_sequence(name="seq1", seq="GCTA")
        alignment2 = sq.Seq_alignment(name="alignment2", seqlist=[seq2])
        alignment2.partitions = [("alignment2", 0, 4, "DNA")]

        appended_alignment = alignment1.appendalignment(alignment2)

        # Ensure that the partitions were updated correctly
        assert appended_alignment.partitions == [("alignment1", 0, 4, "DNA"), ("alignment2", 4, 4, "DNA")]
        assert appended_alignment.getseq("seq1").seq == "ATCGGCTA"

###################################################################################################

class Test_Seq_alignment_alignlen:
    """Test suite for the alignlen method in Seq_alignment."""

    def test_alignlen_non_empty(self):
        """Test the alignment length for a non-empty Seq_alignment object."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Check the length of the alignment
        assert alignment.alignlen() == 4

    def test_alignlen_empty(self):
        """Test the alignment length for an empty Seq_alignment object."""
        alignment = sq.Seq_alignment(name="empty_alignment", seqtype="DNA")
        
        # Check the length of the alignment, should be 0 for an empty alignment
        assert alignment.alignlen() == 0

    def test_alignlen_after_adding_sequence(self):
        """Test the alignment length after adding a sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA")
        
        # Initially empty, so length should be 0
        assert alignment.alignlen() == 0
        
        # Add a sequence and check the length
        alignment.addseq(seq1)
        assert alignment.alignlen() == 4

    def test_alignlen_different_lengths(self):
        """Test the alignment length consistency across multiple sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="CCTT")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2, seq3])
        
        # Check the length of the alignment, should match length of sequences
        assert alignment.alignlen() == 4

    def test_alignlen_mixed_sequence_types(self):
        """Test the alignment length for different sequence types in different partitions."""
        # First partition: DNA sequences
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTA")
        alignment1 = sq.Seq_alignment(name="alignment1", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Second partition: Protein sequences
        seq3 = sq.Protein_sequence(name="seq1", seq="MKFL")
        seq4 = sq.Protein_sequence(name="seq2", seq="HISF")
        alignment2 = sq.Seq_alignment(name="alignment2", seqtype="protein", seqlist=[seq3, seq4])
        
        # Append alignment2 to alignment1
        combined_alignment = alignment1.appendalignment(alignment2)
        
        # Check the total length of the combined alignment
        assert combined_alignment.alignlen() == 8  # 4 from DNA, 4 from protein

        # Check partition lengths are updated correctly
        assert combined_alignment.partitions == [
            ("alignment1", 0, 4, "DNA"),
            ("alignment2", 4, 4, "protein")
        ]

    def test_alignlen_with_annotation(self):
        """Test the alignment length when annotation is added."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment(name="alignment_with_annotation", seqtype="DNA", seqlist=[seq1])
        alignment.annotation = "some_annotation"
        
        # Annotation should not affect the length of the alignment
        assert alignment.alignlen() == 4
        
###################################################################################################

class Test_Seq_alignment_getcolumn:
    """Test suite for the getcolumn method in Seq_alignment."""

    def test_getcolumn_first(self):
        """Test retrieving the first column."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Get the first column (index 0)
        column = alignment.getcolumn(0)
        assert column == ['A', 'G']

    def test_getcolumn_middle(self):
        """Test retrieving a middle column."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Get the middle column (index 2)
        column = alignment.getcolumn(2)
        assert column == ['C', 'T']

    def test_getcolumn_last(self):
        """Test retrieving the last column."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Get the last column (index 3)
        column = alignment.getcolumn(3)
        assert column == ['G', 'G']

    def test_getcolumn_out_of_bounds(self):
        """Test retrieving a column out of bounds (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Attempt to get a column index that is out of bounds
        with pytest.raises(IndexError):
            alignment.getcolumn(4)

    def test_getcolumn_single_sequence(self):
        """Test retrieving a column from an alignment with a single sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1])
        
        # Get the second column (index 1)
        column = alignment.getcolumn(1)
        assert column == ['T']
        
###################################################################################################

class Test_Seq_alignment_columns:
    """Test suite for the columns method in Seq_alignment."""

    def test_columns_iteration(self):
        """Test iterating over all columns in an alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        seq3 = sq.DNA_sequence(name="seq3", seq="TTTT")
        alignment = sq.Seq_alignment(name="multi_seq_alignment", seqtype="DNA", seqlist=[seq1, seq2, seq3])
        
        # Create an iterator for the columns
        column_iterator = alignment.columns()
        all_columns = list(column_iterator)
        
        # Check that all columns are returned correctly
        expected_columns = [['A', 'G', 'T'], ['T', 'G', 'T'], ['C', 'T', 'T'], ['G', 'G', 'T']]
        assert all_columns == expected_columns

    def test_columns_empty_alignment(self):
        """Test iterating over columns in an empty alignment (should yield nothing)."""
        alignment = sq.Seq_alignment(name="empty_alignment", seqtype="DNA")
        
        # Create an iterator for the columns
        column_iterator = alignment.columns()
        all_columns = list(column_iterator)
        
        # Check that no columns are yielded
        assert all_columns == []

    def test_columns_single_column(self):
        """Test iterating over an alignment with a single column."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A")
        seq2 = sq.DNA_sequence(name="seq2", seq="G")
        alignment = sq.Seq_alignment(name="single_column_alignment", seqtype="DNA", seqlist=[seq1, seq2])
        
        # Create an iterator for the columns
        column_iterator = alignment.columns()
        all_columns = list(column_iterator)
        
        # Check that the single column is returned correctly
        assert all_columns == [['A', 'G']]

###################################################################################################

class Test_Seq_alignment_samplecols:
    """Test suite for the samplecols method in Seq_alignment."""

    def test_samplecols_valid(self):
        """Test sampling a valid number of columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Sample 4 columns
        alignment.samplecols(4)

        # Check that the alignment now has 4 columns
        assert alignment.alignlen() == 4

    def test_samplecols_full_length(self):
        """Test sampling the full length of the alignment (should keep all columns)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Sample all columns
        alignment.samplecols(8)

        # Check that the alignment still has all 8 columns
        assert alignment.alignlen() == 8

    def test_samplecols_zero_columns(self):
        """Test sampling zero columns (should result in an empty alignment)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Sample 0 columns
        alignment.samplecols(0)

        # Check that the alignment is now empty
        assert alignment.alignlen() == 0

    def test_samplecols_samplesize_too_large(self):
        """Test sampling more columns than exist (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        with pytest.raises(sq.SeqError, match="Requested samplesize larger than length of alignment"):
            alignment.samplecols(10)  # Attempt to sample more columns than the alignment length

    def test_samplecols_negative_samplesize(self):
        """Test sampling a negative number of columns (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        with pytest.raises(sq.SeqError, match="Requested samplesize is negative - must be positive integer"):
            alignment.samplecols(-1)  # Attempt to sample a negative number of columns

    def test_samplecols_randomness(self):
        """Test that sampling columns produces a random subset each time."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTACCGT")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Seed the random generator for reproducibility in the test
        random.seed(42)
        alignment.samplecols(4)
        sampled_cols_first_run = [seq.seq for seq in alignment]

        # Reset the alignment and seed again
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        random.seed(43)
        alignment.samplecols(4)
        sampled_cols_second_run = [seq.seq for seq in alignment]

        # Check that different seeds produce different results
        assert sampled_cols_first_run != sampled_cols_second_run

###################################################################################################

class Test_Seq_alignment_conscols:
    """Test suite for the conscols method in Seq_alignment."""

    def test_conscols_all_conserved(self):
        """Test conscols with all columns conserved."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AAAA")
        seq2 = sq.DNA_sequence(name="seq2", seq="AAAA")
        alignment = sq.Seq_alignment(name="alignment", seqlist=[seq1, seq2])

        conserved_columns = alignment.conscols()
        assert conserved_columns == [0, 1, 2, 3]

    def test_conscols_no_conserved(self):
        """Test conscols with no conserved columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AATG")
        seq2 = sq.DNA_sequence(name="seq2", seq="CCGA")
        alignment = sq.Seq_alignment(name="alignment", seqlist=[seq1, seq2])

        conserved_columns = alignment.conscols()
        assert conserved_columns == []

    def test_conscols_some_conserved(self):
        """Test conscols with some conserved columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ACATT")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGAGG")
        alignment = sq.Seq_alignment(name="alignment", seqlist=[seq1, seq2])

        conserved_columns = alignment.conscols()
        assert conserved_columns == [0, 2]
        
###################################################################################################

class Test_Seq_alignment_varcols:
    """Test suite for the varcols method in Seq_alignment."""

    def test_varcols_all_variable(self):
        """Test varcols with all columns variable."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AGTC")
        seq2 = sq.DNA_sequence(name="seq2", seq="CTGA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        variable_columns = alignment.varcols()
        assert variable_columns == [0, 1, 2, 3]

    def test_varcols_no_variable(self):
        """Test varcols with no variable columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AAAA")
        seq2 = sq.DNA_sequence(name="seq2", seq="AAAA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        variable_columns = alignment.varcols()
        assert variable_columns == []

    def test_varcols_some_variable(self):
        """Test varcols with some variable columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AATTA-")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTGAA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        variable_columns = alignment.varcols()
        assert variable_columns == [1, 3, 5]
        
###################################################################################################

class Test_Seq_alignment_gappycols:
    """Test suite for the gappycols method in Seq_alignment."""

    def test_gappycols_some_gaps(self):
        """Test gappycols with some columns containing gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AG-A")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        gappy_columns = alignment.gappycols()
        assert gappy_columns == [1, 2]

    def test_gappycols_all_gappy(self):
        """Test gappycols with all columns containing gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="-AC-")
        seq2 = sq.DNA_sequence(name="seq2", seq="A--G")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        gappy_columns = alignment.gappycols()
        assert gappy_columns == [0, 1, 2, 3]

    def test_gappycols_no_gaps(self):
        """Test gappycols with no gappy columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GCTA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        gappy_columns = alignment.gappycols()
        assert gappy_columns == []


###################################################################################################

class Test_Seq_alignment_site_summary:
    """Test suite for the site_summary method in Seq_alignment."""

    def test_site_summary_all_conserved(self):
        """Test site_summary with all conserved columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AAAA")
        seq2 = sq.DNA_sequence(name="seq2", seq="AAAA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        summary = alignment.site_summary()
        assert summary == (4, 0, 0)

    def test_site_summary_all_variable(self):
        """Test site_summary with all variable columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GCTA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        summary = alignment.site_summary()
        assert summary == (4, 4, 0)

    def test_site_summary_conserved_gappy(self):
        """Test site_summary with some columns gappy and conserved."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A--T")
        seq2 = sq.DNA_sequence(name="seq2", seq="G--A")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        summary = alignment.site_summary()
        assert summary == (4, 2, 2)

    def test_site_summary_mixed(self):
        """Test site_summary with a mix of conserved, variable, and gappy columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A--C")
        seq2 = sq.DNA_sequence(name="seq2", seq="A-GA")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        summary = alignment.site_summary()
        assert summary == (4, 2, 2)
        
###################################################################################################

class Test_Seq_alignment_indexfilter:
    """Test suite for the indexfilter method in Seq_alignment."""

    def test_indexfilter_basic(self):
        """Test indexfilter with a basic keeplist."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment", seqlist=[seq1, seq2])

        # Apply indexfilter to keep only columns at indices 0 and 2
        alignment.indexfilter([0, 2])

        # Check that sequences have been filtered correctly
        assert alignment.getseq("seq1").seq == "AC"
        assert alignment.getseq("seq2").seq == "AT"

    def test_indexfilter_with_annotation(self):
        """Test indexfilter with annotation present."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment", seqlist=[seq1, seq2])
        alignment.annotation = "0123"

        # Apply indexfilter to keep only columns at indices 1 and 3
        alignment.indexfilter([1, 3])

        # Check that sequences have been filtered correctly
        assert alignment.getseq("seq1").seq == "TG"
        assert alignment.getseq("seq2").seq == "GG"
        assert alignment.annotation == "13"

    def test_indexfilter_with_partitions(self):
        """Test indexfilter with partition adjustments."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="region1", seqlist=[seq1, seq2])

        # Apply indexfilter to keep only columns at indices 1 and 3
        alignment.indexfilter([1, 3])

        # Check that sequences and partitions have been adjusted correctly
        assert alignment.getseq("seq1").seq == "TG"
        assert alignment.getseq("seq2").seq == "GG"
        assert alignment.partitions == [("region1", 0, 2, "DNA")]

    def test_indexfilter_remove_all_columns(self):
        """Test indexfilter when all columns are removed."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        # Apply indexfilter with an empty keeplist
        alignment.indexfilter([])

        # Check that all sequences are empty
        assert alignment.getseq("seq1").seq == ""
        assert alignment.getseq("seq2").seq == ""

    def test_indexfilter_invalid_indices(self):
        """Test indexfilter with invalid indices in keeplist."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment", seqtype="DNA", seqlist=[seq1, seq2])

        # Apply indexfilter with indices out of range
        with pytest.raises(IndexError):
            alignment.indexfilter([0, 5])  # Index 5 is out of range
            
###################################################################################################

class Test_Seq_alignment_remcols:
    """Test suite for the remcols method in Seq_alignment."""

    def test_remcols_basic(self):
        """Test remcols with a basic discardlist."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Apply remcols to remove columns at indices 1 and 3
        alignment.remcols([1, 3])

        # Check that sequences have been filtered correctly
        assert alignment.getseq("seq1").seq == "AC"
        assert alignment.getseq("seq2").seq == "AT"

    def test_remcols_no_columns_removed(self):
        """Test remcols when no columns are removed."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Apply remcols with an empty discardlist
        alignment.remcols([])

        # Check that sequences remain unchanged
        assert alignment.getseq("seq1").seq == "ATCG"
        assert alignment.getseq("seq2").seq == "AGTG"

    def test_remcols_all_columns_removed(self):
        """Test remcols when all columns are removed."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Apply remcols to remove all columns
        alignment.remcols([0, 1, 2, 3])

        # Check that all sequences are empty
        assert alignment.getseq("seq1").seq == ""
        assert alignment.getseq("seq2").seq == ""

    def test_remcols_invalid_indices(self):
        """Test remcols with invalid indices in discardlist."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Apply remcols with indices out of range or negative, should raise SeqError
        with pytest.raises(sq.SeqError, match=r"Invalid column index in discardlist: 4"):
            alignment.remcols([0, 4])  # Index 4 is out of range

        with pytest.raises(sq.SeqError, match=r"Invalid column index in discardlist: -1"):
            alignment.remcols([-1])  # Index -1 is negative

    def test_remcols_with_partitions(self):
        """Test remcols with partition adjustments."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="region1")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Apply remcols to remove columns at indices 1 and 3
        alignment.remcols([1, 3])

        # Check that sequences and partitions have been adjusted correctly
        assert alignment.getseq("seq1").seq == "AC"
        assert alignment.getseq("seq2").seq == "AT"
        assert alignment.partitions == [("region1", 0, 2, "DNA")]
        
###################################################################################################

class Test_Seq_alignment_remambigcol:
    """Test suite for the remambigcol method in Seq_alignment."""

    def test_remambigcol_no_ambiguities(self):
        """Test remambigcol when there are no ambiguity symbols."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remambigcol()

        # No columns should be removed
        assert alignment.getseq("seq1").seq == "ATCG"
        assert alignment.getseq("seq2").seq == "AGTG"

    def test_remambigcol_with_ambiguities(self):
        """Test remambigcol when there are ambiguity symbols."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGN")
        seq2 = sq.DNA_sequence(name="seq2", seq="AYTGT")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remambigcol()

        # Columns with ambiguities ('N' and 'X') should be removed
        assert alignment.getseq("seq1").seq == "ACG"
        assert alignment.getseq("seq2").seq == "ATG"

    def test_remambigcol_all_ambiguities(self):
        """Test remambigcol when all columns are ambiguous."""
        seq1 = sq.DNA_sequence(name="seq1", seq="NNNN")
        seq2 = sq.DNA_sequence(name="seq2", seq="RRRR")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remambigcol()

        # All columns should be removed, resulting in empty sequences
        assert alignment.getseq("seq1").seq == ""
        assert alignment.getseq("seq2").seq == ""

###################################################################################################

class Test_Seq_alignment_remfracambigcol:
    """Test suite for the remfracambigcol method in Seq_alignment."""

    def test_remfracambigcol_partial_ambiguity(self):
        """Test remfracambigcol with a fraction threshold."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGN")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTGN")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remfracambigcol(frac=0.5)

        # Only column with ambiguities 'N' and 'N' where ambiguity fraction >= 0.5 should be removed
        assert alignment.getseq("seq1").seq == "ATCG"
        assert alignment.getseq("seq2").seq == "AGTG"

    def test_remfracambigcol_all_ambiguities(self):
        """Test remfracambigcol when all columns exceed the fraction threshold."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ANNN")
        seq2 = sq.DNA_sequence(name="seq2", seq="NCGN")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remfracambigcol(frac=0.5)

        # All columns should be removed due to high ambiguity fraction
        assert alignment.getseq("seq1").seq == ""
        assert alignment.getseq("seq2").seq == ""

###################################################################################################

class Test_Seq_alignment_remgapcol:
    """Test suite for the remgapcol method in Seq_alignment."""

    def test_remgapcol_no_gaps(self):
        """Test remgapcol when there are no gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remgapcol()

        # No columns should be removed
        assert alignment.getseq("seq1").seq == "ATCG"
        assert alignment.getseq("seq2").seq == "AGTG"

    def test_remgapcol_with_gaps(self):
        """Test remgapcol when there are gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATC-G")
        seq2 = sq.DNA_sequence(name="seq2", seq="A-T-G")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remgapcol()

        # Columns with gaps should be removed
        assert alignment.getseq("seq1").seq == "ACG"
        assert alignment.getseq("seq2").seq == "ATG"
        
###################################################################################################

class Test_Seq_alignment_remfracgapcol:
    """Test suite for the remfracgapcol method in Seq_alignment."""

    def test_remfracgapcol_partial_gaps(self):
        """Test remfracgapcol with a fraction threshold."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG-")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGT-G")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remfracgapcol(frac=0.5)

        # Columns where gap fraction >= 0.5 should be removed
        assert alignment.getseq("seq1").seq == "ATC"
        assert alignment.getseq("seq2").seq == "AGT"

    def test_remfracgapcol_all_gaps(self):
        """Test remfracgapcol when all columns exceed the fraction threshold."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A---")
        seq2 = sq.DNA_sequence(name="seq2", seq="----")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remfracgapcol(frac=0.5)

        # All columns should be removed due to high gap fraction
        assert alignment.getseq("seq1").seq == ""
        assert alignment.getseq("seq2").seq == ""

###################################################################################################

class Test_Seq_alignment_remendgapcol:
    """Test suite for the remendgapcol method in Seq_alignment."""

    def test_remendgapcol_no_end_gaps(self):
        """Test remendgapcol when there are no end gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remendgapcol(frac=0.5)

        # No columns should be removed
        assert alignment.getseq("seq1").seq == "ATCG"
        assert alignment.getseq("seq2").seq == "AGTG"

    def test_remendgapcol_with_end_gaps(self):
        """Test remendgapcol when there are end gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="--CG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remendgapcol(frac=0.5)

        # End columns with gaps should be removed
        assert alignment.getseq("seq1").seq == "CG"
        assert alignment.getseq("seq2").seq == "TG"

    def test_remendgapcol_all_end_gaps(self):
        """Test remendgapcol when all columns are end gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="----")
        seq2 = sq.DNA_sequence(name="seq2", seq="ACGT")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remendgapcol(frac=0.5)

        # All columns should be removed due to end gaps
        assert alignment.getseq("seq1").seq == ""
        assert alignment.getseq("seq2").seq == ""
        
###################################################################################################

class Test_Seq_alignment_endgapfraclist:
    """Test suite for the endgapfraclist method in Seq_alignment."""

    def test_endgapfraclist_no_gaps(self):
        """Test when there are no gaps in any sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GCTA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        endgapfrac = alignment.endgapfraclist()

        # No columns have end gaps
        assert endgapfrac == [0.0, 0.0, 0.0, 0.0]

    def test_endgapfraclist_some_gaps(self):
        """Test when some sequences have gaps at the ends."""
        seq1 = sq.DNA_sequence(name="seq1", seq="-TCGGC")
        seq2 = sq.DNA_sequence(name="seq2", seq="GCTAG-")
        seq3 = sq.DNA_sequence(name="seq3", seq="---A-T")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        endgapfrac = alignment.endgapfraclist()

        # Mixed gaps at different ends of the sequences
        assert endgapfrac == [2/3, 1/3, 1/3, 0.0, 0.0, 1/3]

    def test_endgapfraclist_all_gaps(self):
        """Test when all sequences are entirely gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="----")
        seq2 = sq.DNA_sequence(name="seq2", seq="----")
        seq3 = sq.DNA_sequence(name="seq3", seq="----")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        endgapfrac = alignment.endgapfraclist()

        # All positions have 100% gaps
        assert endgapfrac == [1.0, 1.0, 1.0, 1.0]

    def test_endgapfraclist_empty_alignment(self):
        """Test behavior when alignment has no sequences."""
        alignment = sq.Seq_alignment(name="empty_alignment")

        endgapfrac = alignment.endgapfraclist()

        # End gap fraction list should be empty
        assert endgapfrac == []
        
###################################################################################################

class Test_Seq_alignment_remendgapseqs:
    """Test suite for the remendgapseqs method in Seq_alignment."""

    def test_remendgapseqs_valid_cutoff(self):
        """Test removing sequences with end gaps above a specified cutoff."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG-")
        seq2 = sq.DNA_sequence(name="seq2", seq="--GTA")
        seq3 = sq.DNA_sequence(name="seq3", seq="AGCTA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Remove sequences with end gaps of 2 or more
        alignment.remendgapseqs(cutoff=2)

        # Only seq1 and seq3 should remain
        assert alignment.getnames() == ["seq1", "seq3"]

    def test_remendgapseqs_cutoff_none(self):
        """Test that providing no cutoff raises an exception."""
        alignment = sq.Seq_alignment(name="alignment")

        with pytest.raises(sq.SeqError, match="Must provide cutoff"):
            alignment.remendgapseqs(cutoff=None)

    def test_remendgapseqs_no_removal(self):
        """Test when no sequences have end gaps above the cutoff."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG-")
        seq2 = sq.DNA_sequence(name="seq2", seq="GTA--")
        seq3 = sq.DNA_sequence(name="seq3", seq="AGCTA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Remove sequences with end gaps of 3 or more
        alignment.remendgapseqs(cutoff=3)

        # No sequences should be removed
        assert alignment.getnames() == ["seq1", "seq2", "seq3"]

    def test_remendgapseqs_all_removed(self):
        """Test when all sequences have end gaps above the cutoff."""
        seq1 = sq.DNA_sequence(name="seq1", seq="---C")
        seq2 = sq.DNA_sequence(name="seq2", seq="A---")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Remove sequences with end gaps of 1 or more
        alignment.remendgapseqs(cutoff=1)

        # All sequences should be removed
        assert alignment.getnames() == []
        
###################################################################################################

class Test_Seq_alignment_remconscol:
    """Test suite for the remconscol method in Seq_alignment."""

    def test_remconscol_basic(self):
        """Test removing conserved columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATCA")
        seq3 = sq.DNA_sequence(name="seq3", seq="ATCG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        alignment.remconscol()

        # Columns 0, 1, and 2 are conserved, so only column 3 should remain
        assert alignment.alignlen() == 1
        assert alignment.getcolumn(0) == ["G", "A", "G"]

    def test_remconscol_no_conserved_columns(self):
        """Test when there are no conserved columns."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GTCA")
        seq3 = sq.DNA_sequence(name="seq3", seq="TCAG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        alignment.remconscol()

        # No columns are conserved, so all should remain
        assert alignment.alignlen() == 4
        assert seq1.seq == "ATCG"
        assert seq2.seq == "GTCA"
        assert seq3.seq == "TCAG"

    def test_remconscol_all_conserved_columns(self):
        """Test when all columns are conserved."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AAAA")
        seq2 = sq.DNA_sequence(name="seq2", seq="AAAA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        alignment.remconscol()

        # All columns are conserved, so none should remain
        assert alignment.alignlen() == 0
        
###################################################################################################

class Test_Seq_alignment_rem_hmmalign_insertcol:
    """Test suite for the rem_hmmalign_insertcol method in Seq_alignment."""

    # Python note: testing with mocked annotation fields. Should i read actual HMMer file?

    def test_rem_hmmalign_insertcol_basic(self):
        """Test removing insert state columns based on annotation."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GACTA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Set annotation indicating an insert state at column 1
        alignment.annotation = "mimmm"

        alignment.rem_hmmalign_insertcol()

        # Insert state at column 1 should be removed
        assert alignment.alignlen() == 4
        assert seq1.seq == "ATCG"
        assert seq2.seq == "GCTA"

    def test_rem_hmmalign_insertcol_no_annotation(self):
        """Test behavior when there's no annotation (should raise an error)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="G-CTA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        with pytest.raises(sq.SeqError, match="This alignment contains no information about hmmalign insert states"):
            alignment.rem_hmmalign_insertcol()

    def test_rem_hmmalign_insertcol_no_insert_states(self):
        """Test when there are no insert state columns in the annotation."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GCTA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Set annotation indicating no insert states
        alignment.annotation = "mmmm"

        alignment.rem_hmmalign_insertcol()

        # No insert state columns should be removed
        assert all(seq.seq == original for seq, original in zip(alignment, ["ATCG", "GCTA"]))
        
###################################################################################################

class Test_Seq_alignment_findgaps:
    """Test suite for the findgaps method in Seq_alignment."""

    def test_findgaps_no_gaps(self):
        """Test finding gaps when there are no gaps in any sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # There should be no gaps
        assert alignment.findgaps() == []

    def test_findgaps_single_gap(self):
        """Test finding a single gap in one sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG-")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG-")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Both sequences have a gap at the same position
        assert alignment.findgaps() == [(4, 4)]

    def test_findgaps_multiple_gaps_same_positions(self):
        """Test finding multiple gaps at the same positions in all sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TCG-G")
        seq2 = sq.DNA_sequence(name="seq2", seq="G-GTG-G")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Gaps are at positions 1 and 5 in both sequences
        assert alignment.findgaps() == [(1, 1), (5, 5)]

    def test_findgaps_multiple_gaps_different_positions(self):
        """Test finding multiple gaps at different positions in the sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TC-GG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGT-G-G")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Gaps are at different positions in the sequences
        assert alignment.findgaps() == [(1, 1), (3,3),(4, 4), (5, 5)]

    def test_findgaps_overlapping_gaps(self):
        """Test finding overlapping gaps across sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-TCGG-")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGT-GG-")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Overlapping gaps should be found
        assert alignment.findgaps() == [(1, 1), (3,3),(6, 6)]

    def test_findgaps_consecutive_gaps(self):
        """Test finding consecutive gaps in the sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATC--GG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGT--GG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Consecutive gaps spanning positions 3 to 4
        assert alignment.findgaps() == [(3, 4)]

    def test_findgaps_mixed_gap_positions(self):
        """Test finding mixed gaps with different lengths and positions."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A--TCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GG---G")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Gaps span different positions in the sequences
        assert alignment.findgaps() == [(1, 2), (2, 4)]

    def test_findgaps_large_alignment(self):
        """Test finding gaps in a larger alignment with multiple sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATC---G")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTCG-G")
        seq3 = sq.DNA_sequence(name="seq3", seq="AT---CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Multiple gaps across sequences
        assert alignment.findgaps() == [(2, 4), (3, 5), (5, 5)]

    def test_findgaps_empty_alignment(self):
        """Test findgaps method with an empty alignment."""
        alignment = sq.Seq_alignment(name="alignment")
        
        # Empty alignment should return no gaps
        assert alignment.findgaps() == []
        
###################################################################################################

class Test_Seq_alignment_gap_encode:
    """Test suite for the gap_encode method in Seq_alignment."""

    def test_gap_encode_no_gaps(self):
        """Test gap encoding when there are no gaps in any sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding contains no gaps (all zeros)
        assert len(gap_alignment) == 2
        assert gap_alignment.getseq("seq1").seq == ""  # No gaps, so no binary encoding
        assert gap_alignment.getseq("seq2").seq == ""

    def test_gap_encode_single_gap(self):
        """Test gap encoding with a single gap of multiple residues in one sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG--")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTGCG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding reflects the single gap correctly
        assert len(gap_alignment) == 2
        assert gap_alignment.getseq("seq1").seq == "1"  # (4, 5)
        assert gap_alignment.getseq("seq2").seq == "0"

    def test_gap_encode_multiple_gaps(self):
        """Test gap encoding with multiple gaps at different positions in sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A---TCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GGTC---")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding reflects gaps at different positions correctly
        assert len(gap_alignment) == 2
        assert gap_alignment.getseq("seq1").seq == "10"  # (1, 3), (4, 6)
        assert gap_alignment.getseq("seq2").seq == "01"

    def test_gap_encode_overlapping_gaps(self):
        """Test gap encoding with overlapping gaps across sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A--TCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GG--CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding reflects overlapping gaps
        assert len(gap_alignment) == 2
        assert gap_alignment.getseq("seq1").seq == "10"  # (1, 2)
        assert gap_alignment.getseq("seq2").seq == "01"  # (2, 3)

    def test_gap_encode_same_gap(self):
        """Test gap encoding with same gap in the sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT--CGG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GG--CGA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding reflects consecutive gaps correctly
        assert len(gap_alignment) == 2
        assert gap_alignment.getseq("seq1").seq == "1"  # (2, 3)
        assert gap_alignment.getseq("seq2").seq == "1"

    def test_gap_encode_empty_alignment(self):
        """Test gap encoding with an empty alignment."""
        alignment = sq.Seq_alignment(name="alignment")

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding is empty
        assert len(gap_alignment) == 0

    def test_gap_encode_all_gaps(self):
        """Test gap encoding with sequences that are entirely gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="------")
        seq2 = sq.DNA_sequence(name="seq2", seq="------")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Create the gap encoding alignment
        gap_alignment = alignment.gap_encode()

        # Check that the gap encoding is all ones
        assert len(gap_alignment) == 2
        assert gap_alignment.getseq("seq1").seq == "1"  # Single continuous gap (0, 5)
        assert gap_alignment.getseq("seq2").seq == "1"

###################################################################################################

class Test_Seq_alignment_align_seq_pos_cache_builder:
    """Test suite for the align_seq_pos_cache_builder method in Seq_alignment."""

    def test_align_seq_pos_cache_builder_basic(self):
        """Test cache builder with a simple sequence."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Build cache
        alignment.align_seq_pos_cache_builder("seq1")

        # Check that the cache is correctly built
        assert alignment.seqpos2alignpos_cache["seq1"] == [0, 1, 3, 4]
        assert alignment.alignpos2seqpos_cache["seq1"] == {
            0: (0, False),
            1: (1, False),
            2: (1, True),  # Gap
            3: (2, False),
            4: (3, False),
        }

    def test_align_seq_pos_cache_builder_all_gaps(self):
        """Test cache builder with a sequence containing only gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="----")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Build cache
        alignment.align_seq_pos_cache_builder("seq1")

        # Check that the cache is correctly built
        assert alignment.seqpos2alignpos_cache["seq1"] == []
        assert alignment.alignpos2seqpos_cache["seq1"] == {
            0: (-1, True),
            1: (-1, True),
            2: (-1, True),
            3: (-1, True),
        }

    def test_align_seq_pos_cache_builder_no_gaps(self):
        """Test cache builder with a sequence with no gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Build cache
        alignment.align_seq_pos_cache_builder("seq1")

        # Check that the cache is correctly built
        assert alignment.seqpos2alignpos_cache["seq1"] == [0, 1, 2, 3]
        assert alignment.alignpos2seqpos_cache["seq1"] == {
            0: (0, False),
            1: (1, False),
            2: (2, False),
            3: (3, False),
        }     
        
###################################################################################################

class Test_Seq_alignment_seqpos2alignpos:
    """Test suite for the seqpos2alignpos method in Seq_alignment."""

    def test_seqpos2alignpos_basic(self):
        """Test seqpos2alignpos with basic sequences and default slice syntax."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Retrieve alignment position corresponding to a sequence position
        assert alignment.seqpos2alignpos("seq1", 0) == 0  # A
        assert alignment.seqpos2alignpos("seq1", 1) == 1  # T
        assert alignment.seqpos2alignpos("seq1", 2) == 3  # C
        assert alignment.seqpos2alignpos("seq1", 3) == 4  # G

    def test_seqpos2alignpos_non_slicesyntax(self):
        """Test seqpos2alignpos with non-slicesyntax."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Retrieve alignment position corresponding to a sequence position with non-slicesyntax
        assert alignment.seqpos2alignpos("seq1", 1, slicesyntax=False) == 1
        assert alignment.seqpos2alignpos("seq1", 2, slicesyntax=False) == 2
        assert alignment.seqpos2alignpos("seq1", 3, slicesyntax=False) == 4
        assert alignment.seqpos2alignpos("seq1", 4, slicesyntax=False) == 5

    def test_seqpos2alignpos_out_of_range(self):
        """Test seqpos2alignpos with a sequence position that is out of range."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        with pytest.raises(IndexError):
            alignment.seqpos2alignpos("seq1", 5)  # Out of range

    def test_seqpos2alignpos_invalid_seqname(self):
        """Test seqpos2alignpos with an invalid sequence name."""
        alignment = sq.Seq_alignment(name="alignment")

        with pytest.raises(sq.SeqError):
            alignment.seqpos2alignpos("seq_invalid", 0)   
            
###################################################################################################

class Test_Seq_alignment_alignpos2seqpos:
    """Test suite for the alignpos2seqpos method in Seq_alignment."""

    def test_alignpos2seqpos_basic(self):
        """Test alignpos2seqpos with basic sequences and default slice syntax."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Retrieve sequence position and gap status corresponding to an alignment position
        assert alignment.alignpos2seqpos("seq1", 0) == (0, False)  # A
        assert alignment.alignpos2seqpos("seq1", 1) == (1, False)  # T
        assert alignment.alignpos2seqpos("seq1", 2) == (1, True)   # Gap
        assert alignment.alignpos2seqpos("seq1", 3) == (2, False)  # C
        assert alignment.alignpos2seqpos("seq1", 4) == (3, False)  # G

    def test_alignpos2seqpos_non_slicesyntax(self):
        """Test alignpos2seqpos with non-slicesyntax."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        # Retrieve sequence position and gap status with non-slicesyntax
        assert alignment.alignpos2seqpos("seq1", 1, slicesyntax=False) == (1, False)
        assert alignment.alignpos2seqpos("seq1", 2, slicesyntax=False) == (2, False)
        assert alignment.alignpos2seqpos("seq1", 3, slicesyntax=False) == (2, True)
        assert alignment.alignpos2seqpos("seq1", 4, slicesyntax=False) == (3, False)
        assert alignment.alignpos2seqpos("seq1", 5, slicesyntax=False) == (4, False)

    def test_alignpos2seqpos_out_of_range(self):
        """Test alignpos2seqpos with an alignment position that is out of range."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)

        with pytest.raises(KeyError):
            alignment.alignpos2seqpos("seq1", 5)  # Out of range

    def test_alignpos2seqpos_invalid_seqname(self):
        """Test alignpos2seqpos with an invalid sequence name."""
        alignment = sq.Seq_alignment(name="alignment")

        with pytest.raises(sq.SeqError):
            alignment.alignpos2seqpos("seq_invalid", 0)
            
###################################################################################################

class Test_Seq_alignment_shannon:
    """Test suite for the shannon method in Seq_alignment."""

    def test_shannon_countgaps_true(self):
        """Test Shannon entropy calculation with countgaps=True."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATGC")
        seq2 = sq.DNA_sequence(name="seq2", seq="A-CC")
        seq3 = sq.DNA_sequence(name="seq3", seq="AT-C")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Compute Shannon entropy with gaps counted as symbols
        shannon_entropies = alignment.shannon(countgaps=True)

        # Calculate expected Shannon entropy values manually
        expected_entropies = [
            -(3/3 * log(3/3, 2)),  # Column 0: A, A, A (no variation)
            -(2/3 * log(2/3, 2) + 1/3 * log(1/3, 2)),  # Column 1: T, -, T
            -(1/3 * log(1/3, 2) + 1/3 * log(1/3, 2) + 1/3 * log(1/3, 2)),  # Column 2: G, C, -
            0.0  # Column 3: C, C, C (no variation)
        ]

        # Assert the results are close due to floating point precision
        assert all(abs(a - b) < 1e-6 for a, b in zip(shannon_entropies, expected_entropies))

    def test_shannon_countgaps_false(self):
        """Test Shannon entropy calculation with countgaps=False."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATGC")
        seq2 = sq.DNA_sequence(name="seq2", seq="A-CC")
        seq3 = sq.DNA_sequence(name="seq3", seq="AT-C")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Compute Shannon entropy without counting gaps as symbols
        shannon_entropies = alignment.shannon(countgaps=False)

        # Calculate expected Shannon entropy values manually
        expected_entropies = [
            0.0,  # Column 0: A, A, A (no variation)
            0.0,  # Column 1: T, T (no variation, ignore gaps)
            -(0.5 * log(0.5, 2) + 0.5 * log(0.5, 2)),  # Column 2: G, C (- ignored)
            0.0   # Column 3: C, C, C (no variation)
        ]

        # Assert the results are close due to floating point precision
        assert all(abs(a - b) < 1e-6 for a, b in zip(shannon_entropies, expected_entropies))

    def test_shannon_all_gaps(self):
        """Test Shannon entropy calculation on an alignment with all gaps in a column."""
        seq1 = sq.DNA_sequence(name="seq1", seq="----")
        seq2 = sq.DNA_sequence(name="seq2", seq="----")
        seq3 = sq.DNA_sequence(name="seq3", seq="----")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Compute Shannon entropy with gaps counted as symbols
        shannon_entropies = alignment.shannon(countgaps=True)
        # All gaps should lead to zero entropy (no variation)
        expected_entropies = [0.0, 0.0, 0.0, 0.0]
        assert shannon_entropies == expected_entropies

        # Compute Shannon entropy without counting gaps
        shannon_entropies = alignment.shannon(countgaps=False)
        # No non-gap symbols should lead to entropy calculation of 0 for each column
        expected_entropies = [0.0, 0.0, 0.0, 0.0]
        assert shannon_entropies == expected_entropies

    def test_shannon_empty_alignment(self):
        """Test Shannon entropy calculation on an empty alignment."""
        alignment = sq.Seq_alignment(name="alignment")
        
        with pytest.raises(sq.SeqError):
            alignment.shannon(countgaps=True)

###################################################################################################

class Test_Seq_alignment_nucfreq:
    """Test suite for the nucfreq method in Seq_alignment."""

    def test_nucfreq_all_positions(self):
        """Test nucleotide frequency distribution for all positions in the alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATGC")
        seq3 = sq.DNA_sequence(name="seq3", seq="ATGG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Calculate nucleotide frequencies for all positions
        freqmat = alignment.nucfreq()

        # Expected frequency matrix
        # Column 1: [A, A, A] => A:1.0, C:0.0, G:0.0, T:0.0
        # Column 2: [T, T, T] => A:0.0, C:0.0, G:0.0, T:1.0
        # Column 3: [C, G, G] => A:0.0, C:0.33, G:0.67, T:0.0
        # Column 4: [G, C, G] => A:0.0, C:0.33, G:0.67, T:0.0
        expected_freqmat = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Column 1
            [0.0, 0.0, 0.0, 1.0],  # Column 2
            [0.0, 0.33, 0.67, 0.0],  # Column 3
            [0.0, 0.33, 0.67, 0.0]   # Column 4
        ])

        # Assert that the frequency matrix is close due to floating-point precision
        np.testing.assert_almost_equal(freqmat, expected_freqmat, decimal=2)

    def test_nucfreq_specific_positions(self):
        """Test nucleotide frequency distribution for specific positions in the alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATGC")
        seq3 = sq.DNA_sequence(name="seq3", seq="ATGG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Calculate nucleotide frequencies for specified positions [1, 3]
        freqmat = alignment.nucfreq(poslist=[1, 3])

        # Expected frequency matrix for positions [1, 3]
        expected_freqmat = np.array([
            [0.0, 0.0, 0.0, 1.0],  # Column 2
            [0.0, 0.33, 0.67, 0.0]  # Column 4
        ])

        # Assert that the frequency matrix is close due to floating-point precision
        np.testing.assert_almost_equal(freqmat, expected_freqmat, decimal=2)

    def test_nucfreq_no_valid_nucleotides(self):
        """Test nucleotide frequency distribution when no valid nucleotides are present (all gaps)."""
        seq1 = sq.DNA_sequence(name="seq1", seq="--")
        seq2 = sq.DNA_sequence(name="seq2", seq="--")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Calculate nucleotide frequencies for all positions
        freqmat = alignment.nucfreq()

        # Expected to handle all-gaps by returning zero frequencies
        expected_freqmat = np.array([
            [0.0, 0.0, 0.0, 0.0],  # Column 1
            [0.0, 0.0, 0.0, 0.0]  # Column 2
        ])

        # Assert that the frequency matrix is close due to floating-point precision
        np.testing.assert_almost_equal(freqmat, expected_freqmat, decimal=2)

###################################################################################################

class Test_Seq_alignment_consensus:
    """Test suite for the consensus method in Seq_alignment."""

    def test_consensus_basic(self):
        """Test the consensus sequence generation for a basic set of sequences."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATGC")
        seq3 = sq.DNA_sequence(name="seq3", seq="ATGG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Compute consensus sequence
        consensus_seq = alignment.consensus()

        # Expected consensus sequence: "ATGG"
        assert consensus_seq.seq == "ATGG"
        assert consensus_seq.name == "alignment"
        assert consensus_seq.seqtype == "DNA"

    def test_consensus_basic_protein(self):
        """Test the consensus sequence generation for a basic set of protein sequences."""
        seq1 = sq.Protein_sequence(name="seq1", seq="KLMN")
        seq2 = sq.Protein_sequence(name="seq2", seq="KLRT")
        seq3 = sq.Protein_sequence(name="seq3", seq="KLRN")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Compute consensus sequence
        consensus_seq = alignment.consensus()

        # Expected consensus sequence: "KLRN"
        assert consensus_seq.seq == "KLRN"
        assert consensus_seq.name == "alignment"
        assert consensus_seq.seqtype == "protein"

    def test_consensus_with_gaps(self):
        """Test the consensus sequence generation with gaps in the alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A-GG")
        seq2 = sq.DNA_sequence(name="seq2", seq="-TGC")
        seq3 = sq.DNA_sequence(name="seq3", seq="-T-G")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Compute consensus sequence
        consensus_seq = alignment.consensus()

        # Expected consensus sequence: "-TGG"
        assert consensus_seq.seq == "-TGG"

    def test_consensus_tie(self):
        """Test the consensus sequence when there is a tie in the most frequent symbol."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="GTCG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Compute consensus sequence
        consensus_seq = alignment.consensus()

        # Expected consensus sequence: either "ATCG" or "GTCG"
        assert consensus_seq.seq in ["ATCG", "GTCG"]

    def test_consensus_empty_alignment(self):
        """Test the consensus sequence generation on an empty alignment."""
        alignment = sq.Seq_alignment(name="alignment")

        with pytest.raises(IndexError):
            alignment.consensus()

    def test_consensus_all_gaps(self):
        """Test the consensus sequence when all sequences are gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="----")
        seq2 = sq.DNA_sequence(name="seq2", seq="----")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Compute consensus sequence
        consensus_seq = alignment.consensus()

        # Expected consensus sequence: "----" (all gaps)
        assert consensus_seq.seq == "----"
                
###################################################################################################

class Test_Seq_alignment_seqset:
    """Test suite for the seqset method in Seq_alignment."""

    def test_seqset_basic(self):
        """Test removing gaps and returning a Seq_set object for a basic alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AT-CG")
        seq2 = sq.DNA_sequence(name="seq2", seq="A-TGC")
        seq3 = sq.DNA_sequence(name="seq3", seq="ATG-C")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Convert alignment to Seq_set (removing gaps)
        seqset = alignment.seqset()

        # Check that the result is a Seq_set object
        assert isinstance(seqset, sq.Seq_set)

        # Check the names and sequences in the Seq_set object
        assert seqset.getseq("seq1").seq == "ATCG"
        assert seqset.getseq("seq2").seq == "ATGC"
        assert seqset.getseq("seq3").seq == "ATGC"

    def test_seqset_with_multiple_gaps(self):
        """Test removing multiple gaps from sequences and returning a Seq_set object."""
        seq1 = sq.DNA_sequence(name="seq1", seq="A--T-CG")
        seq2 = sq.DNA_sequence(name="seq2", seq="-A-TG-C")
        seq3 = sq.DNA_sequence(name="seq3", seq="--AT-GC")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Convert alignment to Seq_set (removing gaps)
        seqset = alignment.seqset()

        # Check that the result is a Seq_set object
        assert isinstance(seqset, sq.Seq_set)

        # Check the names and sequences in the Seq_set object
        assert seqset.getseq("seq1").seq == "ATCG"
        assert seqset.getseq("seq2").seq == "ATGC"
        assert seqset.getseq("seq3").seq == "ATGC"

    def test_seqset_no_gaps(self):
        """Test returning a Seq_set object when there are no gaps in the alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="ATGC")
        seq3 = sq.DNA_sequence(name="seq3", seq="ATGG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)
        alignment.addseq(seq3)

        # Convert alignment to Seq_set (removing gaps)
        seqset = alignment.seqset()

        # Check that the result is a Seq_set object
        assert isinstance(seqset, sq.Seq_set)

        # Check the names and sequences in the Seq_set object remain unchanged
        assert seqset.getseq("seq1").seq == "ATCG"
        assert seqset.getseq("seq2").seq == "ATGC"
        assert seqset.getseq("seq3").seq == "ATGG"

    def test_seqset_all_gaps(self):
        """Test returning a Seq_set object when sequences consist entirely of gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="----")
        seq2 = sq.DNA_sequence(name="seq2", seq="----")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Convert alignment to Seq_set (removing gaps)
        seqset = alignment.seqset()

        # Check that the result is a Seq_set object
        assert isinstance(seqset, sq.Seq_set)

        # Check that the sequences in the Seq_set object are empty after removing gaps
        assert seqset.getseq("seq1").seq == ""
        assert seqset.getseq("seq2").seq == ""

    def test_seqset_empty_alignment(self):
        """Test returning a Seq_set object when the alignment is empty."""
        alignment = sq.Seq_alignment(name="empty_alignment")

        # Convert alignment to Seq_set (removing gaps)
        seqset = alignment.seqset()

        # Check that the result is a Seq_set object
        assert isinstance(seqset, sq.Seq_set)

        # Check that the Seq_set object is empty
        assert len(seqset) == 0
        
###################################################################################################

class Test_Seq_alignment_partitions_as_seqalignments:
    """Test suite for the partitions_as_seqalignments method in Seq_alignment."""

    def test_partitions_as_seqalignments_basic(self):
        """Test creating Seq_alignment objects for partitions in a basic alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG--TG")
        seq2 = sq.DNA_sequence(name="seq2", seq="AT-GCTAG")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Manually set partitions (for testing purposes)
        alignment.partitions = [("Partition_01", 0, 4, "DNA"), ("Partition_02", 4, 4, "DNA")]

        # Retrieve partitions as Seq_alignment objects
        partitions = alignment.partitions_as_seqalignments()

        # Check that the partitions are correctly returned as Seq_alignment objects
        assert len(partitions) == 2
        assert isinstance(partitions[0], sq.Seq_alignment)
        assert isinstance(partitions[1], sq.Seq_alignment)
        
        # Check the content of the first partition
        assert partitions[0].name == "Partition_01"
        assert partitions[0].alignlen() == 4
        assert partitions[0].getseq("seq1_0_4").seq == "ATCG"
        assert partitions[0].getseq("seq2_0_4").seq == "AT-G"

        # Check the content of the second partition
        assert partitions[1].name == "Partition_02"
        assert partitions[1].alignlen() == 4
        assert partitions[1].getseq("seq1_4_8").seq == "--TG"
        assert partitions[1].getseq("seq2_4_8").seq == "CTAG"

    def test_partitions_as_seqalignments_with_one_partition(self):
        """Test behavior when there are no partitions defined in the alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq2 = sq.DNA_sequence(name="seq2", seq="TAGC")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Retrieve partitions as Seq_alignment objects
        partitions = alignment.partitions_as_seqalignments()

        # Check that there is one partition correctly returned as Seq_alignment object
        assert len(partitions) == 1
        assert isinstance(partitions[0], sq.Seq_alignment)

        # Check the content of the first partition
        assert partitions[0].name == "alignment"
        assert partitions[0].alignlen() == 4
        assert partitions[0].getseq("seq1_0_4").seq == "ATCG"
        assert partitions[0].getseq("seq2_0_4").seq == "TAGC"

    def test_partitions_as_seqalignments_with_unnamed_partitions(self):
        """Test creating Seq_alignment objects for unnamed partitions in an alignment."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="TAGC--TA")
        alignment = sq.Seq_alignment(name="alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Manually set partitions with None or "Partition" names
        alignment.partitions = [(None, 0, 4, "DNA"), ("Partition", 4, 4, "DNA")]

        # Retrieve partitions as Seq_alignment objects
        partitions = alignment.partitions_as_seqalignments()

        # Check that the partitions are correctly returned as Seq_alignment objects
        assert len(partitions) == 2
        assert isinstance(partitions[0], sq.Seq_alignment)
        assert isinstance(partitions[1], sq.Seq_alignment)

        # Check the names of the partitions
        assert partitions[0].name == "partition_01"
        assert partitions[1].name == "partition_02"

    def test_partitions_as_seqalignments_large_alignment(self):
        """Test creating Seq_alignment objects for a large alignment with multiple partitions."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCGGCTAGCTA")
        seq2 = sq.DNA_sequence(name="seq2", seq="TAGC--TACGTA")
        alignment = sq.Seq_alignment(name="large_alignment")
        alignment.addseq(seq1)
        alignment.addseq(seq2)

        # Manually set partitions for a large alignment
        alignment.partitions = [("Partition_01", 0, 4, "DNA"), ("Partition_02", 4, 4, "DNA"), ("Partition_03", 8, 4, "DNA")]

        # Retrieve partitions as Seq_alignment objects
        partitions = alignment.partitions_as_seqalignments()

        # Check that the partitions are correctly returned as Seq_alignment objects
        assert len(partitions) == 3
        assert isinstance(partitions[0], sq.Seq_alignment)
        assert isinstance(partitions[1], sq.Seq_alignment)
        assert isinstance(partitions[2], sq.Seq_alignment)

        # Check the content of each partition
        assert partitions[0].name == "Partition_01"
        assert partitions[0].getseq("seq1_0_4").seq == "ATCG"
        assert partitions[0].getseq("seq2_0_4").seq == "TAGC"
        assert partitions[1].name == "Partition_02"
        assert partitions[1].getseq("seq1_4_8").seq == "GCTA"
        assert partitions[1].getseq("seq2_4_8").seq == "--TA"
        assert partitions[2].name == "Partition_03"
        assert partitions[2].getseq("seq1_8_12").seq == "GCTA"
        assert partitions[2].getseq("seq2_8_12").seq == "CGTA"
        
###################################################################################################

class Test_Seq_alignment_distdict:
    """Test suite for the distdict method in Seq_alignment."""

    def setup_method(self):
        """Setup method to create a base Seq_alignment object for testing."""
        self.seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        self.seq2 = sq.DNA_sequence(name="seq2", seq="ATGG")
        self.seq3 = sq.DNA_sequence(name="seq3", seq="TTGG")
        self.alignment = sq.Seq_alignment(name="alignment")
        self.alignment.addseq(self.seq1)
        self.alignment.addseq(self.seq2)
        self.alignment.addseq(self.seq3)

    def test_distdict_default_pdist(self):
        """Test distdict with the default pdist method."""
        dist_matrix = self.alignment.distdict()

        # Check the structure of the distdict output
        assert isinstance(dist_matrix, dict)
        assert len(dist_matrix) == 3
        assert all(len(dist_matrix[key]) == 3 for key in dist_matrix)
        
        # Check that diagonal elements are None (no self-comparison)
        assert dist_matrix["seq1"]["seq1"] is None
        assert dist_matrix["seq2"]["seq2"] is None
        assert dist_matrix["seq3"]["seq3"] is None

        # Check the values in the distance matrix using the pdist method (default)
        assert dist_matrix["seq1"]["seq2"] == pytest.approx(1 / 4)  # 1 mismatch out of 4
        assert dist_matrix["seq1"]["seq3"] == pytest.approx(2 / 4)  # 2 mismatches out of 4
        assert dist_matrix["seq2"]["seq3"] == pytest.approx(1 / 4)  # 1 mismatch out of 4

    def test_distdict_hamming(self):
        """Test distdict with the hamming distance method."""
        dist_matrix = self.alignment.distdict(dist="hamming")

        # Check the structure of the distdict output
        assert isinstance(dist_matrix, dict)
        assert len(dist_matrix) == 3
        assert all(len(dist_matrix[key]) == 3 for key in dist_matrix)

        # Check the values in the distance matrix using the hamming method
        assert dist_matrix["seq1"]["seq2"] == 1  # 1 mismatch
        assert dist_matrix["seq1"]["seq3"] == 2  # 2 mismatches
        assert dist_matrix["seq2"]["seq3"] == 1  # 1 mismatch

    def test_distdict_hamming_ignoregaps(self):
        """Test distdict with the hamming_ignoregaps distance method."""
        seq4 = sq.DNA_sequence(name="seq4", seq="A-CT")
        alignment_with_gaps = sq.Seq_alignment(name="alignment_with_gaps")
        alignment_with_gaps.addseq(self.seq1)
        alignment_with_gaps.addseq(seq4)

        dist_matrix = alignment_with_gaps.distdict(dist="hamming_ignoregaps")

        # Check the structure of the distdict output
        assert isinstance(dist_matrix, dict)
        assert len(dist_matrix) == 2
        assert all(len(dist_matrix[key]) == 2 for key in dist_matrix)

        # Check the values in the distance matrix ignoring gaps
        assert dist_matrix["seq1"]["seq4"] == 1  # 1 mismatch (gaps ignored)
        assert dist_matrix["seq4"]["seq1"] == 1  # 1 mismatch (gaps ignored)

    def test_distdict_pdist_ignoregaps(self):
        """Test distdict with the pdist_ignoregaps distance method."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        seq4 = sq.DNA_sequence(name="seq4", seq="A-CT")
        alignment_with_gaps = sq.Seq_alignment(name="alignment_with_gaps")
        alignment_with_gaps.addseq(self.seq1)
        alignment_with_gaps.addseq(seq4)

        dist_matrix = alignment_with_gaps.distdict(dist="pdist_ignoregaps")

        # Check the structure of the distdict output
        assert isinstance(dist_matrix, dict)
        assert len(dist_matrix) == 2
        assert all(len(dist_matrix[key]) == 2 for key in dist_matrix)

        # Check the values in the distance matrix ignoring gaps
        assert dist_matrix["seq1"]["seq4"] == 1 / 3  # 1 mismatch on 3 positions (gaps ignored)
        assert dist_matrix["seq4"]["seq1"] == 1 / 3  # 1 mismatch on 3 positions (gaps ignored)

    def test_distdict_unknown_method(self):
        """Test distdict with an unknown distance method, should raise an error."""
        with pytest.raises(sq.SeqError, match="Unknown distance measure: unknown"):
            self.alignment.distdict(dist="unknown")

    def test_distdict_single_sequence(self):
        """Test distdict with only one sequence in the alignment, should return a dict with None values."""
        single_seq_alignment = sq.Seq_alignment(name="single_seq_alignment")
        single_seq_alignment.addseq(self.seq1)
        dist_matrix = single_seq_alignment.distdict()

        # Check the structure of the distdict output
        assert isinstance(dist_matrix, dict)
        assert len(dist_matrix) == 1
        assert len(dist_matrix["seq1"]) == 1

        # Check that all values are None (since there's no other sequence to compare)
        assert dist_matrix["seq1"]["seq1"] is None
        
###################################################################################################

class Test_Seq_alignment_sequence_diversity:
    """Test suite for the sequence_diversity method in Seq_alignment."""

    def setup_method(self):
        """Setup method to create a base Seq_alignment object for testing."""
        self.seq1 = sq.DNA_sequence(name="seq1", seq="ATCG")
        self.seq2 = sq.DNA_sequence(name="seq2", seq="ATGG")
        self.seq3 = sq.DNA_sequence(name="seq3", seq="TTGG")
        self.alignment = sq.Seq_alignment(name="alignment")
        self.alignment.addseq(self.seq1)
        self.alignment.addseq(self.seq2)
        self.alignment.addseq(self.seq3)

    def test_sequence_diversity_default(self):
        """Test sequence diversity with default settings (considering gaps)."""
        mean, std, minpi, maxpi = self.alignment.sequence_diversity()

        # Calculate expected mean and std using pairwise pdist
        expected_distances = [
            sq.Sequence.pdist(self.seq1, self.seq2),  # 1 mismatch out of 4
            sq.Sequence.pdist(self.seq1, self.seq3),  # 2 mismatches out of 4
            sq.Sequence.pdist(self.seq2, self.seq3)   # 1 mismatch out of 4
        ]
        expected_mean = sum(expected_distances) / len(expected_distances)
        expected_variance = sum((d - expected_mean) ** 2 for d in expected_distances) / len(expected_distances)
        expected_std = math.sqrt(expected_variance)

        assert mean == pytest.approx(expected_mean, rel=1e-5)
        assert std == pytest.approx(expected_std, rel=1e-5)
        assert minpi == pytest.approx(min(expected_distances), rel=1e-5)
        assert maxpi == pytest.approx(max(expected_distances), rel=1e-5)

    def test_sequence_diversity_ignoregaps(self):
        """Test sequence diversity while ignoring gaps."""
        seq4 = sq.DNA_sequence(name="seq4", seq="A-CG")
        alignment_with_gaps = sq.Seq_alignment(name="alignment_with_gaps")
        alignment_with_gaps.addseq(self.seq1)
        alignment_with_gaps.addseq(seq4)

        mean, std, minpi, maxpi = alignment_with_gaps.sequence_diversity(ignoregaps=True)

        # Calculate expected mean and std using pairwise pdist_ignoregaps
        expected_distances = [
            sq.Sequence.pdist_ignoregaps(self.seq1, seq4)  # No mismatches when ignoring gaps
        ]
        expected_mean = sum(expected_distances) / len(expected_distances)
        expected_variance = sum((d - expected_mean) ** 2 for d in expected_distances) / len(expected_distances)
        expected_std = math.sqrt(expected_variance)

        assert mean == pytest.approx(expected_mean, rel=1e-5)
        assert std == pytest.approx(expected_std, rel=1e-5)
        assert minpi == pytest.approx(min(expected_distances), rel=1e-5)
        assert maxpi == pytest.approx(max(expected_distances), rel=1e-5)

    def test_sequence_diversity_single_sequence(self):
      """Test that sequence diversity raises an error with only one sequence."""
      single_seq_alignment = sq.Seq_alignment(name="single_seq_alignment")
      single_seq_alignment.addseq(self.seq1)

      # Expect an error when trying to compute diversity with a single sequence
      with pytest.raises(sq.SeqError, match="Can't compute diversity for alignment with less than 2 sequences"):
          single_seq_alignment.sequence_diversity()

    def test_sequence_diversity_no_sequences(self):
        """Test sequence diversity with an empty alignment (should return zeros for mean, std, minpi, maxpi)."""
        empty_alignment = sq.Seq_alignment(name="empty_alignment")

        # Expect an error when trying to compute diversity with no sequences
        with pytest.raises(sq.SeqError, match="Can't compute diversity for alignment with less than 2 sequences"):
          empty_alignment.sequence_diversity()

###################################################################################################

class Test_Seq_alignment_overlap:
    """Test suite for the overlap method in Seq_alignment."""

    def test_overlap_no_gaps(self):
        """Test overlap calculation with no gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ACGT")
        seq2 = sq.DNA_sequence(name="seq2", seq="AGTT")
        alignment1 = sq.Seq_alignment(name="alignment1")
        alignment1.addseq(seq1)
        alignment1.addseq(seq2)

        seq3 = sq.DNA_sequence(name="seq1", seq="ACGT")
        seq4 = sq.DNA_sequence(name="seq2", seq="AGTT")
        alignment2 = sq.Seq_alignment(name="alignment2")
        alignment2.addseq(seq3)
        alignment2.addseq(seq4)

        overlap_fraction = alignment1.overlap(alignment2)

        assert overlap_fraction == pytest.approx(1.0, rel=1e-5)  # Expect full overlap

    def test_overlap_with_gaps(self):
        """Test overlap calculation with single gap."""
        seq1 = sq.DNA_sequence(name="seq1", seq="AAAAT")
        seq2 = sq.DNA_sequence(name="seq2", seq="AA-AC")
        alignment1 = sq.Seq_alignment(name="alignment1")
        alignment1.addseq(seq1)
        alignment1.addseq(seq2)

        seq3 = sq.DNA_sequence(name="seq1", seq="AAAAT")
        seq4 = sq.DNA_sequence(name="seq2", seq="A-AAC")
        alignment2 = sq.Seq_alignment(name="alignment2")
        alignment2.addseq(seq3)
        alignment2.addseq(seq4)

        overlap_fraction = alignment1.overlap(alignment2)

        # 2/4 alignment columns agree
        assert overlap_fraction == pytest.approx(0.6, rel=1e-5)

    def test_overlap_no_overlap(self):
        """Test overlap when there is no alignment overlap."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ACGT")
        seq2 = sq.DNA_sequence(name="seq2", seq="TGCA")
        alignment1 = sq.Seq_alignment(name="alignment1")
        alignment1.addseq(seq1)
        alignment1.addseq(seq2)

        seq3 = sq.DNA_sequence(name="seq1", seq="ACGT-")
        seq4 = sq.DNA_sequence(name="seq2", seq="-TGCA")
        alignment2 = sq.Seq_alignment(name="alignment2")
        alignment2.addseq(seq3)
        alignment2.addseq(seq4)

        overlap_fraction = alignment1.overlap(alignment2)

        assert overlap_fraction == pytest.approx(0.0, rel=1e-5)  # No overlap

    def test_overlap_different_sequence_names(self):
        """Test overlap with alignments having different sequence names."""
        seq1 = sq.DNA_sequence(name="seq1", seq="ACGT")
        seq2 = sq.DNA_sequence(name="seq3", seq="TGCA")
        alignment1 = sq.Seq_alignment(name="alignment1")
        alignment1.addseq(seq1)
        alignment1.addseq(seq2)

        seq3 = sq.DNA_sequence(name="seq1", seq="TGCA")
        seq4 = sq.DNA_sequence(name="seq4", seq="ACGT")
        alignment2 = sq.Seq_alignment(name="alignment2")
        alignment2.addseq(seq3)
        alignment2.addseq(seq4)

        with pytest.raises(sq.SeqError, match="Alignments do not contain same sequences - not possible to compute overlap"):
            alignment1.overlap(alignment2)

    def test_overlap_multiple_gaps(self):
        """Test overlap when alignments contain all gaps."""
        seq1 = sq.DNA_sequence(name="seq1", seq="---A")
        seq2 = sq.DNA_sequence(name="seq2", seq="---T")
        alignment1 = sq.Seq_alignment(name="alignment1")
        alignment1.addseq(seq1)
        alignment1.addseq(seq2)

        seq3 = sq.DNA_sequence(name="seq1", seq="--A-")
        seq4 = sq.DNA_sequence(name="seq2", seq="--T-")
        alignment2 = sq.Seq_alignment(name="alignment2")
        alignment2.addseq(seq3)
        alignment2.addseq(seq4)

        overlap_fraction = alignment1.overlap(alignment2)

        # Expect overlap to be 0.5 as gaps are treated as residues when discardgaps is False
        assert overlap_fraction == pytest.approx(0.5, rel=1e-5)

###################################################################################################

class Test_Seq_alignment_phylip:
    """Test suite for the phylip method in Seq_alignment."""

    def setup_method(self):
        """Setup method to create a base Seq_alignment object for testing."""
        self.seq1 = sq.DNA_sequence(name="seq1", seq="ACGTACGTACGT")
        self.seq2 = sq.DNA_sequence(name="seq2", seq="TGCATGCATGCA")
        
        self.alignment = sq.Seq_alignment(name="test_alignment")
        self.alignment.addseq(self.seq1)
        self.alignment.addseq(self.seq2)

    def test_phylip_default_width(self):
        """Test PHYLIP formatting with default width."""
        phylip_output = self.alignment.phylip()
        expected_output = (
            "2   12\n"
            "seq1  ACGTACGTACGT\n"
            "seq2  TGCATGCATGCA"
        )

        assert phylip_output == expected_output

    def test_phylip_custom_width(self):
        """Test PHYLIP formatting with a custom width."""

        expected_output = (
            "2   12\n"
            "seq1  ACGT\n"
            "seq2  TGCA\n"
            "\n"
            "      ACGT\n"
            "      TGCA\n"
            "\n"
            "      ACGT\n"
            "      TGCA"
        )
        phylip_output = self.alignment.phylip(width=4)
        assert phylip_output == expected_output

    def test_phylip_full_width(self):
        """Test PHYLIP formatting with full width (no line breaks)."""
        expected_output = (
            "2   12\n"
            "seq1  ACGTACGTACGT\n"
            "seq2  TGCATGCATGCA"
        )
        phylip_output = self.alignment.phylip(width=12)
        assert phylip_output == expected_output

    def test_phylip_empty_alignment(self):
        """Test PHYLIP formatting raises an error for empty alignment."""
        empty_alignment = sq.Seq_alignment(name="empty_alignment")
        with pytest.raises(sq.SeqError, match="No sequences in sequence set.  Can't create phylip"):
            empty_alignment.phylip()



