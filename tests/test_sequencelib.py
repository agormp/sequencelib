import pytest
import sequencelib as sq
import random
import re
import numpy as np
import collections

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
            assert pdist == (nmut - ngaps) / self.seqlen

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
###################################################################################################

