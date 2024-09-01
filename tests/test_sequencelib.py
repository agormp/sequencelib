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
