# sequencelib

![](https://img.shields.io/badge/version-2.23.2-blue)
[![PyPI downloads](https://static.pepy.tech/personalized-badge/sequencelib?period=total&units=none&left_color=black&right_color=blue&left_text=PyPI%20downloads&service=github)](https://pepy.tech/project/sequencelib)

Using the classes and methods in sequencelib.py, you can read and write text files containing DNA or protein sequences (aligned or unaligned), and analyze or manipulate these sequences in various ways

**Note: ** Much of the functionality in `sequencelib` is also available through the command-line tool [seqconverter](https://github.com/agormp/seqconverter)

## Availability

The sequencelib.py module is available on GitHub: https://github.com/agormp/sequencelib and on PyPI: https://pypi.org/project/sequencelib/

## Installation

```
python3 -m pip install sequencelib
```

Upgrading to latest version:

```
python3 -m pip install --upgrade sequencelib
```


## Quick Start Tutorial for sequencelib

**Note: under construction. This version mostly generated using chatGPT with some editing**

This quick start guide introduces some basic functionalities of `sequencelib`.

### Loading Sequences

`sequencelib` supports various file formats: `fasta`, `nexus`, `clustal`, `phylip`, `raw`, `tab`, `how`, and `genbank`. It automatically detects the file format:

### Reading Unaligned Sequences

```python
import sequencelib as sq

seqfile = sq.Seqfile("seqfilename.fasta")
seqset = seqfile.read_seqs()
```

### Iterate over sequences
```python
for seq in seqset:
    print(seq.name, len(seq))
```

### Reading Aligned Sequences

```python
import sequencelib as sq

seqfile = sq.Seqfile("alignment.fasta")
alignment = seqfile.read_alignment()

print("Number of sequences:", len(alignment))
print("Alignment length:", alignment.alignlen())
```


### Find Columns with More than 50% Gaps

```python
nseqs = len(alignment)
gapcols = []
for i in range(alignment.alignlen()):
    col = alignment.getcolumn(i)
    gapfrac = col.count("-") / nseqs
    if gapfrac >= 0.5:
        gapcols.append(i)
```

### Export Alignment to File

```python
with open("gapcols.fasta", "w") as f:
    f.write(subalignment.fasta())

with open("gapcols.nexus", "w") as f:
    f.write(subalignment.nexus())

with open("gapcols.clustal", "w") as f:
    f.write(subalignment.clustal())
```

### Analyzing Individual Columns

Directly access columns and analyze their conservation:

```python
column = subalignment.getcolumn(0)
if len(set(column)) > 1:
    print("This column is not conserved")
```

### Mapping Sequence and Alignment Positions

Map positions between sequence (without gaps) and alignment:

```python
alignpos_0index = alignment.seqpos2alignpos("seq1", 41)  # Index starts at 0
alignpos_1index = alignment.seqpos2alignpos("seq1", 42, slicesyntax=False) # Index starts at 1
```

Convert from alignment position to sequence position:

```python
seqpos, gapstatus = alignment.alignpos2seqpos("seq1", 153)
if gapstatus:
    print(f"Alignment position is a gap; closest preceding residue is at sequence position {seqpos}")
```

### Working with Individual Sequences

Each sequence object has multiple attributes and methods:

```python
seq = seqset[0]
print(seq.name)
print(len(seq))
print(seq.fasta())

shuffled_seq = seq.shuffle()
protein_seq = seq.translate()
```

### Window Iteration

Iterate through sequence windows:

```python
for seqwindow in seq.windows(wsize=30):
    print(seqwindow.fasta())
```

### More Features

The `sequencelib` library contains many additional functionalities such as:

- Calculating pairwise sequence distances
- Removing conserved or ambiguous columns
- Reverse complementing DNA sequences
- Handling complex alignments with partitions


## SequenceLib: Class and Method Reference

---

### Class: `Sequence`
Base class representing a biological sequence (DNA, protein, or other types).

#### Constructor
```python
Sequence(name, seq, annotation='', comments='', check_alphabet=False, degap=False)
```

- **name**: Identifier for the sequence.
- **seq**: The actual biological sequence string.
- **annotation**: Annotation information for each residue.
- **comments**: Additional metadata or notes.
- **check_alphabet**: Checks sequence against allowed alphabet symbols.
- **degap**: Removes gap characters (`-`).

#### Methods

- `__len__()`: Returns the length of the sequence.
- `__getitem__(index)`: Allows indexing and slicing of the sequence.
- `__setitem__(index, residue)`: Modifies residue at a given index.
- `__str__()`: Returns FASTA-formatted string.
- `copy_seqobject()`: Returns a deep copy of the sequence object.
- `rename(newname)`: Changes the sequence name.
- `subseq(start, stop, slicesyntax=True, rename=False)`: Extracts subsequence between start and stop positions.
- `subseqpos(poslist, namesuffix=None)`: Creates subsequence from specified positions.
- `appendseq(other)`: Appends another sequence at the end.
- `prependseq(other)`: Prepends another sequence at the start.
- `windows(wsize, stepsize=1, l_overhang=0, r_overhang=0, padding="X", rename=False)`: Iterates over windows of the sequence.
- `remgaps()`: Removes gaps from the sequence.
- `shuffle()`: Randomly shuffles the sequence residues.
- `indexfilter(keeplist)`: Keeps only residues at specified positions.
- `seqdiff(other, zeroindex=True)`: Lists differences between two sequences.
- `hamming(other)`: Computes Hamming distance (absolute differences).
- `hamming_ignoregaps(other)`: Computes Hamming distance, ignoring gaps.
- `pdist(other)`: Computes proportional differences per site.
- `pdist_ignoregaps(other)`: Computes proportional differences, ignoring gaps.
- `pdist_ignorechars(other, igchars)`: Proportional differences ignoring specified characters.
- `residuecounts()`: Counts residues and returns a dictionary.
- `composition(ignoregaps=True, ignoreambig=False)`: Calculates composition as frequencies.
- `findgaps()`: Identifies gap positions.
- `fasta(width=60, nocomments=False)`: Returns FASTA format representation.
- `how(width=80, nocomments=False)`: Returns HOW format representation.
- `gapencoded()`: Encodes gaps as binary (1/0) string.
- `tab(nocomments=False)`: Returns TAB format representation.
- `raw()`: Returns sequence in raw format.

---

### Class: `DNA_sequence(Sequence)`
Specialized sequence class for DNA sequences.

#### Methods
- `revcomp()`: Returns reverse complement.
- `translate(reading_frame=1)`: Translates DNA to protein sequence.

---

### Class: `Protein_sequence(Sequence)`
Specialized sequence class for protein sequences. Inherits directly from `Sequence`.

---

### Class: `Sequences_base`
Abstract base class for sequence collections. Should not be instantiated directly. All methods here can be used in both Seq_alignment and Seq_set objects.

#### Methods

- `__len__()`: Returns the number of sequences.
- `__getitem__(index)`: Accesses sequences via indexing or slicing.
- `__setitem__(index, value)`: Sets sequences by integer index.
- `__eq__(other)`: Checks equality with another sequence collection.
- `__ne__(other)`: Checks inequality with another sequence collection.
- `__str__()`: Returns FASTA format of the collection.
- `sortnames(reverse=False)`: Alphabetically sorts sequences by name.
- `addseq(seq, silently_discard_dup_name=False)`: Adds a sequence object.
- `addseqset(other, silently_discard_dup_name=False)`: Adds sequences from another collection.
- `remseq(name)`: Removes sequence by name.
- `remseqs(namelist)`: Removes multiple sequences.
- `changeseqname(oldname, newname, fix_dupnames=False)`: Renames a sequence.
- `getseq(name)`: Retrieves sequence by name.
- `subset(namelist)`: Extracts a subset by names.
- `subsample(samplesize)`: Randomly selects a subset.
- `subseq(start, stop, slicesyntax=True, rename=True, aln_name=None, aln_name_number=False)`: Extracts subset by positions.
- `getnames()`: Returns a list of sequence names.
- `range(rangefrom, rangeto)`: In-place subset of sequences.
- `removedupseqs()`: Removes duplicate sequences.
- `group_identical_seqs()`: Groups identical sequences.
- `residuecounts()`: Counts residues across all sequences.
- `composition(ignoregaps=True, ignoreambig=False)`: Computes frequency composition.
- `clean_names(illegal=":;,()[]", rep="_")`: Cleans illegal characters from names.
- `rename_numbered(basename, namefile=None)`: Renames sequences numerically.
- `rename_regexp(old_regex, new_string, namefile=None)`: Renames sequences using regex.
- `transname(namefile)`: Renames sequences using a mapping file.
- `revcomp()`: Reverse complements all sequences.
- `translate(reading_frame=1)`: Translates all sequences (DNA only).
- `fasta(width=60, nocomments=False)`: FASTA format.
- `how(width=60, nocomments=False)`: HOW format.
- `tab(nocomments=False)`: TAB format.
- `raw()`: RAW format.

---

### Class: `Seq_alignment(Sequences_base)`
Represents aligned sequences. This class also has access to all methods defined in parent class (Sequences_base).

#### Methods

- `alignlen()`: Length of the alignment.
- `getcolumn(i)`: Retrieves column by index.
- `columns()`: Iterates over columns.
- `samplecols(samplesize)`: Randomly samples columns.
- `conscols()`: Lists conserved columns.
- `varcols()`: Lists variable columns.
- `gappycols()`: Lists columns with gaps.
- `site_summary()`: Summarizes alignment sites.
- `indexfilter(keeplist)`: Keeps columns by indices.
- `remcols(discardlist)`: Removes columns by indices.
- `remambigcol()`: Removes ambiguous columns.
- `remfracambigcol(frac)`: Removes columns with high ambiguity fraction.
- `remgapcol()`: Removes columns with gaps.
- `remfracgapcol(frac)`: Removes columns with high gap fraction.
- `remendgapcol(frac=0.5)`: Removes end-gap columns.
- `remconscol()`: Removes conserved columns.
- `findgaps()`: Identifies gap positions.
- `gap_encode()`: Binary encodes gaps.
- `seqpos2alignpos(seqname, seqpos, slicesyntax=True)`: Maps sequence to alignment position.
- `alignpos2seqpos(seqname, alignpos, slicesyntax=True)`: Maps alignment to sequence position.
- `shannon(countgaps=True)`: Computes Shannon entropy.
- `consensus()`: Generates consensus sequence.
- `phylip(width=60)`: PHYLIP format.
- `clustal(width=60)`: CLUSTAL format.
- `nexus(width=60, print_partitioned=False)`: NEXUS format.
- `charsetblock()`: Generates MrBayes charset block for partitioned analyses.
- `mbpartblock()`: Generates detailed MrBayes block (charset, partitions, models, MCMC) for partitioned analyses.
- `bestblock()`: Generates MrBayes BEST block for species-tree analyses (taxsets, charsets, BEST parameters).
- `nexuspart()`: Generates Nexus-formatted MrBayes block with partition and model specifications.


---

### Class `Seqfile_reader`

Base class for reading sequence files. Typically, you do not instantiate this class directly.

#### Methods:

- `makeseq(name, seq, annotation="", comments="")`
  - **Description:** Creates and returns a sequence object based on provided type information.

- `readseq()`
  - **Description:** Reads a single sequence from a file and returns it as a sequence object.

- `read_seqs(silently_discard_dup_name=False)`
  - **Description:** Reads all sequences and returns a `Seq_set` object.

- `read_alignment(silently_discard_dup_name=False)`
  - **Description:** Reads aligned sequences, returning a `Seq_alignment` object.

---

### Class `Fastafilehandle`

Class for handling FASTA files.

#### Methods:

- `__init__(filename, seqtype="autodetect", check_alphabet=False, degap=False, nameishandle=False)`
  - **Description:** Initializes a FASTA file reader, performs format checks.

- `__next__()`
  - **Description:** Parses and returns the next sequence as a sequence object.

---

### Class `Howfilehandle`

Class for reading HOW-formatted files.

#### Methods:

- `__init__(...)`
- `__next__()`

---

### Class `Genbankfilehandle`

Class for reading GenBank files.

#### Methods:

- `__init__(...)`
- `__next__()`
- `find_LOCUS()`
- `read_metadata()`
- `extract_annotation(metadata)`
- `extract_name(metadata)`
- `read_genbankseq()`

---

### Class `Tabfilehandle`

Handles tab-delimited sequence files.

#### Methods:

- `__init__(...)`
- `__next__()`

---

### Class `Rawfilehandle`

Handles raw-format sequence files.

#### Methods:

- `__init__(...)`
- `__next__()`

---

### Class `Alignfile_reader`

Base class for alignment files.

#### Methods:

- `makeseq(name, seq, annotation="", comments="")`
- `read_seqs(silently_discard_dup_name=False)`

---

### Class `Clustalfilehandle`

Reads Clustal-formatted alignment files.

#### Methods:

- `__init__(...)`
- `read_alignment(silently_discard_dup_name=False)`

---

### Class `Phylipfilehandle`

Handles Phylip-formatted alignment files.

#### Methods:

- `__init__(...)`
- `read_alignment(silently_discard_dup_name=False)`

---

### Class `Nexusfilehandle`

Handles Nexus-formatted alignment files.

#### Methods:

- `__init__(...)`
- `read_alignment(silently_discard_dup_name=False)`

---

### Class `Stockholmfilehandle`

Reads Stockholm-formatted alignment files.

#### Methods:

- `__init__(...)`
- `read_alignment(silently_discard_dup_name=False)`

---

### Class `Seqfile`

Factory class to autodetect file formats and instantiate the correct file handler.

#### Methods:

- `__new__(klass, filename, filetype="autodetect", ...)`

Automatically selects the appropriate sequence or alignment reader based on file contents or explicitly provided file type.



