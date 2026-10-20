Run the reader regression tests from the repository root:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests/test_deferred_methylation.py
```

The tests construct real BAM and CRAM files with pysam, including a local CRAM
reference, so no downloads or aligner are required. They compare extraction
against the eager public reader interface, including output order, read-length
arrays, methylation arrays, missing tags, strand flags, supplementary and
secondary reads, mixed FASTA/FASTQ inputs, file closure, queue draining, and
both batching limits. The queue test checks retained-record bounds independently
of the number of input records. Performance measurements belong in a separate
benchmark; these tests do not assert wall-clock thresholds.
