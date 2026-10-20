"""Real BAM/CRAM fixtures for sequence screening before methylation decoding."""
from array import array
import gzip
from pathlib import Path

import numpy as np
import pysam
import pytest

from source.tg_reader import TG_Reader, extract_telomere_reads

KMER, REV_KMER, MIN_HITS = 'TTAGGG', 'CCCTAA', 2
SELECTED = 'ACCGTCCG' + KMER * 3
REJECTED = 'ACCGTCAG' * 4


def alignment(name, sequence, flag=0, tags='both'):
    aln = pysam.AlignedSegment()
    aln.query_name = name
    aln.query_sequence = sequence
    aln.flag = flag
    if sequence is not None:
        aln.reference_id = 0
        aln.reference_start = 0
        aln.cigar = [(0, len(sequence))]
        aln.mapping_quality = 60
        aln.query_qualities = pysam.qualitystring_to_array('I' * len(sequence))
    else:
        aln.flag |= 4
    if tags in ('both', 'mm_only'):
        aln.set_tag('MM', 'C+m,0,1;C+h,1;')
    if tags in ('both', 'ml_only'):
        aln.set_tag('ML', array('B', [0, 255, 128]))
    return aln


def write_alignments(path, records, ref=None):
    kwargs = {'reference_filename': str(ref)} if ref else {}
    with pysam.AlignmentFile(str(path), 'wc' if ref else 'wb',
                             header={'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 100000}]},
                             **kwargs) as output:
        for record in records:
            output.write(record)
    return str(path)


@pytest.fixture
def inputs(tmp_path):
    ref = tmp_path / 'reference.fa'
    ref.write_text('>chr1\n' + 'ACGT' * 25000 + '\n')
    pysam.faidx(str(ref))
    records = [
        alignment('forward', SELECTED),
        alignment('reverse', 'ACCGTCCG' + REV_KMER * 3, flag=16),
        alignment('supplementary', SELECTED, flag=2048),
        alignment('secondary', SELECTED, flag=256),
        alignment('no-tags', SELECTED, tags='none'),
        alignment('mm-only', SELECTED, tags='mm_only'),
        alignment('ml-only', SELECTED, tags='ml_only'),
        alignment('rejected', REJECTED),
        alignment('empty', None, tags='none'),
    ]
    bam = write_alignments(tmp_path / 'reads.bam', records)
    cram = write_alignments(tmp_path / 'reads.cram', records, ref)
    fasta = tmp_path / 'reads.fa'
    fasta.write_text(f'>fa-selected\n{SELECTED[:10]}\n{SELECTED[10:]}\n>fa-rejected\n{REJECTED}\n')
    fastq = tmp_path / 'reads.fq.gz'
    with gzip.open(fastq, 'wt') as output:
        output.write(f'@fq-selected\n{SELECTED}\n+\n' + 'I' * len(SELECTED) + '\n')
    return bam, cram, str(fasta), str(fastq), str(ref)


def eager_reference(paths, ref):
    """Independent reference using the unchanged, eager public reader API."""
    all_lens, selected_lens, names, methyl = [], [], [], {}
    selected = []
    supplementary = 0
    for path in paths:
        reader = TG_Reader(path, verbose=False, ref_fasta=ref)
        try:
            while True:
                name, seq, _, is_sup, decoded = reader.get_next_read(with_quality=False)
                if not name:
                    break
                if not seq:
                    continue
                if is_sup:
                    supplementary += 1
                    continue
                all_lens.append(len(seq))
                if seq.count(KMER) >= MIN_HITS or seq.count(REV_KMER) >= MIN_HITS:
                    selected.append(f'>{name}\n{seq}\n')
                    selected_lens.append(len(seq))
                    if decoded is not None:
                        idx = len(names)
                        names.append(name)
                        methyl[f'{idx}_5mc'], methyl[f'{idx}_5hmc'] = decoded
        finally:
            reader.close()
    result = (len(all_lens), len(selected_lens), supplementary,
              sum(all_lens), sum(selected_lens), all_lens, selected_lens, names, methyl)
    return result, ''.join(selected).encode()


@pytest.mark.parametrize('format_index', [0, 1])
@pytest.mark.parametrize('with_quality', [False, True])
def test_default_payload_and_decode_after_close(inputs, format_index, with_quality):
    filename, ref = inputs[format_index], inputs[-1]
    eager = TG_Reader(filename, verbose=False, ref_fasta=ref)
    deferred = TG_Reader(filename, verbose=False, ref_fasta=ref)
    eager_reads, deferred_reads = [], []
    while True:
        value = eager.get_next_read(with_quality=with_quality)
        other = deferred.get_next_read(with_quality=with_quality, defer_methylation=True)
        if not value[0]:
            assert value == other == ('', '', '', False, None)
            break
        eager_reads.append(value)
        deferred_reads.append(other)
    eager.close()
    deferred.close()
    assert eager_reads[0][4][0][1] == 0.0
    assert eager_reads[0][4][0][5] == 1.0
    assert eager_reads[0][4][1][2] == 128 / 255.0
    assert [read[3] for read in eager_reads][:4] == [False, False, True, False]
    for value, other in zip(eager_reads, deferred_reads):
        assert value[:4] == other[:4]
        if other[4] is None:
            assert value[4] is None
        else:
            assert value[4] == deferred.extract_methylation(other[4])
    assert all(read[4] is None for read in eager_reads[4:7])
    # Untagged, MM-only and ML-only records must not retain an alignment.
    assert all(read[4] is None for read in deferred_reads[4:7])


@pytest.mark.parametrize('processes', [1, 2, 4])
@pytest.mark.parametrize('input_indices', [[0], [1], [2], [3], [0, 2, 1, 3]])
def test_extraction_exact_parity(inputs, tmp_path, processes, input_indices):
    paths = [inputs[i] for i in input_indices]
    expected, expected_bytes = eager_reference(paths, inputs[-1])
    output = str(tmp_path / 'selected.fa.gz')
    actual = extract_telomere_reads(paths, output, KMER, REV_KMER, MIN_HITS,
                                    num_processes=processes, ref_fasta=inputs[-1])
    assert actual == expected
    assert gzip.open(output, 'rb').read() == expected_bytes
    # Match the actual caller's checkpoint dtypes exactly.
    for i in (5, 6):
        np.testing.assert_array_equal(np.array(actual[i], dtype='<i8'), np.array(expected[i], dtype='<i8'))
    for key in expected[8]:
        np.testing.assert_array_equal(np.array(actual[8][key], dtype='<f4'), np.array(expected[8][key], dtype='<f4'))


@pytest.mark.parametrize('processes', [1, 2, 4])
def test_only_selected_alignments_decoded(inputs, tmp_path, monkeypatch, processes):
    decoded_names = []
    original = TG_Reader.extract_methylation
    def observe(self, aln):
        decoded_names.append(aln.query_name)
        return original(self, aln)
    monkeypatch.setattr(TG_Reader, 'extract_methylation', observe)
    extract_telomere_reads([inputs[0], inputs[2]], str(tmp_path / 'out.gz'),
                           KMER, REV_KMER, MIN_HITS, num_processes=processes)
    assert decoded_names == ['forward', 'reverse', 'secondary']


@pytest.mark.parametrize('processes', [1, 2, 4])
@pytest.mark.parametrize('selected', [False, True])
def test_multiple_batches_and_empty_output(tmp_path, processes, selected):
    # More than 2 * workers batches exercises the pending-queue drain, then a
    # short final batch is decoded after closing the input file.
    records = [alignment(f'read-{i}', SELECTED if selected and i % 137 == 0 else REJECTED)
               for i in range(2311)]
    path = write_alignments(tmp_path / 'batches.bam', records)
    expected, expected_bytes = eager_reference([path], '')
    output = str(tmp_path / 'out.gz')
    actual = extract_telomere_reads([path], output, KMER, REV_KMER, MIN_HITS, num_processes=processes)
    assert actual == expected
    assert gzip.open(output, 'rb').read() == expected_bytes


@pytest.mark.parametrize('processes', [1, 2])
def test_base_count_batch_limit(tmp_path, processes):
    # Force the 4 Mb boundary well before the 256-record cap.
    seq = REJECTED * 650
    records = [alignment(f'long-{i}', seq + (KMER * 3 if i in (0, 194, 200) else ''))
               for i in range(201)]
    path = write_alignments(tmp_path / 'long.bam', records)
    expected, expected_bytes = eager_reference([path], '')
    output = str(tmp_path / 'out.gz')
    actual = extract_telomere_reads([path], output, KMER, REV_KMER, MIN_HITS, num_processes=processes)
    assert actual == expected
    assert gzip.open(output, 'rb').read() == expected_bytes


@pytest.mark.parametrize('processes', [2, 4])
def test_alignment_queue_remains_bounded(tmp_path, monkeypatch, processes):
    from collections import deque
    from source import tg_reader
    peaks = []
    class ObservedQueue(deque):
        def append(self, item):
            super().append(item)
            peaks.append((len(self), sum(len(batch[1]) for batch in self)))
    monkeypatch.setattr(tg_reader, 'deque', ObservedQueue)
    path = write_alignments(tmp_path / 'bounded.bam',
                            (alignment(f'read-{i}', REJECTED) for i in range(2311)))
    extract_telomere_reads([path], str(tmp_path / 'out.gz'), KMER, REV_KMER,
                           MIN_HITS, num_processes=processes)
    assert max(count for count, _ in peaks) == 2 * processes
    assert max(count for _, count in peaks) == 2 * processes * 256


@pytest.mark.parametrize('processes', [2, 4])
def test_missing_tags_do_not_retain_alignments(tmp_path, monkeypatch, processes):
    from collections import deque
    from source import tg_reader
    observed_batches = []
    class ObservedQueue(deque):
        def append(self, item):
            assert all(methyl is None for methyl in item[1])
            observed_batches.append(len(item[1]))
            super().append(item)
    monkeypatch.setattr(tg_reader, 'deque', ObservedQueue)
    tag_cases = ['none', 'mm_only', 'ml_only']
    path = write_alignments(tmp_path / 'missing-tags.bam',
                            (alignment(f'read-{i}', SELECTED if i % 137 == 0 else REJECTED,
                                       tags=tag_cases[i % len(tag_cases)]) for i in range(2311)))
    expected, expected_bytes = eager_reference([path], '')
    output = str(tmp_path / 'out.gz')
    actual = extract_telomere_reads([path], output, KMER, REV_KMER, MIN_HITS,
                                    num_processes=processes)
    assert actual == expected
    assert actual[7:] == ([], {})
    assert gzip.open(output, 'rb').read() == expected_bytes
    assert sum(observed_batches) == 2311
