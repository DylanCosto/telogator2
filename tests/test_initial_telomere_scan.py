import gzip
import tempfile
import unittest
from pathlib import Path

import pysam

from source.tg_reader import extract_telomere_reads, quick_grab_all_reads


class InitialTelomereScanTests(unittest.TestCase):
    def test_parallel_scan_preserves_reads_order_and_counts_across_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fasta = root / 'reads.fa.gz'
            fastq = root / 'reads.fq'
            bam = root / 'reads.bam'
            passing = 'TTAGGGTTAGGG' * 8
            reverse = 'CCCTAACCCTAA' * 8
            failing = 'ACGT' * 30
            fasta_reads = [(f'f{i}', passing if i % 3 == 0 else failing)
                           for i in range(300)]
            with gzip.open(fasta, 'wt') as output:
                for name, sequence in fasta_reads:
                    output.write(f'>{name}\n{sequence}\n')
            with open(fastq, 'w') as output:
                output.write(f'@reverse\n{reverse}\n+\n{"I" * len(reverse)}\n')
                output.write(f'@short\n{failing}\n+\n{"I" * len(failing)}\n')
            with pysam.AlignmentFile(bam, 'wb', header={'HD': {'VN': '1.0'}}) as output:
                for name, sequence, flag in [('bam_pass', passing, 4),
                                             ('supplementary', passing, 2052),
                                             ('bam_fail', failing, 4)]:
                    read = pysam.AlignedSegment()
                    read.query_name = name
                    read.query_sequence = sequence
                    read.flag = flag
                    output.write(read)

            expected_names = [name for name, _ in fasta_reads
                              if int(name[1:]) % 3 == 0] + ['reverse', 'bam_pass']
            results = []
            outputs = []
            for workers in (1, 2, 4):
                path = root / f'output-{workers}.fa.gz'
                result = extract_telomere_reads(
                    [str(fasta), str(fastq), str(bam)], str(path),
                    'TTAGGGTTAGGG', 'CCCTAACCCTAA', 8, workers)
                with gzip.open(path, 'rt') as output:
                    content = output.read()
                self.assertEqual(len(quick_grab_all_reads(str(path))), 102)
                results.append(result)
                outputs.append(content)
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], outputs[2])
            self.assertEqual([line[1:] for line in outputs[0].splitlines()
                              if line.startswith('>')], expected_names)
            self.assertEqual(results[0][:5],
                             (304, 102, 1, 101 * len(passing) +
                              202 * len(failing) + len(reverse),
                              101 * len(passing) + len(reverse)))

    def test_parallel_scan_writes_valid_empty_gzip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fasta = root / 'reads.fa'
            fasta.write_text('>no_telomere\nACGTACGT\n')
            for workers in (1, 4):
                output = root / f'empty-{workers}.fa.gz'
                result = extract_telomere_reads(
                    [str(fasta)], str(output), 'TTAGGG', 'CCCTAA', 8, workers)
                self.assertEqual(result[:5], (1, 0, 0, 8, 0))
                with gzip.open(output, 'rt') as reads:
                    self.assertEqual(reads.read(), '')


if __name__ == '__main__':
    unittest.main()
