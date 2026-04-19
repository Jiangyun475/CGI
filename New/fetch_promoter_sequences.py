#!/usr/bin/env python3
"""
Fetch 2kb promoter sequences (TSS upstream) for 978 LINCS landmark genes.

Strategy:
  1. Use NCBI gene table to get genomic coordinates and strand for each Entrez Gene ID
  2. Compute promoter region: [TSS-2000, TSS+200] (2kb upstream + 200bp downstream of TSS)
  3. Fetch sequence from NCBI nucleotide DB using genomic accession + coordinates
  4. Cache results to JSON for fast reload

Output: promoter_sequences.json  {gene_id_str: dna_sequence}
"""

import json
import pickle
import time
import re
import sys
from pathlib import Path
import requests
import pandas as pd

EMAIL = 'yunfei0feiyun@gmail.com'
UPSTREAM   = 2000   # bp upstream of TSS
DOWNSTREAM = 200    # bp downstream of TSS (capture 5' UTR start)
CACHE_FILE = Path('/home/data/jiangyun/cgi_data_pipeline/outputs/gene_data/promoter_sequences.json')
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def fetch_gene_table(entrez_id: str, retries=3) -> str:
    """Fetch gene table text from NCBI Gene."""
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    for attempt in range(retries):
        try:
            r = requests.get(url, params={
                'db': 'gene', 'id': entrez_id,
                'rettype': 'gene_table', 'retmode': 'text',
                'email': EMAIL
            }, timeout=15)
            if r.status_code == 200 and r.text.strip():
                return r.text
        except Exception as e:
            if attempt == retries - 1:
                raise
        time.sleep(0.5 * (attempt + 1))
    return ''


def parse_gene_location(gene_table_text: str):
    """
    Parse primary genomic location from gene table.
    Returns (accession, prom_start, prom_end, fetch_strand) or None.

    NCBI gene table formats:
      (A) With explicit strand:
          "Reference GRCh38.p14 Primary Assembly NC_000023.11  (minus strand) from: 100637104 to: 100627108"
      (B) Without explicit strand:
          "Reference GRCh38.p14 Primary Assembly NC_000006.12  from: 30571442 to: 30591522"

    Key insight:
      - 'from' coordinate is ALWAYS the gene's 5' end (TSS direction)
      - 'from' < 'to'  → plus strand,  TSS = 'from'
      - 'from' > 'to'  → minus strand, TSS = 'from'
    """
    # Pattern A: explicit (strand) notation
    pat_a = r'Reference GRCh38.*?Primary Assembly\s+(\S+)\s+\((\w+)\s+strand\)\s+from:\s+(\d+)\s+to:\s+(\d+)'
    match = re.search(pat_a, gene_table_text)

    if match:
        accession = match.group(1)
        strand    = match.group(2).lower()   # 'plus' or 'minus'
        coord_a   = int(match.group(3))
        coord_b   = int(match.group(4))
        is_plus   = (strand == 'plus')
    else:
        # Pattern B: no explicit strand — infer from from/to ordering
        pat_b = r'Reference GRCh38.*?Primary Assembly\s+(\S+)\s+from:\s+(\d+)\s+to:\s+(\d+)'
        match = re.search(pat_b, gene_table_text)
        if not match:
            # Last fallback: any NC_ accession
            pat_c = r'(NC_\d+\.\d+)\s+from:\s+(\d+)\s+to:\s+(\d+)'
            match = re.search(pat_c, gene_table_text)
        if not match:
            return None
        accession = match.group(1)
        coord_a   = int(match.group(2))
        coord_b   = int(match.group(3))
        is_plus   = (coord_a <= coord_b)  # from < to → plus strand

    # TSS is always at 'from' (coord_a)
    tss = coord_a

    if is_plus:
        prom_start   = max(1, tss - UPSTREAM)
        prom_end     = tss + DOWNSTREAM
        fetch_strand = 1  # forward
    else:  # minus strand: TSS at 'from' which is the higher coord
        prom_start   = tss - DOWNSTREAM
        prom_end     = tss + UPSTREAM
        fetch_strand = 2  # reverse complement

    return accession, prom_start, prom_end, fetch_strand


def fetch_genomic_sequence(accession: str, seq_start: int, seq_end: int,
                            strand: int, retries=3) -> str:
    """Fetch DNA sequence from NCBI nucleotide DB for specific genomic coordinates."""
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    for attempt in range(retries):
        try:
            r = requests.get(url, params={
                'db': 'nucleotide', 'id': accession,
                'rettype': 'fasta', 'retmode': 'text',
                'seq_start': seq_start, 'seq_stop': seq_end,
                'strand': strand,
                'email': EMAIL
            }, timeout=20)
            if r.status_code == 200 and r.text.strip():
                # Parse FASTA: skip header line
                lines = r.text.strip().split('\n')
                seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
                seq = seq.upper().replace('N', 'N')
                if len(seq) > 10:
                    return seq
        except Exception as e:
            pass
        time.sleep(1.0 * (attempt + 1))
    return ''


def main():
    # Load gene IDs from dataset
    csv_path = '/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended/MCF7/dataset.csv'
    df = pd.read_csv(csv_path)
    gene_ids = [str(g) for g in sorted(df['gene_id'].unique())]
    print(f"Target genes: {len(gene_ids)}")

    # Load cache
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Loaded cache: {len(cache)} sequences already fetched")
    else:
        cache = {}

    remaining = [g for g in gene_ids if g not in cache]
    print(f"Remaining to fetch: {len(remaining)}")

    failed = []
    for i, gid in enumerate(remaining):
        if i % 50 == 0:
            print(f"  [{i}/{len(remaining)}] fetching gene {gid}...")
            # Save cache every 50 genes
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)

        try:
            # Step 1: Get genomic coordinates
            gene_table = fetch_gene_table(gid)
            if not gene_table:
                print(f"  ! Gene {gid}: failed to get gene table")
                failed.append(gid)
                time.sleep(0.5)
                continue

            result = parse_gene_location(gene_table)
            if result is None:
                print(f"  ! Gene {gid}: failed to parse coordinates")
                failed.append(gid)
                time.sleep(0.5)
                continue

            accession, prom_start, prom_end, strand = result

            # Step 2: Fetch promoter sequence
            time.sleep(0.35)  # NCBI rate limit: ~3 req/sec
            seq = fetch_genomic_sequence(accession, prom_start, prom_end, strand)
            if not seq:
                print(f"  ! Gene {gid}: failed to fetch sequence")
                failed.append(gid)
                time.sleep(0.5)
                continue

            cache[gid] = seq
            time.sleep(0.35)

        except Exception as e:
            print(f"  ! Gene {gid}: exception {e}")
            failed.append(gid)
            time.sleep(1.0)

    # Final save
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

    print(f"\nDone!")
    print(f"  Successfully fetched: {len(cache)}/{len(gene_ids)}")
    print(f"  Failed: {len(failed)} genes: {failed[:10]}")
    print(f"  Saved to: {CACHE_FILE}")

    # Analyze fetched sequences
    if cache:
        lens = [len(v) for v in cache.values()]
        import numpy as np
        print(f"\nSequence length stats:")
        print(f"  min={min(lens)}, max={max(lens)}, median={np.median(lens):.0f}")


if __name__ == '__main__':
    main()
