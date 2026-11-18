from __future__ import annotations

from itertools import product
from math import log2
from typing import Iterable

import numpy as np


BASES = "ACGT"
DINUCS = ["".join(chars) for chars in product(BASES, repeat=2)]
TRINUCS = ["".join(chars) for chars in product(BASES, repeat=3)]
FEATURE_VERSION = "seq20-pos1-pos2-kmer123-scalars-v1"


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper().replace("U", "T")


def validate_parts(tam5: str, target: str, tam3: str) -> tuple[str, str, str]:
    left = normalize_sequence(tam5)
    core = normalize_sequence(target)
    right = normalize_sequence(tam3)
    if len(left) != 1:
        raise ValueError(f"5' flanking base must be 1 nt, got {len(left)}")
    if len(core) != 18:
        raise ValueError(f"target must be 18 nt, got {len(core)}")
    if len(right) != 1:
        raise ValueError(f"3' flanking base must be 1 nt, got {len(right)}")
    sequence = left + core + right
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f"sequence contains non-ACGT characters: {''.join(invalid)}")
    return left, core, right


def sequence_from_parts(tam5: str, target: str, tam3: str) -> str:
    left, core, right = validate_parts(tam5, target, tam3)
    return left + core + right


def _entropy(sequence: str) -> float:
    if not sequence:
        return 0.0
    values = []
    for base in BASES:
        fraction = sequence.count(base) / len(sequence)
        if fraction:
            values.append(-fraction * log2(fraction))
    return float(sum(values))


def _max_run(sequence: str) -> int:
    if not sequence:
        return 0
    best = current = 1
    for previous, current_base in zip(sequence, sequence[1:]):
        if previous == current_base:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _self_complementarity(sequence: str) -> float:
    rc = _reverse_complement(sequence)
    return sum(a == b for a, b in zip(sequence, rc)) / len(sequence)


def feature_names() -> list[str]:
    names: list[str] = []
    for position in range(20):
        names.extend(f"pos{position + 1:02d}_{base}" for base in BASES)
    for position in range(19):
        names.extend(f"pos{position + 1:02d}_{kmer}" for kmer in DINUCS)
    names.extend(f"target_count_{base}" for base in BASES)
    names.extend(f"target_freq_{kmer}" for kmer in DINUCS)
    names.extend(f"target_freq_{kmer}" for kmer in TRINUCS)
    names.extend(
        [
            "gc_full",
            "gc_target",
            "gc_target_first9",
            "gc_target_last9",
            "gc_target_center6",
            "purine_target",
            "entropy_target",
            "max_homopolymer_target",
            "t_count_target",
            "poly_t4_present",
            "cpg_count_target",
            "self_complementarity_target",
            "terminal_gc_target",
            "middle_gc_target",
        ]
    )
    return names


def _feature_vector(sequence_20: str) -> list[float]:
    sequence = normalize_sequence(sequence_20)
    if len(sequence) != 20 or set(sequence) - set(BASES):
        raise ValueError("sequence_20 must contain exactly 20 A/C/G/T nucleotides")
    target = sequence[1:19]
    values: list[float] = []

    for base in sequence:
        values.extend(float(base == candidate) for candidate in BASES)
    for position in range(19):
        observed = sequence[position : position + 2]
        values.extend(float(observed == candidate) for candidate in DINUCS)

    values.extend(float(target.count(base)) for base in BASES)
    for kmer in DINUCS:
        values.append(sum(target[i : i + 2] == kmer for i in range(17)) / 17.0)
    for kmer in TRINUCS:
        values.append(sum(target[i : i + 3] == kmer for i in range(16)) / 16.0)

    gc = lambda seq: (seq.count("G") + seq.count("C")) / len(seq)
    values.extend(
        [
            gc(sequence),
            gc(target),
            gc(target[:9]),
            gc(target[9:]),
            gc(target[6:12]),
            (target.count("A") + target.count("G")) / len(target),
            _entropy(target),
            float(_max_run(target)),
            float(target.count("T")),
            float("TTTT" in target),
            float(sum(target[i : i + 2] == "CG" for i in range(17))),
            _self_complementarity(target),
            gc(target[:2] + target[-2:]),
            gc(target[7:11]),
        ]
    )
    return values


def feature_matrix(sequences: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    names = feature_names()
    matrix = np.asarray([_feature_vector(sequence) for sequence in sequences], dtype=np.float64)
    if matrix.shape[1] != len(names):
        raise RuntimeError(f"feature length mismatch: {matrix.shape[1]} != {len(names)}")
    return matrix, names
