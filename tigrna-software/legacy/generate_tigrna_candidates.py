from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


VALID = set("ACGT")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def parse_fasta_like(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    pieces: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(pieces).upper().replace("U", "T")))
            name = line[1:].strip()
            pieces = []
        else:
            pieces.append(re.sub(r"\s+", "", line))
    if name is not None:
        records.append((name, "".join(pieces).upper().replace("U", "T")))
    if not records:
        raise ValueError("No FASTA-like records found")
    return records


def enumerate_candidates(gene: str, sequence: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    length = len(sequence)
    rows: list[dict[str, object]] = []
    skipped = 0
    for strand, oriented in [("+", sequence), ("-", reverse_complement(sequence))]:
        ordinal = 0
        for offset in range(0, max(0, length - 19)):
            sequence_20 = oriented[offset : offset + 20]
            if len(sequence_20) != 20 or set(sequence_20) - VALID:
                skipped += 1
                continue
            ordinal += 1
            if strand == "+":
                target_start = offset + 2
                target_end = offset + 19
                genomic_target = sequence[target_start - 1 : target_end]
            else:
                target_start = length - offset - 18
                target_end = length - offset - 1
                genomic_target = sequence[target_start - 1 : target_end]
            rows.append(
                {
                    "id": f"{gene}_{'F' if strand == '+' else 'R'}_{ordinal:04d}",
                    "gene": gene,
                    "strand": strand,
                    "target_start_1based": target_start,
                    "target_end_1based": target_end,
                    "tam5": sequence_20[0],
                    "target": sequence_20[1:19],
                    "tam3": sequence_20[19],
                    "sequence_20": sequence_20,
                    "genomic_target_5to3": genomic_target,
                }
            )
    meta = {
        "gene": gene,
        "length_nt": length,
        "non_acgt_characters": sorted(set(sequence) - VALID),
        "candidate_count": len(rows),
        "forward_count": sum(row["strand"] == "+" for row in rows),
        "reverse_count": sum(row["strand"] == "-" for row in rows),
        "skipped_windows": skipped,
    }
    return rows, meta


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: generate_tigrna_candidates.py <input.txt> <output.csv> <meta.json>")
    input_path, output_path, meta_path = map(Path, sys.argv[1:])
    records = parse_fasta_like(input_path)
    all_rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for gene, sequence in records:
        rows, meta = enumerate_candidates(gene, sequence)
        all_rows.extend(rows)
        metadata.append(meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {"source": str(input_path), "record_count": len(records), "total_candidates": len(all_rows), "records": metadata}
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
