#!/usr/bin/env python3
"""
PDB mmCIF Download Verifier & Inventory Builder
Checks integrity of downloaded files and builds an index for ADiT training.
"""

import os
import gzip
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def check_file(path: Path) -> dict:
    """Check a single mmCIF.gz file for integrity."""
    result = {"path": str(path), "pdb_id": path.stem.replace(".cif", "").upper(), "ok": False, "size_mb": 0, "error": None}
    try:
        result["size_mb"] = round(path.stat().st_size / 1e6, 3)
        with gzip.open(path, "rt") as f:
            # Read first few lines to confirm it's a valid mmCIF
            header = [f.readline() for _ in range(5)]
        if any("data_" in line for line in header):
            result["ok"] = True
        else:
            result["error"] = "No mmCIF data_ block found"
    except Exception as e:
        result["error"] = str(e)
    return result


def build_inventory(pdb_dir: str, output_json: str, workers: int = 8, verify: bool = True):
    pdb_path = Path(pdb_dir)
    files = sorted(pdb_path.rglob("*.cif.gz"))
    log.info(f"Found {len(files)} mmCIF.gz files in {pdb_dir}")

    inventory = []
    errors = []

    if verify:
        log.info(f"Verifying files with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(check_file, f): f for f in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="Verifying"):
                res = future.result()
                inventory.append(res)
                if not res["ok"]:
                    errors.append(res)
    else:
        log.info("Skipping verification, building index only...")
        for f in tqdm(files, desc="Indexing"):
            inventory.append({
                "path": str(f),
                "pdb_id": f.stem.replace(".cif", "").upper(),
                "ok": True,
                "size_mb": round(f.stat().st_size / 1e6, 3),
                "error": None
            })

    # Summary
    ok_count = sum(1 for r in inventory if r["ok"])
    total_size = sum(r["size_mb"] for r in inventory)

    summary = {
        "total_files": len(inventory),
        "ok_files": ok_count,
        "error_files": len(errors),
        "total_size_gb": round(total_size / 1024, 2),
        "inventory": inventory,
        "errors": errors,
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"OK: {ok_count} / {len(inventory)}")
    log.info(f"Errors: {len(errors)}")
    log.info(f"Total size: {summary['total_size_gb']} GB")
    log.info(f"Inventory saved to: {output_json}")

    if errors:
        log.warning("Files with errors:")
        for e in errors[:10]:
            log.warning(f"  {e['pdb_id']}: {e['error']}")
        if len(errors) > 10:
            log.warning(f"  ... and {len(errors) - 10} more. See {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and index PDB mmCIF downloads")
    parser.add_argument("pdb_dir", help="Path to the downloaded PDB mmCIF directory")
    parser.add_argument("--output", default="pdb_inventory.json", help="Output JSON inventory file")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--no-verify", action="store_true", help="Skip gzip integrity check (faster)")
    args = parser.parse_args()

    build_inventory(args.pdb_dir, args.output, args.workers, verify=not args.no_verify)
