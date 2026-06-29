#!/usr/bin/env bash
# =============================================================================
# Full PDB mmCIF Download Script for ADiT Training
# Source: RCSB PDB (rsync.rcsb.org)
# Format: mmCIF (.cif.gz) — divided layout
# =============================================================================

set -euo pipefail

# --- Configuration ---
DEST_DIR="${1:-/data/pdb_mmcif}"        # Override with: ./script.sh /your/path
RSYNC_SERVER="rsync.rcsb.org::ftp_data/structures/divided/mmCIF/"
LOG_FILE="${DEST_DIR}/download.log"
N_RETRIES=5
BANDWIDTH_LIMIT=10000       # KB/s, 0 = unlimited

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  PDB mmCIF Full Mirror Download       ${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo -e "Destination : ${YELLOW}${DEST_DIR}${NC}"
echo -e "Source      : ${YELLOW}${RSYNC_SERVER}${NC}"
echo -e "Log file    : ${YELLOW}${LOG_FILE}${NC}"
echo ""

# --- Check dependencies ---
for cmd in rsync gzip; do
  if ! command -v "$cmd" &> /dev/null; then
    echo -e "${RED}ERROR: '$cmd' is not installed. Please install it first.${NC}"
    exit 1
  fi
done

# --- Create destination ---
mkdir -p "${DEST_DIR}"

# --- Estimate disk space ---
echo -e "${YELLOW}Note:${NC} Full PDB mmCIF archive is ~700GB compressed."
echo -e "      Make sure you have sufficient disk space."
echo ""

# --- Confirm ---
read -rp "Proceed with download? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo -e "${GREEN}Starting rsync...${NC}"
echo "Start time: $(date)" | tee -a "${LOG_FILE}"

# --- rsync download ---
rsync \
  --recursive \
  --links \
  --perms \
  --times \
  --compress \
  --verbose \
  --progress \
  --delete \
  --delete-after \
  --contimeout=60 \
  --timeout=120 \
  $([ "$BANDWIDTH_LIMIT" -gt 0 ] && echo "--bwlimit=${BANDWIDTH_LIMIT}") \
  "${RSYNC_SERVER}" \
  "${DEST_DIR}/" \
  2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo -e "${GREEN}Download completed successfully!${NC}"
  echo "End time: $(date)" | tee -a "${LOG_FILE}"
else
  echo ""
  echo -e "${RED}rsync exited with code $EXIT_CODE. Check log: ${LOG_FILE}${NC}"
  exit $EXIT_CODE
fi

# --- Count downloaded files ---
echo ""
echo "Counting files..."
TOTAL=$(find "${DEST_DIR}" -name "*.cif.gz" | wc -l)
echo -e "Total mmCIF files: ${GREEN}${TOTAL}${NC}"
echo "Total files: ${TOTAL}" >> "${LOG_FILE}"

# --- Disk usage ---
echo ""
du -sh "${DEST_DIR}"
