# Start a named session
screen -S pdb_download

# Then launch the script normally
./download_pdb_mmcif.sh /data/pdb_mmcif

# Detach (leave it running in background)
Ctrl+A  then  D

# Reattach later
screen -r pdb_download

# List active sessions
screen -ls
