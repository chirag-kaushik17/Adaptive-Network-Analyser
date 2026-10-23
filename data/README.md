# Data directory

Place raw CIC-IDS2017 CSV files under this directory. The repository intentionally ignores all CSV files under data/ so that datasets remain untracked.

The expected file names are similar to:

- data/Monday-WorkingHours.pcap_ISCX.csv
- data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

The loader expects the CIC-IDS2017 traffic-labelling CSV format with timestamp and label columns.

The timestamp convention in the CIC files is minute-resolution and uses a 12-hour clock. The project assumes the capture runs roughly 08:00 to 17:00 local time; when an hour is below 8, the code converts it to PM to preserve the capture date and ordering.
