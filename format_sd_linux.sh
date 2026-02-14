#!/usr/bin/env bash
# Format SD card for OpenBCI on Fedora/Linux
# Equivalent of macOS diskutil commands from README
set -euo pipefail

LABEL="OBCI"
REAL_USER="${SUDO_USER:-$USER}"

echo "=== Available block devices ==="
lsblk -d -o NAME,SIZE,TYPE,MODEL,TRAN | grep -E "usb|mmcblk|NAME"
echo ""

read -rp "Enter device name (e.g. sdb): " DEV
DEV="/dev/${DEV}"

if [[ ! -b "$DEV" ]]; then
    echo "Error: $DEV is not a block device"
    exit 1
fi

# safety: only allow USB devices and mmcblk (integrated SD card readers)
TRAN=$(lsblk -n -d -o TRAN "$DEV" 2>/dev/null || true)
if [[ "$DEV" == /dev/mmcblk* ]]; then
    : # integrated SD card reader, allow
elif [[ "$DEV" == /dev/sd* ]] && [[ "$TRAN" == "usb" ]]; then
    : # USB SD card reader, allow
else
    echo "Error: $DEV is not an SD card device (transport: ${TRAN:-unknown})"
    exit 1
fi

echo ""
echo "=== $DEV details ==="
lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT "$DEV"
echo ""
echo "WARNING: This will ERASE ALL DATA on $DEV"
read -rp "Continue? [y/N]: " CONFIRM
if [[ "$CONFIRM" != [yY] ]]; then
    echo "Aborted"
    exit 1
fi

DISK_SIZE=$(lsblk -b -n -d -o SIZE "$DEV")
DISK_SIZE_MB=$((DISK_SIZE / 1024 / 1024))

# unmount all partitions
echo ""
echo "[1/5] Unmounting partitions..."
umount "${DEV}"* 2>/dev/null || true
echo "  Done."

# fill with zeros (oflag=direct bypasses page cache so progress is real disk speed)
echo ""
echo "[2/4] Zeroing disk ($DISK_SIZE_MB MB)..."
dd if=/dev/zero of="$DEV" bs=1M oflag=direct 2>&1 &
DD_PID=$!
while kill -0 $DD_PID 2>/dev/null; do
    WRITTEN=$(awk '/^write_bytes:/{print $2}' "/proc/$DD_PID/io" 2>/dev/null || echo 0)
    PCT=$((WRITTEN * 100 / DISK_SIZE))
    WRITTEN_MB=$((WRITTEN / 1024 / 1024))
    printf "\r  %d%% (%d / %d MB)  " "$PCT" "$WRITTEN_MB" "$DISK_SIZE_MB"
    sleep 1
done
wait $DD_PID 2>/dev/null || true
printf "\r  Disk fully zeroed. (%d MB)    \n" "$DISK_SIZE_MB"

# create MBR partition table + single FAT32 partition
echo ""
echo "[3/4] Creating MBR partition table..."
echo -e "o\nn\np\n1\n\n\nt\nc\nw" | fdisk "$DEV" > /dev/null 2>&1
sleep 1
partprobe "$DEV"
sleep 1

# detect partition name (sdb1 or mmcblk0p1)
PART="${DEV}1"
if [[ ! -b "$PART" ]]; then
    PART="${DEV}p1"
fi
if [[ ! -b "$PART" ]]; then
    echo "  Error: cannot find partition on $DEV"
    exit 1
fi
echo "  Done."

# format as FAT32
echo ""
echo "[4/4] Formatting $PART as FAT32 (label: $LABEL)..."
mkfs.vfat -F 32 -n "$LABEL" "$PART" > /dev/null
echo "  Done."

# mount
MOUNT_DIR="/run/media/$REAL_USER/$LABEL"
echo ""
echo "Mounting at $MOUNT_DIR..."
mkdir -p "$MOUNT_DIR"
mount "$PART" "$MOUNT_DIR"

# verify
echo ""
echo "=== Verify ==="
FSTYPE=$(lsblk -n -o FSTYPE "$PART")
PART_LABEL=$(lsblk -n -o LABEL "$PART")
PART_SIZE=$(lsblk -n -o SIZE "$PART")

echo "  Partition: $PART ($PART_SIZE)"
echo "  Type:      $FSTYPE"
echo "  Label:     $PART_LABEL"
echo "  Mount:     $MOUNT_DIR"

if [[ "$FSTYPE" != "vfat" ]]; then
    echo "  FAIL: Expected vfat, got $FSTYPE"
    exit 1
fi
if [[ "$PART_LABEL" != "$LABEL" ]]; then
    echo "  FAIL: Expected label $LABEL, got $PART_LABEL"
    exit 1
fi

# read 100MB after 50MB offset and confirm zeros
echo ""
echo "Checking zeros at 50MB offset..."
HEXOUT=$(dd if="$DEV" bs=1M skip=50 count=100 2>/dev/null | hexdump -C | head -5)
echo "$HEXOUT"
if echo "$HEXOUT" | grep -q "^\*$"; then
    echo "  OK: disk is zeroed."
else
    echo "  WARNING: unexpected non-zero data found."
fi

echo ""
echo "Done. SD card ready at $MOUNT_DIR"
