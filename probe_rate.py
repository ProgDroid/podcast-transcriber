"""MEASURE the sample rate of the local clip audio from the MP3 frame headers.

The clips were cut by speaker_tool.py with `ffmpeg -ss <seconds> -i ... -t
<seconds> -ac 1` and NO `-ar` flag, so ffmpeg preserves the source episode's
sample rate. These files are therefore direct evidence about the archive audio.

Pure header parsing -- no ffprobe, no torchaudio, no decoding.
"""

import collections
from pathlib import Path

# MPEG audio version -> sampling-rate table (index 0..2)
RATES = {
    3: [44100, 48000, 32000],  # MPEG-1
    2: [22050, 24000, 16000],  # MPEG-2
    0: [11025, 12000, 8000],  # MPEG-2.5
}


def first_frame_rate(path: Path):
    """Sample rate from the first valid MPEG audio frame header, or None."""
    data = path.read_bytes()
    i = 0
    # skip an ID3v2 tag if present (these clips use -map_metadata -1, but be safe)
    if data[:3] == b"ID3" and len(data) > 10:
        size = 0
        for b in data[6:10]:
            size = (size << 7) | (b & 0x7F)
        i = 10 + size
    limit = min(len(data) - 4, i + 200000)
    while i < limit:
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            ver = (data[i + 1] >> 3) & 0x03  # 3=MPEG1, 2=MPEG2, 0=MPEG2.5
            rate_idx = (data[i + 2] >> 2) & 0x03
            bitrate_idx = (data[i + 2] >> 4) & 0x0F
            if ver != 1 and rate_idx != 3 and bitrate_idx not in (0, 15):
                return RATES[ver][rate_idx]
        i += 1
    return None


clips = sorted(Path("clips").glob("*.mp3"))
print("local clip files: %d" % len(clips))
if not clips:
    print("POSITIVE CONTROL FAILED: no clips found. Rate is UNKNOWN.")
    raise SystemExit(1)

rates = collections.Counter()
unreadable = []
per_episode = {}
for p in clips:
    r = first_frame_rate(p)
    if r is None:
        unreadable.append(p.name)
        continue
    rates[r] += 1
    ep = p.stem.split("--")[0]
    per_episode.setdefault(ep, set()).add(r)

print("\nMEASURED sample rates across clip files:")
for r, n in rates.most_common():
    print("  %6d Hz : %4d files  (%.1f%%)" % (r, n, 100 * n / sum(rates.values())))
print("  unreadable headers: %d %s" % (len(unreadable), unreadable[:3]))

print("\nDistinct EPISODES covered: %d" % len(per_episode))
ep_rates = collections.Counter(
    tuple(sorted(v))[0] for v in per_episode.values() if len(v) == 1
)
mixed = {k: v for k, v in per_episode.items() if len(v) > 1}
for r, n in ep_rates.most_common():
    print("  %6d Hz : %3d episodes" % (r, n))
print("  episodes with MIXED rates across their own clips: %d" % len(mixed))

print("\nImplied seek ratio 16000/native, per rate present:")
for r in sorted(rates):
    print("  %6d Hz -> crop lands at %.4f of nominal time" % (r, 16000 / r))

# shows present, to say whether other feeds could differ
shows = collections.Counter(ep.rsplit(" - Episode", 1)[0] for ep in per_episode)
print("\nShows represented in the local clips: %s" % dict(shows))
