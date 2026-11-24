"""Who spoke when: turn a meeting recording plus its recognised words into
"[mm:ss] Speaker N: text" paragraphs.

Runs as a SEPARATE process, started by whisper_dictate_macos_m.py after a
hands-free recording (DIARIZE_HANDS_FREE). It is deliberately not imported by
the daemon: sherpa-onnx lives in its own interpreter (DIARIZE_PYTHON), the
models are loaded only for the minute or two the job takes, and a crash or a
hang here cannot touch the hotkey. Everything is local: two ONNX models read
from disk, no network.

  python whisper_diarize.py --wav rec.wav --words words.json --out text.txt \
      --seg segmentation.onnx --emb embedding.onnx [--speakers 5]

--words is a JSON list of {"start": sec, "end": sec, "word": " text"} — what
mlx-whisper returns with word_timestamps=True. --speakers 0 lets the
clustering threshold decide how many voices there are (less reliable than a
given number); in that case the voices it found are then compared with each
other and the ones that sound like the same person are joined (merge_voices,
--merge-sim 0 turns that off). The last line printed is a JSON summary for the
daemon.
"""
import argparse
import json
import sys
import time
import wave


def read_wav(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("need 16-bit PCM")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels())[:, 0]
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0), rate


def diarize(audio, rate, seg, emb, speakers, threshold, threads):
    """[(start, end, speaker index from 0)], sorted by start."""
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speakers if speakers > 0 else -1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise ValueError("sherpa-onnx rejected the configuration (model paths?)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if rate != sd.sample_rate:
        raise ValueError(f"need {sd.sample_rate} Hz audio, got {rate}")
    return [(r.start, r.end, r.speaker) for r in sd.process(audio).sort_by_start_time()]


def make_embedder(audio, rate, emb, threads, cap_sec=90.0):
    """embed(spans) -> unit vector for the audio of [(start, end)], or None when
    there is too little of it to describe a voice. One long stretch of a voice
    describes it far more steadily than the short pieces the clustering sees."""
    import numpy as np
    import sherpa_onnx
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads))

    def embed(spans):
        chunks, n = [], 0
        for s, e in spans:
            c = audio[int(s * rate):int(e * rate)]
            chunks.append(c)
            n += len(c)
            if n >= cap_sec * rate:
                break
        if n < MIN_EMBED_SEC * rate:
            return None
        stream = ex.create_stream()
        stream.accept_waveform(rate, np.concatenate(chunks)[:int(cap_sec * rate)])
        stream.input_finished()
        v = np.array(ex.compute(stream), dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm else None
    return embed


MIN_EMBED_SEC = 2.0     # less audio than this does not describe a voice


def _cos(a, b):
    return float(sum(x * y for x, y in zip(a, b)))


def merge_voices(turns, embed, merge_sim=0.9, small_sec=10.0, small_share=0.03,
                 small_sim=0.6):
    """Join the voices that are the same person. turns = [(start, end, id)].

    The clustering inside sherpa-onnx compares short pieces of speech, and on
    a laptop microphone it splits one person into several "voices" (measured
    2026-11-20: two people came out as eleven). Here every voice is described
    once more, by up to 90 s of its own audio, and then:

      1. two voices whose descriptions agree by merge_sim or more are one
         person (halves of the SAME voice agree by 0.95-0.99, different
         people by 0.3-0.8 — measured on a real meeting), most alike first;
      2. a voice that said very little (under small_sec seconds and under
         small_share of all speech) goes to the voice it resembles most when
         the likeness is at least small_sim — otherwise it stays, it may be a
         person who spoke once;
      3. a voice with too little audio to describe at all goes to whoever
         spoke nearest in time.

    Returns turns with the surviving ids. merge_sim <= 0 returns them as is."""
    if merge_sim <= 0 or not turns:
        return list(turns)
    turns = list(turns)

    def spans(vid):
        return [(s, e) for s, e, sp in turns if sp == vid]

    def dur(vid):
        return sum(e - s for s, e in spans(vid))

    def relabel(src, dst):
        turns[:] = [(s, e, dst if sp == src else sp) for s, e, sp in turns]

    ids = sorted({sp for _, _, sp in turns})
    vec = {v: embed(spans(v)) for v in ids}

    # 1 — the same person twice
    while True:
        known = [v for v in sorted({sp for _, _, sp in turns}) if vec.get(v) is not None]
        best = None
        for i, a in enumerate(known):
            for b in known[i + 1:]:
                sim = _cos(vec[a], vec[b])
                if sim >= merge_sim and (best is None or sim > best[0]):
                    best = (sim, a, b)
        if best is None:
            break
        _, a, b = best
        keep, gone = (a, b) if dur(a) >= dur(b) else (b, a)
        relabel(gone, keep)
        vec.pop(gone, None)
        vec[keep] = embed(spans(keep))

    # 2 and 3 — scraps
    total = sum(e - s for s, e, _ in turns)
    for v in sorted({sp for _, _, sp in turns}, key=dur):
        others = [o for o in {sp for _, _, sp in turns} if o != v]
        if not others:
            break
        d = dur(v)
        if vec.get(v) is None:
            mid = sum((s + e) / 2 for s, e in spans(v)) / len(spans(v))
            near = min((t for t in turns if t[2] != v),
                       key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))
            relabel(v, near[2])
            continue
        if d >= small_sec or d >= small_share * total:
            continue
        big = [o for o in others if vec.get(o) is not None and dur(o) > d]
        if not big:
            continue
        to = max(big, key=lambda o: _cos(vec[v], vec[o]))
        if _cos(vec[v], vec[to]) >= small_sim:
            relabel(v, to)
    return sorted(turns, key=lambda t: t[0])


def renumber(turns):
    """Number the voices 1, 2, 3… in the order they first speak — the raw
    cluster ids have gaps and no meaning."""
    order = {}
    for _, _, sp in turns:
        order.setdefault(sp, len(order) + 1)
    return [(s, e, order[sp]) for s, e, sp in turns]


def speaker_of(word, turns):
    """The voice whose turn overlaps the word most; a word that fell into a
    gap between turns goes to the nearest turn. None when there are no turns."""
    best, best_overlap = None, 0.0
    for s, e, sp in turns:
        overlap = min(word["end"], e) - max(word["start"], s)
        if overlap > best_overlap:
            best, best_overlap = sp, overlap
    if best is None and turns:
        mid = (word["start"] + word["end"]) / 2
        best = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]
    return best


def stamp(sec):
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def paragraphs(words, turns, label="Speaker"):
    """Consecutive words of one voice become one paragraph."""
    paras = []
    for w in words:
        sp = speaker_of(w, turns)
        if paras and paras[-1][0] == sp:
            paras[-1][2].append(w["word"])
        else:
            paras.append([sp, w["start"], [w["word"]]])
    out = []
    for sp, start, ws in paras:
        text = "".join(ws).strip()
        if text:
            who = f"{label} {sp}" if sp is not None else f"{label} ?"
            out.append(f"[{stamp(start)}] {who}: {text}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--speakers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--merge-sim", type=float, default=0.9)
    ap.add_argument("--label", default="Speaker")
    a = ap.parse_args()

    t0 = time.monotonic()
    audio, rate = read_wav(a.wav)
    with open(a.words, encoding="utf-8") as f:
        words = json.load(f)
    turns = diarize(audio, rate, a.seg, a.emb, a.speakers, a.threshold, a.threads)
    found = len({t[2] for t in turns})
    if a.speakers <= 0 and a.merge_sim > 0 and found > 1:
        # The count was the model's own guess — join the voices that are one person.
        turns = merge_voices(turns, make_embedder(audio, rate, a.emb, a.threads), a.merge_sim)
    turns = renumber(turns)
    paras = paragraphs(words, turns, a.label)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras) + "\n")
    print(json.dumps({"speakers": len({t[2] for t in turns}), "found": found,
                      "paragraphs": len(paras),
                      "sec": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # the daemon reads stderr's last line into its log
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
