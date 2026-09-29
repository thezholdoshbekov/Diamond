"""Score a folder of restored wavs (produced by ANY model) with the SAME
DNSMOS-P.835 + CER used for Diamond eval — so baselines are apples-to-apples.

For each manifest row: input wav = row['audio_path']; restored wav =
<restored_dir>/<id>.wav (falls back to .flac). Scores DNSMOS (input+restored)
and, with --cer, CER/WER vs row['text']. Writes per-sample CSV + summary.

Usage:
  python scripts/score_dir.py --restored_dir data/bench/out/restored_sidon \
      --manifest data/bench/manifest_557.csv --out data/bench/out/score_sidon_557.csv \
      --cer --asr_device cuda --name sidon
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, "scripts")
from eval_heldout import make_dnsmos, make_cer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restored_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dnsmos", default="data/pretrained/dnsmos_p835.onnx")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--name", default="model")
    p.add_argument("--cer", action="store_true")
    p.add_argument("--no_dnsmos", action="store_true")
    p.add_argument("--asr_model", default="small.en")
    p.add_argument("--asr_device", default="cuda")
    p.add_argument("--skip_input", action="store_true", help="score restored only (input scores reused from an existing CSV)")
    args = p.parse_args()

    score = None if args.no_dnsmos else make_dnsmos(args.dnsmos)
    transcribe = cer_fn = wer_fn = None
    if args.cer:
        transcribe, cer_fn, wer_fn = make_cer(args.asr_model, device=args.asr_device)

    rows = list(csv.DictReader(open(args.manifest)))[: args.n]
    rdir = Path(args.restored_dir)
    out_rows, missing = [], 0
    for i, r in enumerate(rows):
        rid = r["id"]
        rp = rdir / f"{rid}.wav"
        if not rp.exists():
            rp = rdir / f"{rid}.flac"
        if not rp.exists():
            missing += 1
            continue
        try:
            rest, rsr = sf.read(str(rp), dtype="float32", always_2d=False)
            rec = dict(id=rid)
            if score is not None:
                s_r = score(rest, rsr)
                rec.update(r_sig=s_r[0], r_bak=s_r[1], r_ovr=s_r[2])
                if not args.skip_input:
                    inp, isr = sf.read(r["audio_path"], dtype="float32", always_2d=False)
                    s_i = score(inp, isr)
                    rec.update(in_sig=s_i[0], in_bak=s_i[1], in_ovr=s_i[2],
                               d_ovr=s_r[2]-s_i[2], d_sig=s_r[0]-s_i[0], d_bak=s_r[1]-s_i[1])
            if cer_fn is not None:
                ref = r.get("text", "")
                rh = transcribe(rest, rsr)
                rec.update(r_wer=wer_fn(ref, rh), r_cer=cer_fn(ref, rh))
                if not args.skip_input:
                    inp2, isr2 = sf.read(r["audio_path"], dtype="float32", always_2d=False)
                    ih = transcribe(inp2, isr2)
                    rec.update(in_wer=wer_fn(ref, ih), in_cer=cer_fn(ref, ih))
                    rec["d_wer"] = rec["r_wer"] - rec["in_wer"]
                    rec["d_cer"] = rec["r_cer"] - rec["in_cer"]
            out_rows.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  [{args.name}] {len(out_rows)} scored", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {rid}: {str(e)[:70]}", flush=True)

    if not out_rows:
        print(f"[{args.name}] no rows scored (missing {missing})"); return
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    def col(k): return np.array([o[k] for o in out_rows if k in o])
    print(f"\n[{args.name}] n={len(out_rows)}  missing={missing}")
    if score is not None:
        print(f"  DNSMOS restored OVRL {col('r_ovr').mean():.3f}  (Δ {col('d_ovr').mean():+.3f}, "
              f"improved {(col('d_ovr')>0).sum()}/{len(out_rows)}, median Δ {np.median(col('d_ovr')):+.3f})")
        print(f"         sig {col('r_sig').mean():.3f}  bak {col('r_bak').mean():.3f}")
    if cer_fn is not None:
        print(f"  CER restored {np.nanmean(col('r_cer')):.4f}  (input {np.nanmean(col('in_cer')):.4f})")
        print(f"  WER restored {np.nanmean(col('r_wer')):.4f}  (input {np.nanmean(col('in_wer')):.4f})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
