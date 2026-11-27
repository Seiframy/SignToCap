"""Build the experiment report from immutable benchmark logs and the paired source manifest."""

from collections import Counter
from html import escape
import json
import math
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Circle, Rect, String
from reportlab.lib.colors import HexColor, white


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "signgpt-results" / "20261127T062446887049Z"
NAMES = ["how2sign-oracle", "continuous-oracle", "continuous-replay"]
LABELS = ["32 official clips", "Six gold-bounded clips", "Continuous replay"]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


events = [jsonl(RESULTS / f"{name}-events.jsonl") for name in NAMES]
summary = [json.loads((RESULTS / f"{name}-summary.json").read_text(encoding="utf-8")) for name in NAMES]
tracks = jsonl(HERE / "continuous-inputs.jsonl")
oracle, paired = [[row for row in arm if row["kind"] == "oracle"] for arm in events[:2]]
replay = events[2]
timed_updates = [row for row in replay if "source_end_ms" in row]
lags = sorted(row["emit_ms"] - row["source_end_ms"] for row in timed_updates)
assert len(oracle) == 32 and all(row["parity"] is True for row in oracle)
assert len(paired) == sum(len(track["segments"]) for track in tracks) == 6
assert all(row[-1]["kind"] == "gpu_memory" for row in events)
assert all(events[1][0][key] == events[2][0][key] for key in ("beams", "tokens"))


def graph():
    navy, gray, faint, teal, violet, orange = map(
        HexColor, ("#172540", "#44556E", "#DCE5EF", "#087E83", "#6544AC", "#C44724")
    )
    drawing = Drawing(1010, 480)
    drawing.add(Rect(0, 0, 1010, 480, fillColor=white, strokeColor=None))

    def label(x, y, text, size=12, color=navy, bold=False, anchor="start"):
        drawing.add(String(x, y, text, fontName="Helvetica-Bold" if bold else "Helvetica",
                           fontSize=size, fillColor=color, textAnchor=anchor))

    def line(x1, y1, x2, y2, color, weight=1, dash=None):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=weight, strokeDashArray=dash))

    label(25, 450, "Model work per inference", 20, bold=True)
    label(25, 427, "Warmed timings on one NVIDIA L40S. Replay target p95 <400 ms.", 12, gray)
    x = lambda ms: 231 + ms * 0.72
    for ms in range(0, 1001, 100):
        line(x(ms), 283, x(ms), 398, faint)
        if ms % 200 == 0:
            label(x(ms), 262, str(ms), 11, gray, anchor="middle")
    line(x(400), 283, x(400), 398, orange, 1.4, [4, 3])
    line(x(500), 283, x(500), 398, gray, 1.0, [2, 3])
    label(x(400), 402, "400 target", 10, orange, anchor="middle")
    label(x(500), 402, "500 cadence", 10, gray, anchor="middle")
    for i, (name, arm) in enumerate(zip(LABELS, summary)):
        y = 372 - 35 * i
        a, b = (arm["model_work_ms"][key] for key in ("p50", "p95"))
        label(25, y - 4, name)
        line(x(a), y, x(b), y, violet, 3)
        drawing.add(Circle(x(a), y, 4, fillColor=white, strokeColor=violet, strokeWidth=2))
        drawing.add(Circle(x(b), y, 5, fillColor=violet, strokeColor=None))
        label(x(b) + 10, y - 4, f"{b:.0f}", 11, violet)
    label(948, 262, "ms", 11, gray, anchor="end")
    label(25, 231, "Continuous playback against reference sentence boundaries", 19, bold=True)
    label(25, 210, "Blocks = reference spans  |  dots = partial updates  |  orange ticks = uncertain resets or EOF", 12, gray)
    sx = lambda ms: 231 + ms / 1000 * 29
    for i in range(0, 26, 5):
        line(sx(i * 1000), 51, sx(i * 1000), 187, faint)
        label(sx(i * 1000), 32, f"{i} s", 11, gray, anchor="middle")
    for idx, (track, title) in enumerate(zip(tracks, ("Roller derby", "Flight school"))):
        y = 150 - idx * 65
        label(25, y - 3, title, 13, bold=True)
        label(25, y - 20, "0 final captions", 11, orange)
        for number, unit in enumerate(track["segments"], 1):
            left, right = sx(unit["start_ms"]), sx(unit["end_ms"])
            drawing.add(Rect(left, y, right - left, 21, strokeColor=white, strokeWidth=1,
                             fillColor=HexColor("#CDE9E8")))
            label((left + right) / 2, y + 6, str(number), 10, teal, anchor="middle")
        duration = track["source"]["frames"] / track["source"]["source_fps"] * 1000
        for item in replay:
            if item.get("id") != track["id"]:
                continue
            if item["kind"] == "partial":
                drawing.add(Circle(sx(item["source_end_ms"]), y - 8, 2, fillColor=violet, strokeColor=None))
            elif item["kind"] == "uncertain":
                pos = sx(item.get("source_end_ms", duration))
                line(pos, y - 15, pos, y - 2, orange, 2)
    renderSVG.drawToFile(drawing, str(HERE / "phase2-results.svg"))


# One-pass comparison of English output to the supplied English reference.
# Indices are the order in the benchmark JSONL, starting at one.
PRESERVED = {1, 4, 9, 18, 30}
PARTIAL = {2, 3, 5, 7, 8, 11, 12, 14, 15, 16, 22, 24, 27, 28, 29, 31}


def table(headers, rows):
    head = "".join(f"<th>{escape(str(v))}</th>" for v in headers)
    content = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{content}</tbody></table>"


def render():
    graph()
    translations = []
    for index, row in enumerate(oracle, 1):
        assessment = "Preserved" if index in PRESERVED else "Partial" if index in PARTIAL else "Wrong"
        translations.append((index, row["id"], row["reference"], row["text"], assessment, f"{row['work_ms']:.0f}"))
    counts = Counter(row[4] for row in translations)
    metrics = []
    for idx, arm in enumerate(summary):
        work, mem = arm["model_work_ms"], arm["gpu_memory"]
        bleu = arm.get("oracle_bleu")
        metrics.append((LABELS[idx], work["n"], f"{bleu['score']:.2f}" if bleu else "—",
                        f"{work['p50']:.0f} / {work['p95']:.0f}",
                        f"{arm['generation_ms']['p50']:.0f} / {arm['generation_ms']['p95']:.0f}",
                        f"{mem['torch_peak_allocated_mib']:.1f} / {mem['torch_peak_reserved_mib']:.1f}"))
    # Decode only the known UTF-8-as-Latin-1 apostrophe sequence for display.
    paired_rows = [(r["id"].split("/")[-1], r["reference"].replace("â\x80\x99", "’"), r["text"]) for r in paired]
    p95 = lags[math.ceil(.95 * len(lags)) - 1]
    lag_p50 = lags[math.ceil(.5 * len(lags)) - 1]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignToCap Phase 2 experiment and Phase 3 decision</title>
<style>
body{{margin:0;background:#f3f6fa;color:#172540;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:36px 26px 64px;background:white}}
h1,h2{{line-height:1.15}} h1{{font-size:2.15rem;margin:0 0 .65em}} h2{{font-size:1.45rem;margin:1.7em 0 .4em}}
p{{margin:.5em 0 1em}} .decision{{padding:14px 18px;border-left:4px solid #087e83;background:#eaf5f5}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:14px 0 24px}} td,th{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #dce5ef}} th{{background:#edf2f8}}
figure{{margin:18px 0}} img{{width:100%;height:auto}} figcaption,small{{color:#44556e}} summary{{cursor:pointer;font-weight:700}}
.scroll{{overflow-x:auto}} code{{overflow-wrap:anywhere}} li{{margin:.35em 0}}
</style></head><body><main>
<h1>SignToCap Phase 2 experiment</h1>
<p>Run <code>20261127T062446887049Z</code> · 27 November 2026 · frozen Uni-Sign on NVIDIA L40S</p>
<p class="decision"><strong>No-go for streaming LoRA and unrestricted live captions.</strong> Start Phase 3 with <strong>C</strong>, a controlled pose-compatibility and oracle-quality check, and address decoder tail latency only after meaning is established. <strong>D</strong> is the fallback if the frozen backend still fails on new independent material. Implement a small controller fix in parallel. Keep the competition demo scoped to a verified replay if the live gate stays open.</p>

<h2>1. Integrity and shutdown</h2>
<p>The runner reported <code>completed</code>, and all three event logs end in their GPU-memory record. The frozen task checkpoint and mT5 weight SHA-256 values match the pins. The runner logged seven asset checksum passes, the expected Uni-Sign revision <code>eed438bcb49e30405cd6ccdfcccca330c134e830</code>, clean dependency verification, and a successful BF16 preflight on one L40S with PyTorch 2.6.0 / CUDA 12.4. The loader uses a strict state-dict load. The source-only decode matched the reference implementation token-for-token on <strong>32/32</strong> official clips. The two paired arms used the same patched path at one beam and 48 new tokens. Those arms did not independently repeat the parity flag.</p>
<p>The two patch error messages came from reverse dry-runs on a fresh checkout. Bootstrap then checks and applies each patch forward and exits if either step fails. The successful <code>source_only</code> inference and token parity corroborate the applied model patch. Transformers also reported that the fixed translation prompt was not truncated because the tokenizer had no configured maximum. Generation still used the explicit 100- or 48-token cap.</p>
<p>The run completed at <strong>08:42 Cairo time</strong>, about 17m 17s after the runner started. API rate was <strong>$1.09/h</strong>, or about <strong>$0.31 for the runner interval</strong> before provisioning time, storage, and tax. The persisted state is <code>requesting_stop</code>. The watchdog log confirms it armed for <strong>11:45 Cairo time</strong>. A read-only check of the matching Pod in RunPod showed a pause indicator, <strong>$0.00/hr</strong>, and a Start button. This confirms the Pod was stopped. The logs do not capture the Stop response, so the initiator, exact stop time, and final billed amount are unverified.</p>

<h2>2. Translation and performance</h2>
{table(["Arm", "Timed calls", "SacreBLEU", "Model p50 / p95 (ms)", "Generation p50 / p95 (ms)", "Peak allocated / reserved (MiB)"], metrics)}
<p>SacreBLEU used one reference, mixed case, 13a tokenization and exponential smoothing (<code>nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1</code>). These subset results are descriptive. No chrF, BLEURT or qualified ASL panel score was recorded. GPU figures are PyTorch allocator peaks. Device-wide peak power, memory and utilization were not captured. Cold model load took 30–32 seconds per process.</p>
<p>A one-pass English reference review classifies <strong>{counts["Preserved"]} preserved, {counts["Partial"]} partial, and {counts["Wrong"]} wrong</strong>. Preserved means the main actionable meaning is retained. Partial means an identifiable proposition survives with a material omission, error or addition. Wrong means the main proposition is lost, reversed or replaced. This is a preliminary text comparison. A qualified ASL reviewer should adjudicate against the source video before any accuracy claim.</p>
<ul><li><strong>Representative successes.</strong> “See how that works?” → “See how it works.” The caption about an excuse to spend time with friends also retained the message.</li><li><strong>Representative failures.</strong> “Reverse layups” became “rescue diamonds.” “Two ounces of gin” became “two ounces of jeans.” A safety instruction involving the wall became “it's not using the wall.” The dealer example lost the winning condition and repeated “you've got a.” Some captions preserve topic words while corrupting the action, number or entity.</li></ul>

<figure><img src="phase2-results.svg" alt="Inference timings and two timelines of reference spans, partial updates and uncertain resets."><figcaption>Times in the lower panel show source progress, not wall-clock caption delivery. The top panel shows p50 and p95 for each arm.</figcaption></figure>

<h2>3. Matched continuous replay</h2>
<p>Same two original recordings, same RTMW track per comparison, one beam, and up to 48 new tokens. The six reference spans total three per recording. The bounded arm supplied each correct source interval, with the existing 256-frame cap and uniform subsampling if needed. The streamed arm supplied causal windows at 500 ms cadence and committed using the Phase 2 policy.</p>
<div class="scroll">{table(["Aligned sentence", "Reference", "Gold-bounded prediction"], paired_rows)}</div>
<p><strong>Accuracy and partial input.</strong> None of the six gold-bounded outputs preserves the entire meaning on preliminary text review. The derby count keeps the proposition but changes 270 to 280. The flight-school introduction invents an alternate identity even with the correct boundaries. Replay adds cross-sentence mixing and repetition. It drifts into “playing cards” in the derby track, and repeatedly says “they're very versatile” during the flight lesson. These are divergences from the supplied references, with perceived sign content still subject to human review. Zero committed sentences leaves no attributable final-caption SacreBLEU. The evidence shows a streaming failure on top of an existing semantic failure. Buffering, resets and partial evidence changed together, so their individual effects have not been isolated.</p>
<p><strong>Commitment.</strong> 78 partials, <strong>zero finals</strong>, four buffer-cap resets and two end-of-input uncertain events. All six reference finals are missing. Annotated inter-sentence gaps were 280 ms and 200 ms in derby, and zero in flight school. They provide little opportunity for the controller's 350 ms (punctuation path) or 650 ms (generic path) quiet requirement plus two identical hypotheses. Two flight spans also exceed the eight-second cap. The timing explains why the policy is a poor fit for these examples. It does not establish a general-purpose boundary detector's accuracy.</p>
<p><strong>Timing and stability.</strong> First nonempty text appeared at 0.603 s (“Okay.”) and 0.567 s (“Hi.”) from replay start. These are raw events, not first-useful latency. Emission minus latest consumed source time across 82 timed updates was <strong>{lag_p50:.0f} ms p50 / {p95:.0f} ms p95</strong>. No finalized-caption latency is measurable. Normalized erasure was <strong>0.789</strong> vs the 0.30 gate, with <strong>92.3 revisions per source minute</strong>. The runner skipped <strong>4/87 updates (4.6%)</strong>, within the 5% gate, and model-work real-time factors were <strong>0.705</strong> and <strong>0.696</strong>, within 0.8. Replay work p95 was <strong>731 ms</strong>, above 400 ms and the 500 ms schedule. Twenty-three of 82 calls exceeded 500 ms. Generation made up 94% of total replay model work. No idle-only video was included, so the zero reported idle finals gives no evidence on idle hallucinations.</p>
<p><strong>Scope.</strong> Two 24 fps studio tracks spanning 42.9 seconds constitute a pilot. These results exclude live pose extraction, upload, transport, UI rendering and signer generalization. The earlier laptop extraction measurement on these tracks was 3.7–4.5 fps. The preserved source manifest documents the cadence and timings. Two paired English references contained a misencoded apostrophe, corrected in the display only. The logged scores remain as run.</p>

<h2>4. Ranked bottlenecks and decision</h2>
{table(["Rank", "Constraint", "Evidence and engineering implication"], [
    (1, "Frozen translation quality", "Major meaning errors on published poses and 0/6 fully preserved on fresh RTMW poses with gold boundaries. First test source/pose fidelity. Model capacity and representation mismatch have not been isolated."),
    (2, "Sentence segmentation", "0/6 finals, short reference gaps and four cap resets. A pause-only policy cannot cover this pilot."),
    (3, "Pose extraction and compatibility", "Laptop extraction was 3.7–4.5 fps on 24 fps inputs. Pre-extracted and fresh-pose cohorts differ. A same-clip comparison is needed before attributing quality loss to RTMW."),
    (4, "Streaming controller", "Erasure 0.789 with frequent rewrite and no immutable finals. Expose uncertainty and prevent unbounded or misleading partial display."),
    (5, "Inference tail", "Replay p95 731 ms, generation 94% of model work. Mean work can keep up on pre-extracted input and memory headroom is large. Optimize decode cadence and token policy before hardware."),
    (6, "Desktop integration", "No live transport or render trace. It remains a demo prerequisite and cannot explain the offline quality failures."),
])}
<ul>
<li><strong>Primary C.</strong> Verify same-source pose compatibility and frozen semantic quality first. A useful diagnostic compares the publisher's poses with freshly extracted RTMW poses on the same source frames and references. Inspect joint ordering, orientation, confidence and frame coverage before touching the model. Profile decoder settings after semantic quality is adequate.</li>
<li><strong>A in parallel.</strong> Add signer-controlled start/commit and a bounded partial preview to the existing session interface. Record buffer resets and token erasure. Carry it into the demo only on material with verified semantic quality. It can improve timing and readability but cannot repair the gold-bound errors.</li>
<li><strong>B no-go.</strong> The frozen gold-bounded gate failed. Streaming-aware LoRA would confound the diagnosis, and two tracks do not establish a training target or justify compute. Do not train or tune on any Phase 2 test clips. If a later independent test shows good oracle meaning and a consistent streaming-only gap, revisit the existing training-only How2Sign and held-out grouped evaluation plan.</li>
<li><strong>Fallback D.</strong> If the same-source check validates the pose path and Uni-Sign still fails new gold-bound webcam trials, compare one available alternate backend on the same independent material. SHuBERT can serve as an offline comparator where approved access exists. Treat a vetted streaming provider as a separate access, consent and cost decision. If neither passes before the competition, show a clearly labeled, recorded replay and the measured limitations.</li>
</ul>

<h2>5. Phase 3 plan before November 29</h2>
{table(["Workstream", "Essential work", "Dependency / timing"], [
    ("AI model", "Review failures with an ASL-qualified reviewer. Freeze a small new consented webcam set with signer IDs and reference boundaries. Choose a matched publisher-vs-RTMW probe where source frames are available.", "Start locally now, 2–4 h. Required before a paid comparison."),
    ("Streaming translation", "Use the shared session API and simulate deadline, reset and explicit commit with recorded pose events. Add uncertain-state UX and fixed acceptance logging. Keep two Phase 2 tracks as diagnostic fixtures only.", "Parallel, 3–5 h locally. Optional after diagnosis: compare decode interval and token caps."),
    ("Desktop application", "Wire a Tauri caption pane to recorded/mock events, keep partial and final states distinct, and capture receive-to-render timestamps. Add direct SSH tunnel transport after model acceptance.", "Parallel, 3–6 h locally. Webcam transport and render timing are required for a live claim."),
    ("GPU infrastructure", "The prior Pod is stopped. Retain the archived environment, inputs and hashes. Prepare one warm single-process rerun with matched settings.", "30–60 min local prep. Only with explicit approval, budget 30–60 GPU min ($0.55–$1.10 compute at $1.10/h, plus storage/tax)."),
    ("Prototype demonstration", "Prepare a labeled replay and a fallback video. If the new independent oracle and streaming checks pass, run a consented webcam rehearsal and record actual render latency.", "Parallel, 1–2 h. Freeze demo scope by November 28 after gates."),
])}
<p><strong>Critical path.</strong> Verify same-source poses and independent gold-bounded meaning → pick backend and scope → test live capture-to-render latency before calling the demo live. For the qualified prototype gate, retain the approved 24-unit evaluation across at least three signers, 18/24 meaning-preserved gold-bounded units, 22/24 attributable finals, and the independent idle and latency checks. A smaller new pilot can inform selection without being reported as a passed gate. Local work requires no incremental GPU spend. Adapter training, export and generalized open-domain support are optional and outside the immediate run.</p>
<p><strong>Immediate next engineering action.</strong> Run a paired pose audit on the same source clips while an ASL reviewer checks the six full-reference failures. Use the result to decide whether any new GPU time should go to Uni-Sign or an alternate backend.</p>

<details><summary>All 32 official test references and predictions</summary>
<p>Raw sentence strings from the captured JSONL. Assessments are the preliminary English-reference review described above.</p>
<div class="scroll">{table(["#", "Clip ID", "Reference", "Prediction", "Text review", "Work (ms)"], translations)}</div></details>
<p><small>Reproduction: run <code>python phase2-analysis/build_report.py</code> with ReportLab available. Inputs are the three event logs and summaries in the original results folder and the archived continuous source manifest. Banner and tables use the saved run configuration. Variability and latency percentiles use the same nearest-rank definition as <code>phase2/score.py</code>.</small></p>
</main></body></html>"""
    (HERE / "experiment-report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    render()
