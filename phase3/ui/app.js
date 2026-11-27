import {beginEvidence,evidenceEvent,frameMetric,startRecording,stopRecording,finishEvidence,updateCost} from "./evidence.js";
const $ = id => document.getElementById(id);
const owner = crypto.randomUUID();
let catalog = {}, position = 0, duration = 0, deadlineTimer, finishing = false;
const ui = Object.fromEntries(["mode","mode-label","track","track-label","preview","start","stop","commit","copy",
  "status","video","empty","empty-title","empty-text","partial","finals","no-finals"].map(id => [id, $(id)]));
let capabilities, tracks, stream, active = false, busy = false, pending = null, timer, epoch = 0, beginning;
let partial = "", finals = [], lastSlot = -1;
const frameQueue = [];
const maxQueuedFrames = 6;
let captureFps = 3, batchSize = 1, sentenceLimit = 0, sentenceClosed = false;
let sentenceFrames = 0, sentenceGap = false, sentenceBatchTimer;
let frameCallback, commitCutoff, bufferDrops = 0, lastPresentedFrame = -1;
let gpuRequest = false, gpuError = "", currentOptions = "", refreshingGPU;
let discoveringGPU = false, discoveryError = "";
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

const live = () => ui.mode.value === "live";
const sentence = () => ui.mode.value === "sentence";
const camera = () => live() || sentence();
const prerecorded = () => ui.mode.value === "prerecorded";
const say = text => { ui.status.textContent = text; };

async function api(path, body) {
  const headers={"X-Session":owner};
  if (capabilities?.token) headers["X-Control"]=capabilities.token;
  const response = await fetch(`/api/${path}`, body === undefined ? {headers} : {
    method:"POST", headers:{...headers,"Content-Type":"application/json"}, body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error);
  return payload;
}

function render() {
  ui.partial.textContent = partial || "—";
  ui.finals.replaceChildren(...finals.map(text => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
  ui["no-finals"].hidden = finals.length > 0;
  ui.copy.disabled = !finals.length && !partial;
  ui.start.disabled = !capabilities || active || busy ||
    ((camera() || prerecorded()) && !capabilities.live) ||
    (sentence() && !capabilities.gpu?.runtime?.sentence_frames) ||
    (prerecorded() && (!catalog[ui.track.value]?.available || !$("segment").value));
  ui.stop.disabled = !active || busy;
  ui.commit.disabled = !camera() || !active || busy || (sentence() && sentenceFrames < 3);
  ui.mode.disabled = active || busy;
  ui.track.disabled = active || busy;
  $("segment").disabled = active || busy;
  renderGPU();
  ui.start.textContent=sentence()?"Start Signing":live()?"Start Live Test":prerecorded()?"Translate sentence":"Play";
  ui.commit.textContent=sentence()?"Translate":"Commit sentence";
  ui.stop.textContent=camera()?"End Test and Stop GPU":"Pause";
  if(camera() && !active) $("recordStart").disabled=true;
}

function display(event) {
  if (!event) return;
  if (event.kind === "partial") partial = event.text;
  if (event.kind === "uncertain") {
    partial = "";
    say(camera() ? (event.reason === "capture_gap" ? "Capture gap. Translation requires a complete sentence." :
      "Sentence not committed. Try a shorter, clearly framed signing turn.") : "Replay: sentence was not committed.");
  }
  if (event.kind === "final") {
    finals.push(event.text);
    partial = "";
    say(camera() ? "Sentence committed." : "Replay: recorded final caption.");
  }
  render();
}

function changeMode() {
  $("recorded").pause();
  $("recorded").controls = prerecorded();
  if(camera()) $("recorded").hidden=true;
  $("replayTools").hidden=camera() || prerecorded();
  $("segment-label").hidden=!prerecorded();
  $("prerecordedReference").hidden=!prerecorded();
  if (!camera() && stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    ui.video.srcObject = null;
  }
  ui["mode-label"].textContent = sentence() ? "Sign & Translate" : live() ? "Live camera" : prerecorded() ? "Prerecorded translation" : "Recorded replay";
  ui["mode-label"].classList.toggle("live", camera());
  ui["track-label"].hidden = camera();
  ui.preview.hidden = !camera();
  ui.video.hidden = !stream;
  ui.empty.hidden = !!stream;
  ui["empty-title"].textContent = camera() ? "Camera preview" : "Recorded replay";
  ui["empty-text"].textContent = camera() ? "Enable your camera to preview the shot." :
    prerecorded() ? "Choose an annotated sentence to preview its source video." :
      "Play the actual caption events from the Phase 2 evaluation.";
  if (prerecorded()) ui["empty-title"].textContent = "Prerecorded translation";
  say(camera() ? (capabilities?.live ? (sentence() ? "Capture one complete sentence, then click Translate." :
    "Live model connected. Preview uses a reduced frame rate.") :
    "Camera preview is available. Live translation needs the GPU backend.") :
    prerecorded() ? "Choose a complete annotated sentence. Frames are processed at their source cadence." :
    "Recorded model outputs. Camera frames are not used.");
  partial = ""; finals = [];
  if(prerecorded()) selectPrerecordedTrack();
  else if(!camera()) selectTrack();
  render();
}

async function preview() {
  if (stream?.active && ui.video.videoWidth) return;
  stream?.getTracks().forEach(track=>track.stop());
  stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}}, audio:false});
  ui.video.srcObject = stream;
  try {await ui.video.play();}
  catch(e) {stream.getTracks().forEach(track=>track.stop());stream=null;ui.video.srcObject=null;throw e;}
  ui.video.hidden = false;
  ui.empty.hidden = true;
}
window.addEventListener("pagehide",()=>stream?.getTracks().forEach(track=>track.stop()));

function captureFrame(now, presentedFrame) {
  if (!active || sentenceClosed || !ui.video.videoWidth || presentedFrame <= lastPresentedFrame) return;
  const at_ms = now - beginning;
  if (at_ms < 0 || at_ms >= 60_000) return;
  const slot = Math.floor(at_ms * captureFps / 1000);
  if (slot <= lastSlot) return;
  if (sentence() && slot >= sentenceLimit) {
    if (frameCallback !== undefined) ui.video.cancelVideoFrameCallback(frameCallback);
    clearInterval(timer);
    frameCallback = undefined;
    sentenceClosed = true;
    sendNextFrame();
    say("Capture complete. Click Translate.");
    render();
    return;
  }
  if (sentence() && slot !== lastSlot + 1) sentenceGap = true;
  lastSlot = slot;
  lastPresentedFrame = presentedFrame;
  if (frameQueue.length === maxQueuedFrames) {
    bufferDrops++;
    if (sentence()) {
      sentenceGap = true;
      evidenceEvent({kind:"telemetry", reason:"capture_buffer_overflow", slot});
      return;
    }
    if (commitCutoff !== undefined && frameQueue[0].slot <= commitCutoff) {
      evidenceEvent({kind:"telemetry", reason:"capture_buffer_overflow", slot});
      return;
    }
    const discarded = frameQueue.shift();
    evidenceEvent({kind:"telemetry", reason:"capture_buffer_overflow", slot:discarded.slot});
  }
  canvas.width = Math.min(640, ui.video.videoWidth);
  canvas.height = Math.round(canvas.width * ui.video.videoHeight / ui.video.videoWidth);
  const encodeStart = performance.now();
  context.drawImage(ui.video, 0, 0, canvas.width, canvas.height);
  const jpeg = canvas.toDataURL("image/jpeg", .8).split(",")[1];
  frameQueue.push({slot, at_ms, captured:now, jpeg, captureEncodeMs:performance.now()-encodeStart});
  if (sentence()) {
    sentenceFrames++;
    if (sentenceFrames === 3) render();
  }
  sendNextFrame();
}

function sendNextFrame(flush = false) {
  if (pending || !active || !frameQueue.length || (sentence() && sentenceGap)) return;
  if (sentence() && !sentenceClosed && !flush && frameQueue.length < batchSize) {
    sentenceBatchTimer ??= setTimeout(() => {
      sentenceBatchTimer = undefined;
      sendNextFrame(true);
    }, 185);
    return;
  }
  clearTimeout(sentenceBatchTimer);
  sentenceBatchTimer = undefined;
  let count = 0;
  while (count < batchSize && count < frameQueue.length &&
         (commitCutoff === undefined || frameQueue[count].slot <= commitCutoff)) count++;
  if (!count) return;
  const frames = frameQueue.splice(0, count);
  const generation = epoch;
  pending = deliverFrames(frames, generation).catch(async error => {
    if (epoch === generation) {
      await endTest("Frame or connection failure","frame_failed");
      say(error.message);
    }
  }).finally(() => {
    pending = null;
    if (epoch === generation) sendNextFrame();
  });
}

async function deliverFrames(frames, generation) {
  const sent=performance.now();
  const request = frames.map(({slot, at_ms, jpeg}) => ({slot, at_ms, jpeg}));
  const results = batchSize > 1 ? (await api("frames", {frames:request})).frames :
    [await api("frame", request[0])];
  if (!Array.isArray(results) || results.length !== frames.length ||
      (batchSize > 1 && results.some((value,i) => value.slot !== frames[i].slot || value.at_ms !== frames[i].at_ms))) {
    throw Error("Frame batch did not match the original capture timestamps");
  }
  if (epoch !== generation) return;
  const roundTrip=performance.now()-sent;
  for (const [index, result] of results.entries()) {
      const {slot, at_ms, captured, captureEncodeMs} = frames[index];
      frameMetric({slot,atMs:at_ms,batchSize:frames.length,roundTripMs:roundTrip,queueMs:Math.max(0,sent-captured-captureEncodeMs),captureEncodeMs,poseMs:result.pose_ms,
        jpegMs:result.jpeg_ms,workerMs:result.worker_ms,
        modelMs:result.event?.work_ms,dropped:result.dropped,bufferDrops});
      evidenceEvent(result.event); display(result.event);
      $("metrics").textContent=`Batch ${frames.length} · round trip ${Math.round(roundTrip)} ms · pose ${Math.round(result.pose_ms||0)} ms · model ${result.event?.work_ms?.toFixed(0)??"idle"} ms · dropped slots ${result.dropped??0}`;
  }
}

function beginCapture() {
  if (ui.video.requestVideoFrameCallback) {
    function next(now, frame) {
      if (!active || !camera() || sentenceClosed) return;
      captureFrame(frame.presentationTime ?? now, frame.presentedFrames);
      if (active && !sentenceClosed) frameCallback = ui.video.requestVideoFrameCallback(next);
    }
    frameCallback = ui.video.requestVideoFrameCallback(next);
  } else {
    timer = setInterval(() => captureFrame(performance.now(),
      ui.video.getVideoPlaybackQuality().totalVideoFrames), 40);
  }
}

function selectTrack() {
  const name=ui.track.value, entry=catalog[name], video=$("recorded");
  if(!name) return;
  position=0;
  duration=Math.max((tracks[name].at(-1)?.emit_ms||0)/1000, entry?entry.source.frames/entry.source.source_fps:0);
  $("seek").max=duration;
  video.hidden=!entry?.available; ui.empty.hidden=!!entry?.available;
  if(entry?.available) {video.src="/media/"+encodeURIComponent(name); video.currentTime=entry.source.source_start_ms/1000;}
  else ui["empty-text"].textContent="Original footage unavailable locally. Caption timeline only.";
  paintReplay();
}

function selectPrerecordedTrack() {
  const entry = catalog[ui.track.value];
  $("segment").replaceChildren(new Option("Choose a sentence", ""),
    ...(entry?.segments ?? []).map((segment,index) => new Option(`Sentence ${index+1}`, index)));
  $("prerecordedReference").textContent = "";
  $("recorded").hidden = true;
  ui.empty.hidden = false;
  render();
}
function paintReplay() {
  const rows=tracks[ui.track.value]||[], entry=catalog[ui.track.value];
  partial=""; finals=[];
  for(const e of rows) {
    if(e.emit_ms>position*1000) break;
    if(e.kind==="final") finals.push(e.text);
    partial=e.kind==="partial"?e.text:"";
  }
  const ref=entry?.segments.find(s=>position*1000>=s.start_ms && position*1000<s.end_ms);
  $("reference").textContent=(ref?.reference||"—").replaceAll("\u00e2\u0080\u0099","’");
  $("seek").value=position; $("progress").textContent=position.toFixed(1)+" / "+duration.toFixed(1)+" s";
  render();
}
function seek(to) {
  position=to; beginning=performance.now()-position*1000;
  const entry=catalog[ui.track.value];
  if(entry?.available) {
    const span=entry.source.frames/entry.source.source_fps;
    $("recorded").currentTime=entry.source.source_start_ms/1000+Math.min(to,span-1/entry.source.source_fps);
    if(active && to<span) $("recorded").play().catch(e=>say(e.message));
    if(to>=span) $("recorded").pause();
  }
  paintReplay();
}
async function runReplay() {
  if(position>=duration) seek(0);
  const entry=catalog[ui.track.value], video=$("recorded");
  beginning=performance.now()-position*1000;
  if(entry?.available && position<entry.source.frames/entry.source.source_fps) await video.play();
  timer=setInterval(()=>{
    const span=entry?entry.source.frames/entry.source.source_fps:0;
    if(entry?.available && position<span && !video.ended) {
      position=Math.max(0,video.currentTime-entry.source.source_start_ms/1000);
      beginning=performance.now()-position*1000;
      if(position>=span) {video.pause();video.currentTime=(entry.source.source_start_ms/1000)+span-1/entry.source.source_fps;}
    } else position=(performance.now()-beginning)/1000;
    if(position>=duration) {position=duration;video.pause();clearInterval(timer);active=false;say("Recorded replay complete.");}
    paintReplay();
  },32);
}

async function start() {
  if (ui.start.disabled) return;
  busy = true;
  render();
  try {
    if (prerecorded()) {
      partial = "";
      finals = [];
      say("Extracting poses from the complete sentence, then running Uni-Sign…");
      const result = await api("translate-video", {track:ui.track.value,segment:Number($("segment").value)});
      if (result.error) throw Error(result.error);
      if (result.event.kind === "final") finals.push(result.event.text);
      say(result.event.kind === "final" ? "Prerecorded translation complete." :
        "Uni-Sign did not produce a valid caption for this sentence.");
      $("metrics").textContent = `${result.pose.frames} real frames · ${result.pose.source_fps} FPS source · ` +
        `Pose ${result.pose.pose_work_fps.toFixed(2)} FPS · Model ${Math.round(result.event.work_ms ?? 0)} ms`;
      return;
    }
    if (camera()) {
      await preview();
      if (!ui.video.requestVideoFrameCallback && !ui.video.getVideoPlaybackQuality) {
        throw Error("This browser cannot identify distinct camera frames");
      }
      const session = await api("start", {mode:sentence()?"sentence":"live"});
      const runtime = session.runtime ?? {};
      batchSize = runtime.batch_frames === 3 ? 3 : 1;
      captureFps = sentence() ? runtime.sentence_fps : capabilities.fps;
      sentenceLimit = sentence() ? runtime.sentence_frames : 0;
      if (sentence() && (captureFps !== 6 || sentenceLimit !== 48 || batchSize !== 3)) {
        throw Error("This worker does not support Sign & Translate");
      }
      sentenceClosed = false;
      sentenceFrames = 0;
      sentenceGap = false;
      beginEvidence(runtime, captureFps, sentence()?"sentence":"live");
      beginning = performance.now();
      deadlineTimer=setTimeout(()=>endTest("60-second deadline","test_deadline"),60000);
      $("recordStart").disabled=false;
      if($("recordEnabled").checked) {
        try {startRecording(ui.video,()=>partial||finals.at(-1)||"");} catch(e){$("recordStatus").textContent=e.message;}
      }
      lastSlot = -1;
      lastPresentedFrame = -1;
      bufferDrops = 0;
      frameQueue.length = 0;
      say(sentence() ? "Sign one sentence within 8 seconds, then click Translate." :
        "Live capture running. Commit each completed sentence.");
    } else {
      await runReplay();
      say("Playing recorded model output.");
    }
    epoch++;
    if(camera()) {partial = ""; finals = [];}
    active = true;
    if(camera()) beginCapture();
  } catch (error) {
    say(error.message);
    if(camera() && capabilities.gpu?.can_stop) {
      await endTest("Live test could not start","connection");
    }
  } finally {
    busy = false;
    render();
  }
}

async function control(action) {
  if (busy || !active) return;
  busy = true;
  if (live() && action === "commit") commitCutoff = lastSlot;
  render();
  try {
    if (sentence() && action === "commit") {
      sentenceClosed = true;
      evidenceEvent({kind:"telemetry", reason:"sentence_translate_requested", frames:sentenceFrames});
      if (sentenceGap) {
        const event = {kind:"uncertain", reason:"capture_gap"};
        evidenceEvent(event);
        display(event);
        await endTest("Incomplete sentence capture");
        return;
      }
      say("Translating your complete sentence…");
      const generation = epoch;
      sendNextFrame(true);
      while (pending && epoch === generation) await pending;
      if (!active || epoch !== generation) return;
      const {event} = await api("commit", {});
      if (epoch !== generation) return;
      evidenceEvent(event);
      display(event);
      await endTest("Sign & Translate complete");
      return;
    }
    if(action!=="stop" || !camera()) {
      while (pending) await pending;
      if (!active) return;
    }
    if(camera() && action==="stop") {await endTest("Ended by operator");return;}
    if (live()) {const event=(await api(action, {})).event; evidenceEvent(event);display(event);}
    if (action === "stop") {
      active = false;
      epoch++;
      clearInterval(timer);
      $("recorded").pause();say("Recorded replay paused.");
    }
  } catch (error) {
    if (sentence() && active) await endTest("Sentence translation failed","frame_failed");
    say(error.message);
  } finally {
    commitCutoff = undefined;
    if (live()) sendNextFrame();
    busy = false;
    render();
  }
}

ui.mode.addEventListener("change", changeMode);
ui.track.onchange=()=>prerecorded()?selectPrerecordedTrack():selectTrack();
$("segment").onchange=()=>{
  const entry=catalog[ui.track.value], selected=$("segment").value;
  const segment=selected === "" ? null : entry?.segments?.[Number(selected)];
  $("prerecordedReference").textContent = segment ? "English reference · " +
    segment.reference.replaceAll("\u00e2\u0080\u0099","’") : "";
  if (segment && entry.available) {
    const video=$("recorded"), url="/media/"+encodeURIComponent(ui.track.value);
    if (video.getAttribute("src") !== url) video.src=url;
    video.currentTime=(entry.source.source_start_ms+segment.start_ms)/1000;
    video.hidden=false;
    ui.empty.hidden=true;
  } else {
    $("recorded").hidden=true;
    ui.empty.hidden=false;
  }
  render();
};
$("seek").oninput=()=>seek(Number($("seek").value));
$("restart").onclick=()=>{seek(0);if(!active) start();else if(catalog[ui.track.value]?.available) $("recorded").play().catch(e=>say(e.message));};
async function endTest(reason,stopReason,alreadyStopping=false) {
  if(finishing) return;
  finishing=true;clearInterval(timer);clearTimeout(deadlineTimer);epoch++;
  clearTimeout(sentenceBatchTimer);
  sentenceBatchTimer = undefined;
  if (frameCallback !== undefined) ui.video.cancelVideoFrameCallback(frameCallback);
  frameCallback = undefined;
  commitCutoff = undefined;
  frameQueue.length = 0;
  active=false;render();
  const stopFallback=stopReason||alreadyStopping?null:setTimeout(()=>api("gpu/stop",{reason:"finish_failed"}).catch(()=>{}),1200);
  try {
    if(stopReason) await api("gpu/stop",{reason:stopReason});
    else if(!alreadyStopping) { const response=await api("stop",{});evidenceEvent(response.event);display(response.event); }
  } catch(e) {say(e.message);await api("gpu/stop",{reason:stopReason||"finish_failed"}).catch(()=>{});}
  finally {
    clearTimeout(stopFallback);
    try { await finishEvidence(evidenceGPU(capabilities?.gpu),reason); }
    catch(e){$("recordStatus").textContent="Evidence save failed: "+e.message;}
    stream?.getTracks().forEach(t=>t.stop());stream=null;ui.video.srcObject=null;ui.video.hidden=true;ui.empty.hidden=false;
    finishing=false;render();
  }
}
$("recordStart").onclick=()=>{try {startRecording(ui.video,()=>partial||finals.at(-1)||"");}catch(e){say(e.message);}};
$("recordStop").onclick=()=>stopRecording().catch(e=>say(e.message));
const money = value => `$${value.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 9})}`;
const storageRate = row => {
  const input = $("gpuStorage");
  return input.value || input.validity.badInput ? input.valueAsNumber : row.storage;
};
const validStorage = row => Number.isFinite(storageRate(row)) && storageRate(row) > 0 && storageRate(row) >= row.storage;
const automatic = () => $("gpuPod").value === "@auto";
const runningPods = gpu => (gpu?.candidates ?? []).filter(row => row.status === "RUNNING");
const spendingCap = row => {
  const input = $("gpuCeiling").valueAsNumber;
  if (Number.isFinite(input) && input > 0) return Math.min(input, row.cap ?? input);
  return automatic() || $("gpuCeiling").value || $("gpuCeiling").validity.badInput ? NaN : row.cap;
};
// Avoid a binary floating-point carry changing the amount shown in the confirmation.
const approvedTotal = row => Number((row.rate + storageRate(row)).toPrecision(15));
const selectedPod = () => {
  const gpu = capabilities?.gpu;
  const rows = runningPods(gpu);
  if (!automatic()) return rows.find(row => row.pod === $("gpuPod").value);
  return rows.filter(row => !row.issue && validStorage(row) && (!gpu.active_pod || gpu.active_pod === row.pod) &&
    approvedTotal(row) <= spendingCap(row)).sort((a,b) => approvedTotal(a) - approvedTotal(b) || a.pod.localeCompare(b.pod))[0];
};

function automaticBlocker(gpu) {
  const rows = runningPods(gpu);
  if (!rows.length) return gpu.active_pod ? `Waiting for Pod ${gpu.active_pod} to be ready, or check its status in RunPod.` :
    gpu.discovery_detail || (gpu.key_configured ? "Start an existing Pod in RunPod. It will appear here once running." : "");
  const available = rows.filter(row => !gpu.active_pod || gpu.active_pod === row.pod);
  if (!available.length) return `Check active Pod ${gpu.active_pod} on the shared volume in RunPod.`;
  const eligible = available.filter(row => !row.issue);
  if (!eligible.length) return `${available[0].pod} · ${available[0].issue}. Select a Pod to see its details.`;
  if (!Number.isFinite(spendingCap(eligible[0]))) return "Enter a positive spending ceiling to select a Pod automatically.";
  const priced = eligible.filter(validStorage);
  if (!priced.length) return "Clear the storage override to use the calculated allowance, or enter a higher rate.";
  const cheapest = priced.reduce((a,b) => approvedTotal(a) < approvedTotal(b) ? a : b);
  return `No eligible Pod fits its ceiling. Lowest total is ${money(approvedTotal(cheapest))}/hour for ${cheapest.gpu} (${cheapest.pod}). Review the spending ceiling or select a Pod for details.`;
}

const evidenceGPU = gpu => gpu && Object.fromEntries(
  ["gpu","pod","began","seconds","bill_seconds","rate","rate_cap","current_rate","current_total","cost","state","detail",
   "approved_total","windows_guard","remote_guard"].map(field => [field, gpu[field]]));

function renderGPU() {
  const gpu = capabilities?.gpu;
  $("gpuConfig").hidden = !gpu;
  $("gpuPin").hidden = !gpu?.pin_command;
  $("emergency").disabled = !gpu?.can_stop && !gpuRequest;
  if (!gpu) {
    $("gpuStatus").textContent = "GPU controls are available in the Windows app.";
    $("gpuStart").disabled = true;
    return;
  }
  const rows = runningPods(gpu);
  const options = JSON.stringify(rows);
  if (options !== currentOptions) {
    const previous = $("gpuPod").value;
    $("gpuPod").replaceChildren(new Option("Automatic under ceiling", "@auto"),
      ...rows.map(row => new Option(`${row.gpu} · ${row.pod}`, row.pod)));
    if (rows.some(row => row.pod === previous)) $("gpuPod").value = previous;
    currentOptions = options;
  }
  if (!gpu.key_configured) $("gpuCredentials").open = true;
  $("gpuPinCommand").textContent = gpu.pin_command || "";
  const row = selectedPod();
  const total = row?.rate != null && validStorage(row) ? approvedTotal(row) : null;
  const ceiling = row ? spendingCap(row) : NaN;
  const conflict = gpu.active_pod && gpu.active_pod !== row?.pod;
  $("gpuQuote").textContent = row ? [
    automatic() ? `Selected ${row.gpu} · ${row.pod}` : null,
    "1 GPU · EUR-IS-1 · volume 4q8k7ibb17 at /workspace",
    row.rate != null ? `Compute ${money(row.rate)}/hour` : "Compute quote unavailable",
    row.storage != null ? `${row.disk} GB container disk · Allowance ${money(row.storage)}/hour` : null,
    total != null ? `Total ${money(total)}/hour` : null,
    Number.isFinite(ceiling) ? `Ceiling ${money(ceiling)}/hour` : "Enter a spending ceiling",
    row.issue || (!validStorage(row) ? "Clear the storage override or enter at least the calculated allowance" :
      total > ceiling ? "Total exceeds the ceiling" : null),
    conflict ? `Shared volume in use by ${gpu.active_pod}` : null,
  ].filter(Boolean).join(" · ") : automaticBlocker(gpu);
  const configure = gpu.busy || gpuRequest;
  $("gpuStart").disabled = configure || !gpu.available || !row || !!row.issue || !!conflict ||
    !Number.isFinite(total) || !Number.isFinite(ceiling) || total > ceiling || active || busy;
  for (const id of ["gpuAuth","gpuKey","gpuRemember","gpuPod","gpuStorage","gpuPodId","gpuCeiling"]) $(id).disabled = configure;
  $("gpuForget").disabled = configure || !gpu.key_configured;
  $("gpuRefresh").disabled = configure || !gpu.key_configured;
  $("gpuLookup").disabled = configure || !gpu.key_configured || !$("gpuPodId").value.trim();
  $("gpuTrust").disabled = gpuRequest || !gpu.pin_command;
  const status = [];
  if (gpu.pod) {
    status.push(`${gpu.gpu} · ${gpu.pod}`, gpu.pod_status);
    if (gpu.approved_total != null) status.push(`Approved ${money(gpu.approved_total)}/hour`);
    if (gpu.began && Number.isFinite(gpu.current_total)) {
      status.push(`Current total ${money(gpu.current_total)}/hour`);
    }
    if (gpu.began) status.push(`${gpu.seconds}s elapsed`, `${gpu.remaining}s of 15m left`);
    if (gpu.setup_remaining != null) status.push(`${gpu.setup_remaining}s setup left`);
    if (gpu.test_remaining != null) status.push(`${gpu.test_remaining}s test left`);
    if (gpu.began) status.push(`${money(gpu.cost)} estimated`);
  }
  if (gpu.busy || gpu.pod) {
    status.push(`Windows watchdog ${gpu.windows_guard ? "armed" : "inactive"}`,
      `Pod watchdog ${gpu.remote_guard ? "armed" : "inactive"}`);
  }
  if (gpu.runtime?.pose_provider) {
    status.push(`Pose ${gpu.runtime.pose_provider}${gpu.runtime.pose_fallback ? " · " + gpu.runtime.pose_fallback : ""}`);
    if (Number.isFinite(gpu.runtime.warmup_ms)) status.push(`Model warmed in ${Math.round(gpu.runtime.warmup_ms)} ms`);
  }
  if (gpu.fingerprint && gpu.busy) status.push(gpu.fingerprint);
  const fallback = gpu.key_configured ? `${rows.length} running Pods found` : "Enter a scoped RunPod API key to find your Pods";
  const detail = gpu.state === "shutdown_unverified" ? "Shutdown needs verification" : gpu.detail;
  status.unshift(gpuRequest ? "Updating GPU Control Center…" : detail || (gpu.discovery_detail ? "Pod discovery complete" : fallback));
  if (gpu.state === "watching_delayed_start") {
    const seconds = gpu.start_remaining;
    status.push(seconds == null ? "Waiting for the Start response" :
      `Monitoring delayed startup · ${Math.floor(seconds / 60)}m ${seconds % 60}s left`);
    if (gpu.cancel_requested) status.push("Stop requested. A late start will be stopped when its generation is verified.");
  }
  $("gpuStatus").textContent = status.filter(Boolean).join(" · ");
  $("warning").textContent = gpu.state === "shutdown_unverified" ?
    `STOP UNVERIFIED. ${gpu.detail} ${gpu.pending_unknown ?
      `Check ${gpu.pending_pod || gpu.pod || "the selected Pod"} in RunPod, then use Stop GPU to reconcile.` :
      "Use Stop GPU to check RunPod again."}` : gpuError || discoveryError;
}

async function gpuAction(path, body, followup) {
  gpuRequest = true;
  gpuError = "";
  render();
  try {
    await api(path, body);
    if (followup) await followup();
    if (path === "gpu/discover" || path === "gpu/credentials") discoveryError = "";
  } catch (error) {
    gpuError = error.message;
  } finally {
    await refreshingGPU?.catch(() => {});
    await refreshGPU().catch(() => { gpuError ||= "Desktop connection lost. Check shutdown in RunPod."; });
    gpuRequest = false;
    render();
  }
}

$("gpuPod").onchange = $("gpuStorage").oninput = $("gpuPodId").oninput = $("gpuCeiling").oninput = renderGPU;
$("gpuAuth").onclick=()=>{
  const key = $("gpuKey").value;
  $("gpuKey").value = "";
  gpuAction("gpu/credentials", {key, remember:$("gpuRemember").checked},
    async () => { $("gpuCredentials").open = false; await api("gpu/discover", {}); });
};
$("gpuForget").onclick=()=>gpuAction("gpu/credentials",{forget:true});
$("gpuRefresh").onclick=()=>gpuAction("gpu/discover",{});
$("gpuLookup").onclick=()=>gpuAction("gpu/discover",{pod:$("gpuPodId").value.trim()});
$("gpuCopyCommand").onclick=async()=>{
  try { await navigator.clipboard.writeText($("gpuPinCommand").textContent); }
  catch { gpuError = "Select and copy the command above."; renderGPU(); }
};
$("gpuTrust").onclick=()=>{
  const output = $("gpuPinOutput").value;
  $("gpuPinOutput").value = "";
  gpuAction("gpu/pin",{trusted:true, output});
};
$("gpuStart").onclick=()=>{
  const row = selectedPod();
  if ($("gpuStart").disabled || !row) return;
  const total = approvedTotal(row);
  const ceiling = spendingCap(row);
  if (!confirm(`Attach to this running Pod and authorize its shutdown?\n\n${row.gpu} · ${row.pod} · ${row.status}${row.generation ? "\nStarted " + row.generation : ""}
1 GPU · EUR-IS-1 · volume 4q8k7ibb17 at /workspace
Total ${money(total)}/hour, including ${money(storageRate(row))}/hour for the container disk.
Ceiling ${money(ceiling)}/hour. The shared network volume is billed separately.
Shutdown begins before 15 minutes from the actual Pod start.`)) return;
  if (!prerecorded() && !sentence()) {
    ui.mode.value = "live";
    changeMode();
  }
  gpuAction("gpu/start",{confirmed:true,pod:row.pod,status:row.status,generation:row.generation,
                         compute:row.rate,storage:storageRate(row),total,ceiling});
};
$("emergency").onclick=async()=>{
  const gpu = capabilities?.gpu;
  let console_confirmed = false;
  if (gpu?.pending_unknown) {
    if (gpu.pending_remaining > 0) {
      gpuError = `The delayed-start window has ${gpu.pending_remaining}s left. Check again once it ends.`;
      return renderGPU();
    }
    console_confirmed = confirm(`Did you check the RunPod console and verify ${gpu.pending_pod} is Exited with no pending Start?`);
    if (!console_confirmed) return;
  }
  try {
    await api("gpu/stop",{console_confirmed});
    if(active&&camera()) await endTest("Emergency stop",null,true);
    await refreshGPU();
  } catch {
    gpuError = "Shutdown unverified. Stop the selected Pod in the RunPod console.";
    renderGPU();
  }
};
function refreshGPU() {
  if (refreshingGPU) return refreshingGPU;
  refreshingGPU = (async () => {
    const last=capabilities?.gpu;
    capabilities=await api("status");
    const gpu=capabilities.gpu;
    if(gpu) {
      if(active && camera() && !gpu.ready) await endTest(gpu.detail||"Backend disconnected",null,true);
      if(["stopped","shutdown_unverified"].includes(gpu.state)&&last?.state!==gpu.state) await updateCost(evidenceGPU(gpu));
    }
    if(live()) ui["mode-label"].textContent=active?"Live inference":capabilities.live?"Live model ready":"Camera preview · disconnected";
    render();
  })().finally(() => { refreshingGPU = undefined; });
  return refreshingGPU;
}
async function discoverRunningPods() {
  const gpu = capabilities?.gpu;
  if (document.hidden || discoveringGPU || gpuRequest || !gpu?.key_configured || !gpu.available) return;
  discoveringGPU = true;
  try {
    await api("gpu/discover", {});
    discoveryError = "";
    await refreshingGPU?.catch(() => {});
    await refreshGPU();
  } catch (error) {
    if (!gpuRequest && capabilities?.gpu?.available) discoveryError = error.message;
    renderGPU();
  } finally {
    discoveringGPU = false;
  }
}
ui.preview.addEventListener("click", () => preview().catch(error => say(error.message)));
ui.start.addEventListener("click", start);
ui.stop.addEventListener("click", () => control("stop"));
ui.commit.addEventListener("click", () => control("commit"));
ui.copy.addEventListener("click", async () => {
  try {
    const content = [...finals, ...(partial ? [`[Partial] ${partial}`] : [])].join("\n");
    await navigator.clipboard.writeText(camera() || prerecorded() ? content : `[Recorded replay]\n${content}`);
    say("Caption text copied.");
  } catch (error) { say(error.message); }
});
document.addEventListener("keydown", event => {
  if (event.repeat) return;
  if (event.code === "F8") {
    event.preventDefault();
    active ? control("stop") : start();
  } else if (event.ctrlKey && event.code === "Enter" && camera() && !ui.commit.disabled) {
    event.preventDefault();
    control("commit");
  }
});

try {
  [capabilities, tracks,catalog] = await Promise.all([api("status"), api("replay"),api("catalog")]);
  for (const id of Object.keys(tracks)) {
    const title = id.startsWith("fzue") ? "Roller derby" : id.startsWith("G3OM") ? "Flight school" : id;
    ui.track.add(new Option(title, id));
  }
  changeMode();
  discoverRunningPods();
  setInterval(discoverRunningPods, 8000);
  document.addEventListener("visibilitychange", discoverRunningPods);
  setInterval(()=>refreshGPU().catch(()=>{
    capabilities.live=false;
    if(capabilities.gpu) {
      capabilities.gpu.available=false;
      gpuError="Connection lost. Verify shutdown in the RunPod console.";
    }
    if(active&&camera()) endTest("Connection lost","connection");
    render();
  }),2000);
} catch (error) { say(error.message); }
