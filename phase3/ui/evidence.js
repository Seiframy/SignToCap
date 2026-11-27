// All webcam evidence stays in this browser's local IndexedDB.
const $ = id => document.getElementById(id);
const db = new Promise((resolve,reject) => {
  const req = indexedDB.open("SignToCap-evidence",1);
  req.onupgradeneeded = () => req.result.createObjectStore("tests",{keyPath:"id"});
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});
let session, recording, latest, captureTimer, checkpointTimer, writing = Promise.resolve(), saveQueue = Promise.resolve();
const elapsed = () => performance.now() - session.clock;
async function persist(item) {
  const database = await db;
  await new Promise((resolve,reject) => {
    const tx = database.transaction("tests","readwrite");
    tx.objectStore("tests").put(item);
    tx.oncomplete = resolve;
    tx.onabort = () => reject(tx.error);
  });
}
function message(text) { $("recordStatus").textContent = text; }
function recorder(stream) {
  const mimeType = ["video/mp4","video/webm;codecs=vp8","video/webm"].find(x => MediaRecorder.isTypeSupported(x));
  const rec = new MediaRecorder(stream,mimeType ? {mimeType} : {});
  const chunks = [];
  const done = new Promise((resolve,reject) => {
    rec.ondataavailable = e => { if(e.data.size) chunks.push(e.data); };
    rec.onerror = e => reject(e.error);
    rec.onstop = () => resolve(new Blob(chunks,{type:rec.mimeType}));
  });
  rec.start(500);
  return {rec,done,snapshot:()=>new Blob(chunks,{type:rec.mimeType})};
}
export function beginEvidence(runtime, fps=3, mode="live") {
  session = {id:new Date().toISOString(), clock:performance.now(), events:[],frames:[], recordingStart:null,
    model:"Uni-Sign How2Sign pose-only 1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d",
    config:{fps,mode,beams:1,previewTokens:24,finalTokens:48,policy:"explicit",bufferSeconds:mode==="sentence"?8:10,runtime}};
  checkpointTimer=setInterval(()=>{
    if(!session) return;
    const draft={...session,events:[...session.events],frames:[...session.frames],durationMs:elapsed(),
      reason:"Interrupted test · last local checkpoint"};
    delete draft.clock;
    if(recording) {
      draft.raw=recording.raw.snapshot();draft.overlay=recording.overlay?.snapshot();
    }
    writing=writing.then(()=>persist(draft)).catch(e=>message("Local checkpoint failed: "+e.message));
  },3000);
}
export function evidenceEvent(event) {
  if(session && event) session.events.push({...event,emit_ms:elapsed()});
}
export function frameMetric(metric) { if(session) session.frames.push({ms:elapsed(),...metric}); }
export function startRecording(video, caption) {
  if(!session || recording || session.recordingStart !== null) return;
  if(!window.MediaRecorder) throw Error("This browser does not support MediaRecorder");
  recording = {raw:recorder(video.srcObject)};
  session.recordingStart = elapsed();
  try {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    const draw = () => {
      ctx.drawImage(video,0,0,canvas.width,canvas.height);
      const text = caption();
      if(text) {
        ctx.fillStyle = "#132438dd"; ctx.fillRect(0,canvas.height-54,canvas.width,54);
        ctx.font = "20px sans-serif"; ctx.fillStyle = "#fff"; ctx.fillText(text,12,canvas.height-23,canvas.width-24);
      }
    };
    draw(); recording.overlayStream = canvas.captureStream(15);
    recording.overlay = recorder(recording.overlayStream);
    captureTimer = setInterval(draw,67);
  } catch { message("Recording original video. Overlay unavailable."); }
  $("recordStart").disabled = true; $("recordStop").disabled = false;
  message("Recording to this browser.");
}
export async function stopRecording() {
  if(!recording) return saveQueue;
  const current = recording, item = session;
  recording = null; clearInterval(captureTimer);
  item.recordingEnd = elapsed();
  $("recordStop").disabled = true;
  const pending = [current.raw,current.overlay].filter(Boolean);
  pending.forEach(r => {if(r.rec.state !== "inactive") r.rec.stop();});
  saveQueue = Promise.allSettled(pending.map(r=>r.done)).then(([raw,overlay]) => {
    item.raw=raw.value || current.raw.snapshot(); item.overlay=overlay?.value;
    if(raw.status==="rejected" || overlay?.status==="rejected") item.recordingError="Recorder interrupted";
    current.overlayStream?.getTracks().forEach(t=>t.stop());
    message("Recording finalized locally.");
  });
  return saveQueue;
}
export async function finishEvidence(gpu,reason) {
  if(!session) return;
  clearInterval(checkpointTimer);
  const item=session;
  item.durationMs=elapsed(); item.reason=reason; item.gpu=gpu;
  await stopRecording(); await writing; session=null; delete item.clock;
  latest=item; await persist(item);
  await refresh(item.id); message("Test saved on this device.");
}
export async function updateCost(gpu) {
  if(latest && latest.gpu?.seconds != null && latest.gpu?.began === gpu.began) {
    latest.gpu=gpu; await persist(latest);
  }
}
let url;
async function refresh(selected) {
  const database=await db;
  const req=database.transaction("tests").objectStore("tests").getAllKeys();
  req.onsuccess=()=> {
    $("saved").replaceChildren(...req.result.reverse().map(id=>new Option(id,id)));
    if(selected) $("saved").value=selected;
    $("saved").dispatchEvent(new Event("change"));
  };
}
let selected;
$("playback").ontimeupdate=()=>{
  if(!selected) return;
  const t=$("playback").currentTime*1000+(selected.recordingStart||0);
  const event=selected.events.filter(e=>e.emit_ms<=t && ["partial","final","uncertain"].includes(e.kind)).at(-1);
  $("savedCaption").textContent=event ? event.kind+" · "+(event.text||"No committed caption") : "";
};
$("saved").onchange=async()=>{
  const database=await db;
  if(!$("saved").value) return;
  const req=database.transaction("tests").objectStore("tests").get($("saved").value);
  req.onsuccess=()=>{
    selected=req.result; if(url) URL.revokeObjectURL(url);
    const media=selected.overlay || selected.raw;
    $("playback").hidden=!media;
    if(media) {url=URL.createObjectURL(media); $("playback").src=url;}
    $("export").disabled=false;
    $("evidenceSummary").textContent=selected.events.length+" events · "+(selected.durationMs/1000).toFixed(1)+" s · "+selected.reason;
  };
};
function download(blob,name) {
  const link=document.createElement("a"), address=URL.createObjectURL(blob);
  link.href=address; link.download=name; link.click(); setTimeout(()=>URL.revokeObjectURL(address),20000);
}
$("export").onclick=()=>{
  if(!selected) return;
  const {raw,overlay,...metadata}=selected, name=selected.id.replace(/:/g,"-");
  for(const [suffix,file] of [["original",raw],["captioned",overlay]]) if(file) {
    download(file,name+"-"+suffix+(file.type.includes("mp4")?".mp4":".webm"));
  }
  download(new Blob([JSON.stringify(metadata,null,2)],{type:"application/json"}),name+".json");
};
refresh().catch(e=>message(e.message));
