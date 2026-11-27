// Offline Edge test. The desktop serves real replay and the test supplies a fake GPU transport.
const assert = require("node:assert/strict");
const {spawn} = require("node:child_process");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const {setTimeout: delay} = require("node:timers/promises");

async function until(check, timeout = 12000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const result = await check();
    if (result) return result;
    await delay(75);
  }
  throw Error("Timed out waiting for the local app");
}

async function connect(url) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, {once: true});
    socket.addEventListener("error", reject, {once: true});
  });
  const pending = new Map();
  let id = 0;
  socket.addEventListener("message", ({data}) => {
    const message = JSON.parse(data);
    const reply = pending.get(message.id);
    if (!reply) return;
    pending.delete(message.id);
    if (message.error) reply.reject(Error(message.error.message));
    else reply.resolve(message.result);
  });
  return {
    close: () => socket.close(),
    call(method, params = {}) {
      return new Promise((resolve, reject) => {
        const request = ++id;
        const timeout = setTimeout(() => {
          pending.delete(request);
          reject(Error(`Edge did not respond to ${method}`));
        }, 15000);
        const finish = fn => value => { clearTimeout(timeout); fn(value); };
        pending.set(request, {resolve:finish(resolve), reject:finish(reject)});
        socket.send(JSON.stringify({id:request, method, params}));
      });
    },
  };
}

function mockTransport() {
  const nativeFetch = window.fetch.bind(window);
  const rows = [
    {pod:"fixturel4",gpu:"NVIDIA L4",status:"RUNNING",rate:.69,cap:.70,issue:"",generation:"fixture"},
    {pod:"fixturertx",gpu:"NVIDIA RTX 4090",status:"RUNNING",rate:.89,cap:.90,issue:"",generation:"fixture"},
    {pod:"fixturel40s",gpu:"NVIDIA L40S",status:"RUNNING",rate:1.09,cap:null,issue:"",generation:"fixture"},
    {pod:"fixturestopped",gpu:"NVIDIA L4",status:"EXITED",rate:.69,cap:.70,issue:"",generation:null},
  ].map(row => ({...row, disk:50, storage:.008}));
  const test = window.gpuFixture = {confirm:false, starts:[], posts:[], state:"idle", detail:"", rows};
  window.confirm = () => test.confirm;
  const reply = data => Response.json(data);
  window.fetch = async (url, options) => {
    if (url === "/api/status") {
      const running = ["host_key","ready","watching_delayed_start"].includes(test.state);
      const pod = test.starts.at(-1)?.pod;
      return reply({live:test.state==="ready",fps:3,token:"mock-control",gpu:{
        state:test.state, detail:test.detail, available:!running, busy:running, can_stop:running,
        ready:test.state==="ready", key_configured:true, candidates:test.rows, discovery_detail:test.discoveryDetail,
        pod, gpu:rows.find(row => row.pod===pod)?.gpu, pod_status:running?"RUNNING":"EXITED",
        began:running || test.state==="stopped" ? 1234 : 0, seconds:12, remaining:888, cost:.003,
        approved_total:.898, current_rate:.89, windows_guard:running, remote_guard:test.state==="ready",
        setup_remaining:test.state==="host_key"?588:null,
        start_remaining:test.startRemaining, cancel_requested:test.cancelRequested,
        pin_command:test.state==="host_key"?"fixture host key command":null,
      }});
    }
    const body = options?.body ? JSON.parse(options.body) : {};
    if (options?.method === "POST") test.posts.push(url);
    if (url === "/api/gpu/discover") {
      if (test.discovery) {
        test.rows = test.discovery.rows;
        test.discoveryDetail = test.discovery.detail;
      }
      return reply({});
    }
    if (url === "/api/gpu/start") {
      test.starts.push(body);
      test.state = "host_key";
      test.detail = "Confirm the host key";
      return reply({});
    }
    if (url === "/api/gpu/pin") {
      if (body.trusted !== true || body.output !== "fixture trusted output") throw Error("Bad mock pin");
      test.state = "ready";
      test.detail = "Ready for the 60-second live test";
      return reply({});
    }
    if (url === "/api/gpu/stop") {
      test.state = "shutdown_unverified";
      test.detail = "Fixture shutdown response lost";
      return reply({});
    }
    if (url === "/api/start") return reply({event:null});
    if (url === "/api/translate-video") {
      if (!body.track || body.segment !== 2) throw Error("Wrong prerecorded sentence");
      return reply({event:{kind:"final",text:"Fixture prerecorded",work_ms:300},
        pose:{frames:81,source_fps:24,pose_work_fps:5.86}});
    }
    if (url === "/api/stop") {
      test.state = "stopped";
      test.detail = "Fixture verified stopped";
      return reply({event:null});
    }
    if (url === "/api/frame") return reply({event:{kind:"partial",text:"Fixture partial"}});
    if (url === "/api/commit") return reply({event:{kind:"final",text:"Fixture final"}});
    return nativeFetch(url, options);
  };
}

function mockCamera() {
  navigator.mediaDevices.getUserMedia = async () => {
    const source = document.createElement("canvas");
    source.width = 640;
    source.height = 480;
    const context = source.getContext("2d");
    const stream = source.captureStream(12);
    const [track] = stream.getVideoTracks();
    let frame = 0;
    function paint() {
      if (track.readyState !== "live") return;
      context.fillStyle = frame++ % 2 ? "navy" : "teal";
      context.fillRect(0, 0, source.width, source.height);
      requestAnimationFrame(paint);
    }
    paint();
    return stream;
  };
}

async function run() {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "SignToCap-Edge-"));
  const proc = spawn("python", ["-B", path.join(__dirname, "desktop.py"), "--port", "0"]);
  let edge, cdp;
  try {
    const stdout = await new Promise((resolve, reject) => {
      proc.once("error", reject);
      proc.once("exit", code => reject(Error(`Preview server exited (${code})`)));
      proc.stdout.once("data", resolve);
    });
    const origin = stdout.toString().match(/http:\/\/localhost:\d+/)[0].replace("localhost","127.0.0.1");
    assert.equal((await fetch(origin + "/api/status")).status, 200);
    const edgePath = process.env.SIGNTOCAP_EDGE ||
      path.join(process.env["ProgramFiles(x86)"], "Microsoft/Edge/Application/msedge.exe");
    edge = spawn(edgePath, ["--headless=new","--no-first-run","--disable-background-networking","--no-proxy-server",
      "--disable-component-update","--disable-sync",
      "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1, EXCLUDE localhost",
      "--remote-debugging-port=0","--use-fake-device-for-media-stream","--use-fake-ui-for-media-stream",
      `--user-data-dir=${temp}`,"about:blank"], {stdio:"ignore",windowsHide:true});
    const port = await until(async () => {
      if (edge.exitCode != null) throw Error(`Edge exited (${edge.exitCode})`);
      return fs.readFile(path.join(temp,"DevToolsActivePort"), "utf8")
        .then(content => content.split(/\r?\n/)[0], () => false);
    });
    async function openTab() {
      const target = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`, {method:"PUT"});
      assert.ok(target.ok, "Edge could not open a fresh test tab");
      const tab = await target.json();
      return connect(tab.webSocketDebuggerUrl);
    }
    cdp = await openTab();
    async function evaluate(expression) {
      const result = await cdp.call("Runtime.evaluate", {expression,returnByValue:true,awaitPromise:true,userGesture:true});
      if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
      return result.result.value;
    }
    const wait = expression => until(() => evaluate(expression));
    const click = id => evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
    const field = (id, value) => evaluate(`(() => {
      const el = document.getElementById(${JSON.stringify(id)});
      el.value = ${JSON.stringify(value)}; el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change"));
    })()`);
    const key = (code, ctrl=false) => evaluate(`document.dispatchEvent(new KeyboardEvent("keydown",
      {code:${JSON.stringify(code)},ctrlKey:${ctrl},bubbles:true,cancelable:true}))`);
    assert.equal(await evaluate("1 + 1"), 2);
    try {
      await wait("document.querySelector('#start') && !document.querySelector('#start').disabled");
    } catch (error) {
      console.error(await evaluate("({url:location.href,title:document.title,status:document.getElementById('status')?.textContent})"));
      throw error;
    }
    await key("F8");
    await wait("document.getElementById('partial').textContent !== '—'");
    assert.equal(await evaluate("document.getElementById('mode-label').textContent"),"Recorded replay");
    await key("F8");
    assert.equal(await evaluate("document.getElementById('status').textContent"),"Recorded replay paused.");
    assert.ok(await evaluate("document.getElementById('recorded').videoWidth > 0"));
    await field("seek",2);
    assert.notEqual(await evaluate("document.getElementById('reference').textContent"),"—");
    await evaluate(`(${mockCamera.toString()})()`);
    await field("mode","live");
    await click("preview");
    await wait("document.getElementById('video').videoWidth > 0");
    assert.ok(await evaluate("document.getElementById('start').disabled"));
    await evaluate(`(${mockTransport.toString()})()`);
    await wait("document.getElementById('gpuPod')?.options.length === 4");
    await wait("document.getElementById('gpuQuote').textContent.includes('Enter a positive spending ceiling')");
    assert.ok(await evaluate("document.getElementById('gpuStart').disabled"));
    await field("gpuCeiling",.60);
    await wait("document.getElementById('gpuQuote').textContent.includes('Lowest total is $0.698')");
    await evaluate("gpuFixture.availableRows = gpuFixture.rows; gpuFixture.discovery = {rows:[], detail:'RunPod returned no Pods. Check the key workspace.'}");
    await click("gpuRefresh");
    await wait("document.getElementById('gpuQuote').textContent.includes('Check the key workspace')");
    assert.ok(await evaluate("document.getElementById('gpuStart').disabled"));
    await evaluate("gpuFixture.discovery = {rows:gpuFixture.availableRows, detail:''}");
    await wait("!document.getElementById('gpuRefresh').disabled");
    await click("gpuRefresh");
    await wait("document.getElementById('gpuPod')?.options.length === 4");
    await field("gpuCeiling",.9);
    await field("gpuStorage",.007);
    await wait("document.getElementById('gpuQuote').textContent.includes('Clear the storage override')");
    assert.ok(await evaluate("document.getElementById('gpuStart').disabled"));
    await field("gpuStorage","");
    await wait("document.getElementById('gpuQuote').textContent.includes('fixturel4')");
    await click("gpuStart");
    assert.equal(await evaluate("gpuFixture.starts.length"),0);
    await field("gpuPod","fixturertx");
    await evaluate("gpuFixture.confirm = true");
    await click("gpuStart");
    await wait("!document.getElementById('gpuPin').hidden");
    await evaluate(`Object.assign(gpuFixture,{state:'watching_delayed_start',detail:'RunPod start returned HTTP 400',
      startRemaining:82,cancelRequested:true})`);
    await wait("document.getElementById('gpuStatus').textContent.includes('1m 22s left')");
    assert.match(await evaluate("document.getElementById('gpuStatus').textContent"), /HTTP 400.*Stop requested/);
    assert.ok(await evaluate("document.getElementById('gpuStart').disabled"));
    await evaluate("gpuFixture.state = 'host_key'; gpuFixture.detail = 'Confirm the host key'");
    await wait("!document.getElementById('gpuPin').hidden");
    assert.equal(await evaluate("gpuFixture.starts[0].total"),.898);
    assert.equal(await evaluate("gpuFixture.starts[0].status"),"RUNNING");
    assert.ok(await evaluate("document.getElementById('start').disabled"));
    await field("gpuPinOutput","fixture trusted output");
    await click("gpuTrust");
    await wait("!document.getElementById('start').disabled");
    await field("mode","prerecorded");
    await field("segment",2);
    await click("start");
    await wait("document.getElementById('finals').textContent.includes('Fixture prerecorded')");
    assert.match(await evaluate("document.getElementById('metrics').textContent"), /81 real frames/);
    await field("mode","live");
    await evaluate("document.getElementById('recordEnabled').checked = true");
    await click("start");
    try {
      await wait("document.getElementById('partial').textContent === 'Fixture partial'");
    } catch (error) {
      console.error(await evaluate(`({status:document.getElementById('status').textContent,
        width:document.getElementById('video').videoWidth, posts:gpuFixture.posts})`));
      throw error;
    }
    await key("Enter",true);
    await wait("document.getElementById('finals').textContent.includes('Fixture final')");
    await delay(800);
    await click("stop");
    await wait("document.getElementById('playback')?.readyState >= 2");
    assert.match(await evaluate("document.getElementById('evidenceSummary').textContent"),/Ended by operator/);
    cdp.close();
    cdp = await openTab();
    await wait("document.getElementById('playback')?.readyState >= 2");
    await evaluate(`(${mockTransport.toString()})()`);
    await wait("document.getElementById('gpuPod')?.options.length === 4");
    await field("gpuCeiling",1.1);
    await field("gpuPod","fixturel40s");
    await evaluate("gpuFixture.confirm = true");
    await click("gpuStart");
    await wait("!document.getElementById('gpuPin').hidden");
    await click("emergency");
    await wait("document.getElementById('warning').textContent.includes('STOP UNVERIFIED')");
    console.log("Passed Edge replay, GPU auto and manual selection, preview, host confirmation, mock captions, recording, saved evidence and emergency shutdown UI.");
  } finally {
    cdp?.close();
    edge?.kill();
    proc.kill();
    const real = await fs.realpath(temp);
    if (path.dirname(real) === await fs.realpath(os.tmpdir()) && path.basename(real).startsWith("SignToCap-Edge-")) {
      await fs.rm(real,{recursive:true,force:true,maxRetries:5,retryDelay:150}).catch(() => {});
    }
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
