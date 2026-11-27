const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "ui/app.js"), "utf8");
const capture = source.slice(source.indexOf("function captureFrame("), source.indexOf("async function deliverFrames("));
const control = source.slice(source.indexOf("async function control("), source.indexOf('ui.mode.addEventListener("change"'));
const fixture = `
const ui = {video:{videoWidth:640,videoHeight:480}};
const canvas = {toBlob:()=>{}}, context = {drawImage:()=>{}};
const sentence = () => true, live = () => false, camera = () => true;
const say = () => {}, display = () => {}, render = () => {}, evidenceEvent = () => {};
let pending=null, active=true, sentenceGap=false, sentenceClosed=false, sentenceBatchTimer, epoch=7;
let frameQueue=[], batchSize=3, commitCutoff, busy=false, sentenceFrames=0;
let captureBase=null, lastSlot=-1, lastPresentedFrame=-1, captureFps=24, sentenceLimit=192;
let maxQueuedFrames=6, bufferDrops=0, frameCallback, timer;
const order = [];
async function deliverFrames(batch) {order.push(batch.map(x=>x.slot));}
async function api(action) {assert.equal(action,"commit");order.push("commit");return {event:{}};}
async function endTest() {order.push("stop");active=false;}
`;

function run(test) {
  return vm.runInNewContext(fixture + capture + control + test,
    {assert, setTimeout, clearTimeout, performance});
}

(async () => {
  await run(`
    (async () => {
      let complete;
      const first = {slot:0};
      first.encoding = new Promise(resolve => {complete=()=>{first.jpeg="encoded";resolve();};});
      frameQueue.push(first, {slot:1,jpeg:"ready"}, {slot:2,jpeg:"ready"}, {slot:3,jpeg:"ready"});
      sentenceFrames=12;
      sendNextFrame();
      assert.equal(order.length,0);
      const translation = control("commit");
      assert.equal(order.length,0);
      complete();
      await translation;
      assert.equal(JSON.stringify(order), '[[0,1,2],[3],"commit","stop"]');
    })()
  `);
  run(`
    for (let slot=0; slot<13; slot++) {
      captureFrame(200 + slot*1000/24 + .01, slot+1);
      captureFrame(200 + slot*1000/24 + .02, slot+1);
    }
    assert.equal(frameQueue.length,12);
    assert.equal(sentenceFrames,12);
    assert.equal(bufferDrops,1);
    assert.equal(sentenceGap,true);
    assert.equal(order.length,0);
  `);
  console.log("Capture ordering, duplicate rejection, bounded backlog and end-of-sentence draining passed");
})().catch(error=>{console.error(error); process.exitCode=1;});
