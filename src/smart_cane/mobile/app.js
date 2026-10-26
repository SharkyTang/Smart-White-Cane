"use strict";

const $ = (id) => document.getElementById(id);
const ui = {
  video: $("video"), overlay: $("overlay"), capture: $("captureCanvas"),
  placeholder: $("cameraPlaceholder"), connection: $("connectionText"),
  count: $("objectCount"), cameraLabel: $("cameraLabel"), alertCard: $("alertCard"),
  alertTitle: $("alertTitle"), alertDetail: $("alertDetail"),
  start: $("startButton"), stop: $("stopButton"), switcher: $("switchButton"), sound: $("soundButton"),
  directions: { left: $("directionLeft"), center: $("directionCenter"), right: $("directionRight") },
};

const state = { running:false, generation:0, stream:null, facing:"environment", timer:null,
  request:null, sessionReady:false, failures:0, audio:null, lastPayload:null, frameWidth:0, frameHeight:0 };
const FRAME_MS = 125;
const LABELS = { person:"行人", bicycle:"自行车", car:"汽车", motorcycle:"摩托车", bus:"公交车",
  truck:"卡车", chair:"椅子", bench:"长椅", dog:"狗", cat:"猫", suitcase:"行李箱",
  backpack:"背包", umbrella:"雨伞", bottle:"瓶子", "dining table":"餐桌", couch:"沙发",
  "potted plant":"盆栽", "traffic light":"红绿灯", "stop sign":"停止标志" };
const DIRECTION = { left:"左侧", center:"前方", right:"右侧" };
const LEVEL = { low:"低风险", medium:"中风险", high:"高风险" };
const RANK = { none:-1, low:0, medium:1, high:2 };

function setConnection(text, kind="idle") { ui.connection.textContent=text; ui.connection.dataset.state=kind; }
function setButtons(active) { ui.start.disabled=active; ui.stop.disabled=!active; ui.switcher.disabled=!active; }
function clearLocalView() {
  state.sessionReady=false; state.failures=0; state.lastPayload=null; state.frameWidth=0; state.frameHeight=0;
  ui.count.textContent="0 个目标"; ui.alertCard.dataset.level="none"; ui.alertTitle.textContent="等待检测";
  ui.alertDetail.textContent="识别结果将在这里显示";
  Object.values(ui.directions).forEach((node)=>{ node.dataset.level="none"; node.querySelector("strong").textContent="—"; });
  const ctx=ui.overlay.getContext("2d"); ctx.clearRect(0,0,ui.overlay.width,ui.overlay.height);
}
function abortWork() { if(state.timer) clearTimeout(state.timer); state.timer=null; if(state.request) state.request.abort(); state.request=null; }
function stopTracks() { if(state.stream) state.stream.getTracks().forEach((track)=>track.stop()); state.stream=null; ui.video.pause(); ui.video.srcObject=null; }

async function unlockAudio() {
  const AudioContext=window.AudioContext||window.webkitAudioContext;
  if(AudioContext && !state.audio) state.audio=new AudioContext();
  if(state.audio) { await state.audio.resume(); const osc=state.audio.createOscillator(); const gain=state.audio.createGain(); gain.gain.value=0.0001; osc.connect(gain).connect(state.audio.destination); osc.start(); osc.stop(state.audio.currentTime+.02); }
  if("speechSynthesis" in window) { window.speechSynthesis.cancel(); const u=new SpeechSynthesisUtterance("。"); u.lang="zh-CN"; u.volume=0; window.speechSynthesis.speak(u); }
}
function beep(level="medium") {
  if(!state.audio || state.audio.state!=="running") return;
  const times=level==="high"?[0,.18]:[0];
  times.forEach((offset)=>{ const o=state.audio.createOscillator(), g=state.audio.createGain(), t=state.audio.currentTime+offset;
    o.frequency.value=level==="high"?920:680; g.gain.setValueAtTime(.001,t); g.gain.exponentialRampToValueAtTime(.24,t+.02); g.gain.exponentialRampToValueAtTime(.001,t+.14); o.connect(g).connect(state.audio.destination); o.start(t); o.stop(t+.16); });
}
function chineseLabel(label="") { const key=String(label).trim().toLowerCase(); return /[\u3400-\u9fff]/.test(label)?label:(LABELS[key]||"障碍物"); }
function speak(assessment) {
  if(!("speechSynthesis" in window)) return;
  const level=assessment?.level==="high"?"危险":"注意";
  const text=`${level}，${DIRECTION[assessment?.direction]||"附近"}有${chineseLabel(assessment?.label)}`;
  window.speechSynthesis.cancel(); const u=new SpeechSynthesisUtterance(text); u.lang="zh-CN"; u.rate=.95; window.speechSynthesis.speak(u);
}

async function timedFetch(url, options, timeout=4500) {
  const controller=new AbortController(); state.request=controller; let timedOut=false;
  const timer=setTimeout(()=>{ timedOut=true; controller.abort(); },timeout);
  try { return await fetch(url,{...options,signal:controller.signal}); }
  catch(error) { if(timedOut) throw new Error("timeout"); throw error; }
  finally { clearTimeout(timer); if(state.request===controller) state.request=null; }
}
async function resetSession(gen) {
  const response=await timedFetch("/api/session/reset",{method:"POST",headers:{Accept:"application/json"}});
  if(gen!==state.generation) throw new Error("stale"); if(!response.ok) throw new Error("reset"); state.sessionReady=true;
}
async function openCamera(gen) {
  if(!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("secure");
  const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:state.facing},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:15,max:30}}});
  if(gen!==state.generation) { stream.getTracks().forEach((track)=>track.stop()); throw new Error("stale"); }
  state.stream=stream; ui.video.srcObject=stream; ui.video.classList.toggle("mirrored",state.facing==="user"); await ui.video.play();
  ui.placeholder.hidden=true; ui.cameraLabel.textContent=state.facing==="environment"?"后置摄像头":"前置摄像头";
}
function canvasBlob() {
  const vw=ui.video.videoWidth, vh=ui.video.videoHeight; if(!vw||!vh) return Promise.reject(new Error("camera"));
  const scale=Math.min(1,640/Math.max(vw,vh)); const w=Math.round(vw*scale), h=Math.round(vh*scale);
  if(ui.capture.width!==w||ui.capture.height!==h){ui.capture.width=w;ui.capture.height=h;} ui.capture.getContext("2d").drawImage(ui.video,0,0,w,h);
  state.frameWidth=w; state.frameHeight=h;
  return new Promise((resolve,reject)=>ui.capture.toBlob((blob)=>blob?resolve(blob):reject(new Error("jpeg")),"image/jpeg",.68));
}
async function postFrame(blob) {
  const response=await timedFetch("/api/frame",{method:"POST",headers:{"Content-Type":"image/jpeg",Accept:"application/json"},body:blob});
  if(!response.ok) throw new Error("frame"); return response.json();
}
function schedule(delay=0,gen=state.generation) { if(!state.running||document.hidden||gen!==state.generation)return; if(state.timer)clearTimeout(state.timer); state.timer=setTimeout(()=>frameTick(gen),delay); }
async function frameTick(gen) {
  state.timer=null; if(!state.running||document.hidden||gen!==state.generation)return; const started=performance.now();
  try {
    if(!state.sessionReady) await resetSession(gen); const blob=await canvasBlob(); const payload=await postFrame(blob);
    if(gen!==state.generation)return; state.failures=0; render(payload); setConnection("检测中","running");
    schedule(Math.max(0,FRAME_MS-(performance.now()-started)),gen);
  } catch(error) {
    if(gen!==state.generation||!state.running)return; state.sessionReady=false; state.failures+=1; setConnection("正在重新连接…","waiting");
    schedule(Math.min(4000,500*(2**Math.min(3,state.failures-1))),gen);
  }
}

function boxOf(item) { if(Array.isArray(item?.bbox)) return item.bbox; const b=item?.bbox||{}; return [b.x_min,b.y_min,b.x_max,b.y_max]; }
function bestByDirection(assessments) {
  const best={left:null,center:null,right:null}; assessments.forEach((a)=>{if(best[a.direction]===undefined)return;if(!best[a.direction]||RANK[a.level]>RANK[best[a.direction].level])best[a.direction]=a;}); return best;
}
function render(payload) {
  const detections=Array.isArray(payload?.detections)?payload.detections:[];
  const assessments=Array.isArray(payload?.assessments)?payload.assessments:[]; state.lastPayload=payload;
  ui.count.textContent=assessments.length?`${assessments.length} 个稳定目标`:`${detections.length} 个识别目标`; const best=bestByDirection(assessments);
  Object.entries(best).forEach(([direction,a])=>{const node=ui.directions[direction];node.dataset.level=a?.level||"none";node.querySelector("strong").textContent=a?LEVEL[a.level]||"已检测":"—";});
  const urgent=[...assessments].sort((a,b)=>(RANK[b.level]||0)-(RANK[a.level]||0))[0];
  if(urgent){ui.alertCard.dataset.level=urgent.level||"low";ui.alertTitle.textContent=`${DIRECTION[urgent.direction]||"附近"} · ${LEVEL[urgent.level]||"已检测"}`;ui.alertDetail.textContent=chineseLabel(urgent.label);} else {ui.alertCard.dataset.level="none";ui.alertTitle.textContent="视野内暂无稳定风险";ui.alertDetail.textContent="请缓慢移动手机观察周围";}
  drawBoxes(assessments.length?assessments:detections,payload);
  if(payload?.alert?.should_emit===true){const a=payload.alert.assessment||payload.alert;beep(a.level);speak(a);}
}
function drawBoxes(items,payload) {
  const rect=ui.overlay.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2); ui.overlay.width=Math.round(rect.width*dpr);ui.overlay.height=Math.round(rect.height*dpr);
  const ctx=ui.overlay.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,rect.width,rect.height);
  const fw=payload?.frame?.width||payload?.frame_width||items[0]?.frame_width||state.frameWidth; const fh=payload?.frame?.height||payload?.frame_height||items[0]?.frame_height||state.frameHeight;if(!fw||!fh)return;
  const scale=Math.max(rect.width/fw,rect.height/fh), ox=(rect.width-fw*scale)/2, oy=(rect.height-fh*scale)/2;
  items.forEach((item)=>{const b=boxOf(item);if(b.some((v)=>!Number.isFinite(Number(v))))return;const level=item.level||"low", color=level==="high"?"#ff4545":level==="medium"?"#ffd43b":"#22d3ee";
    let x=Number(b[0])*scale+ox,y=Number(b[1])*scale+oy,w=(Number(b[2])-Number(b[0]))*scale,h=(Number(b[3])-Number(b[1]))*scale;if(state.facing==="user")x=rect.width-x-w;
    ctx.strokeStyle=color;ctx.lineWidth=4;ctx.strokeRect(x,y,w,h);const label=`${chineseLabel(item.label)} ${LEVEL[level]||""}`;ctx.font="bold 15px -apple-system";const tw=ctx.measureText(label).width+14;ctx.fillStyle=color;ctx.fillRect(x,Math.max(0,y-27),tw,27);ctx.fillStyle="#06111f";ctx.fillText(label,x+7,Math.max(19,y-8)); });
}

async function start() {
  if(state.running)return; const gen=++state.generation; state.running=true; clearLocalView();setButtons(true);setConnection("正在打开摄像头…","waiting");
  try { await unlockAudio(); await openCamera(gen); setConnection("正在初始化检测…","waiting"); await resetSession(gen); if(gen!==state.generation)return;schedule(0,gen); }
  catch(error){if(gen!==state.generation)return;state.running=false;abortWork();stopTracks();setButtons(false);ui.placeholder.hidden=false;setConnection(error.message==="secure"?"请使用 HTTPS 打开页面":error.name==="NotAllowedError"?"请在 Safari 网站设置中允许相机":"摄像头启动失败","error");}
}
function stop() { ++state.generation;state.running=false;abortWork();stopTracks();window.speechSynthesis?.cancel();clearLocalView();setButtons(false);ui.placeholder.hidden=false;setConnection("已停止"); }
async function switchCamera() { if(!state.running)return;const gen=++state.generation;abortWork();stopTracks();clearLocalView();state.facing=state.facing==="environment"?"user":"environment";setConnection("正在切换摄像头…","waiting");try{await unlockAudio();await openCamera(gen);await resetSession(gen);schedule(0,gen);}catch(error){if(gen===state.generation){state.running=false;stopTracks();setButtons(false);ui.placeholder.hidden=false;setConnection("切换摄像头失败","error");}} }
async function testSound(){await unlockAudio();beep("medium");if("speechSynthesis" in window){window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance("声音测试正常");u.lang="zh-CN";window.speechSynthesis.speak(u);}}

ui.start.addEventListener("click",start);ui.stop.addEventListener("click",stop);ui.switcher.addEventListener("click",switchCamera);ui.sound.addEventListener("click",testSound);
async function resumeFromBackground(gen) {
  try {
    const track=state.stream?.getVideoTracks()[0];
    if(!track||track.readyState!=="live"){stopTracks();await openCamera(gen);}
    if(gen!==state.generation||!state.running)return;state.sessionReady=false;await resetSession(gen);setConnection("正在恢复检测…","waiting");schedule(0,gen);
  } catch(error) { if(gen===state.generation&&state.running)setConnection("恢复失败，请停止后重新开始","error"); }
}
document.addEventListener("visibilitychange",()=>{if(!state.running)return;if(document.hidden){abortWork();state.sessionReady=false;setConnection("后台暂停，返回后恢复","waiting");}else{void resumeFromBackground(state.generation);}});
window.addEventListener("resize",()=>{if(state.lastPayload)render(state.lastPayload);});window.addEventListener("pagehide",()=>{state.running=false;abortWork();stopTracks();});
