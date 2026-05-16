const stage = document.querySelector("#stage");
const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const overlayCtx = overlay.getContext("2d", { alpha: true });
const fxCanvas = document.querySelector("#fxCanvas");
const fxCtx = fxCanvas.getContext("2d", { alpha: true });
const orb = document.querySelector("#orb");
const hintToast = document.querySelector("#hintToast");
const startButton = document.querySelector("#startButton");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const modeLabel = document.querySelector("#modeLabel");
const gestureLabel = document.querySelector("#gestureLabel");
const chargeLabel = document.querySelector("#chargeLabel");
const faceLabel = document.querySelector("#faceLabel");

const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const PINCH_MS = 1000;
const FRAME_COUNT = 30;

let handLandmarker;
let faceLandmarker;
let lastVideoTime = -1;
let lastPoint = { x: 0.5, y: 0.5 };
let fistStartedAt = 0;
let fistWasActive = false;
let armed = true;
let hiddenUntil = 0;
let audioContext;
let fallbackCanvas;
let fallbackCtx;
let previousFrame;
let spriteEffects = [];
let lastFrameAt = performance.now();
let powercoreFrames = [];
let powercoreLoadingPromise;
let idleFrameIndex = 0;
let loopStarted = false;
let lastDetectionAt = 0;
let trackedPoint = { x: 0.5, y: 0.5 };
let trackedLabel = "손 찾는 중";
let fistConfidence = 0;
let lastFaceAt = 0;

const lerp = (from, to, amount) => from + (to - from) * amount;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function setStatus(message, ready = false) {
  statusText.textContent = message;
  statusDot.classList.toggle("is-ready", ready);
}

function sizeCanvas() {
  const rect = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  for (const canvas of [overlay, fxCanvas]) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas === overlay ? overlayCtx : fxCtx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function getAudioContext() {
  audioContext ||= new AudioContext();
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }
  return audioContext;
}

function playChargeTick(charge) {
  const context = getAudioContext();
  const osc = context.createOscillator();
  const gain = context.createGain();
  const now = context.currentTime;

  osc.type = "triangle";
  osc.frequency.setValueAtTime(420 + charge * 420, now);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.04, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);
  osc.connect(gain);
  gain.connect(context.destination);
  osc.start();
  osc.stop(now + 0.09);
}

function playExplosionSound() {
  const context = getAudioContext();
  const now = context.currentTime;
  const osc = context.createOscillator();
  const gain = context.createGain();

  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(120, now);
  osc.frequency.exponentialRampToValueAtTime(38, now + 0.55);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.28, now + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.7);
  osc.connect(gain);
  gain.connect(context.destination);
  osc.start();
  osc.stop(now + 0.75);
}

function mirroredPoint(point, rect) {
  return {
    x: (1 - point.x) * rect.width,
    y: point.y * rect.height,
  };
}

function handSpellPoint(landmarks) {
  const knuckleIds = [5, 9, 13, 17];
  const knuckles = knuckleIds.map((id) => landmarks[id]);
  const wrist = landmarks[0];
  const palmBase = landmarks[9];
  const centerPoints = [wrist, palmBase, ...knuckles];
  const center = centerPoints.reduce(
    (sum, point) => {
      sum.x += point.x;
      sum.y += point.y;
      return sum;
    },
    { x: 0, y: 0 },
  );

  const palmCenter = {
    x: center.x / centerPoints.length,
    y: center.y / centerPoints.length,
  };
  const palmSize = Math.max(0.001, distance(wrist, palmBase));
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinchDistance = distance(thumbTip, indexTip);
  const pinchScore = 1 - clamp((pinchDistance - palmSize * 0.28) / (palmSize * 0.5), 0, 1);

  return {
    x: 1 - (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
    pinch: pinchScore > 0.55,
    pinchScore,
  };
}

function drawHand(landmarks) {
  const rect = overlay.getBoundingClientRect();
  const fingers = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
  ];

  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = "rgba(82, 226, 255, 0.68)";
  overlayCtx.fillStyle = "rgba(255, 255, 255, 0.86)";

  for (const finger of fingers) {
    overlayCtx.beginPath();
    finger.forEach((index, order) => {
      const point = mirroredPoint(landmarks[index], rect);
      if (order === 0) overlayCtx.moveTo(point.x, point.y);
      else overlayCtx.lineTo(point.x, point.y);
    });
    overlayCtx.stroke();
  }

  for (const rawPoint of landmarks) {
    const point = mirroredPoint(rawPoint, rect);
    overlayCtx.beginPath();
    overlayCtx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

function drawWizardFilter(face) {
  const rect = overlay.getBoundingClientRect();
  const top = mirroredPoint(face[10], rect);
  const chin = mirroredPoint(face[152], rect);
  const leftCheek = mirroredPoint(face[454], rect);
  const rightCheek = mirroredPoint(face[234], rect);
  const nose = mirroredPoint(face[1], rect);
  const faceWidth = Math.max(80, Math.abs(leftCheek.x - rightCheek.x));
  const faceHeight = Math.max(100, Math.abs(chin.y - top.y));
  const centerX = (leftCheek.x + rightCheek.x) / 2;

  overlayCtx.save();
  overlayCtx.lineJoin = "round";
  overlayCtx.lineCap = "round";

  const hatBaseY = top.y - faceHeight * 0.1;
  const hatTipY = top.y - faceHeight * 1.25;
  const hatHalfWidth = faceWidth * 0.72;
  const brimY = top.y - faceHeight * 0.08;

  overlayCtx.shadowColor = "rgba(0, 0, 0, 0.45)";
  overlayCtx.shadowBlur = 16;
  overlayCtx.shadowOffsetY = 8;

  const hatGradient = overlayCtx.createLinearGradient(centerX, hatTipY, centerX, hatBaseY);
  hatGradient.addColorStop(0, "rgba(78, 70, 190, 0.98)");
  hatGradient.addColorStop(0.52, "rgba(38, 32, 118, 0.96)");
  hatGradient.addColorStop(1, "rgba(18, 14, 60, 0.96)");

  overlayCtx.fillStyle = hatGradient;
  overlayCtx.strokeStyle = "rgba(218, 224, 255, 0.85)";
  overlayCtx.lineWidth = Math.max(2, faceWidth * 0.018);
  overlayCtx.beginPath();
  overlayCtx.moveTo(centerX - faceWidth * 0.08, hatTipY);
  overlayCtx.bezierCurveTo(
    centerX - hatHalfWidth * 0.92,
    top.y - faceHeight * 0.72,
    centerX - hatHalfWidth * 0.42,
    top.y - faceHeight * 0.34,
    centerX - hatHalfWidth * 0.5,
    brimY,
  );
  overlayCtx.quadraticCurveTo(centerX, brimY + faceHeight * 0.08, centerX + hatHalfWidth * 0.5, brimY);
  overlayCtx.bezierCurveTo(
    centerX + hatHalfWidth * 0.2,
    top.y - faceHeight * 0.5,
    centerX + faceWidth * 0.12,
    top.y - faceHeight * 0.82,
    centerX - faceWidth * 0.08,
    hatTipY,
  );
  overlayCtx.fill();
  overlayCtx.stroke();

  const brimGradient = overlayCtx.createLinearGradient(centerX, brimY - 20, centerX, brimY + 28);
  brimGradient.addColorStop(0, "rgba(82, 72, 190, 0.95)");
  brimGradient.addColorStop(1, "rgba(15, 12, 45, 0.96)");
  overlayCtx.fillStyle = brimGradient;
  overlayCtx.beginPath();
  overlayCtx.ellipse(centerX, brimY, hatHalfWidth, faceHeight * 0.16, -0.04, 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.stroke();

  overlayCtx.shadowBlur = 0;
  overlayCtx.fillStyle = "rgba(255, 209, 102, 0.96)";
  for (const [sx, sy, size] of [
    [centerX - faceWidth * 0.24, top.y - faceHeight * 0.58, 6],
    [centerX + faceWidth * 0.17, top.y - faceHeight * 0.42, 5],
    [centerX - faceWidth * 0.02, top.y - faceHeight * 0.78, 5],
  ]) {
    drawStar(sx, sy, size);
  }

  const beardGradient = overlayCtx.createLinearGradient(centerX, nose.y, centerX, chin.y + faceHeight * 0.7);
  beardGradient.addColorStop(0, "rgba(255, 255, 255, 0.9)");
  beardGradient.addColorStop(0.55, "rgba(218, 228, 245, 0.82)");
  beardGradient.addColorStop(1, "rgba(255, 255, 255, 0.12)");
  overlayCtx.strokeStyle = beardGradient;
  overlayCtx.lineWidth = Math.max(4, faceWidth * 0.035);

  for (let i = -5; i <= 5; i += 1) {
    const spread = i / 5;
    overlayCtx.beginPath();
    overlayCtx.moveTo(nose.x + spread * faceWidth * 0.08, nose.y + faceHeight * 0.12);
    overlayCtx.bezierCurveTo(
      nose.x + spread * faceWidth * 0.22,
      nose.y + faceHeight * 0.34,
      centerX + spread * faceWidth * 0.24,
      chin.y + faceHeight * 0.28,
      centerX + spread * faceWidth * 0.08,
      chin.y + faceHeight * (0.62 + Math.abs(spread) * 0.15),
    );
    overlayCtx.stroke();
  }

  overlayCtx.restore();
}

function drawStar(x, y, radius) {
  overlayCtx.beginPath();
  for (let i = 0; i < 10; i += 1) {
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    const length = i % 2 === 0 ? radius : radius * 0.42;
    const px = x + Math.cos(angle) * length;
    const py = y + Math.sin(angle) * length;
    if (i === 0) overlayCtx.moveTo(px, py);
    else overlayCtx.lineTo(px, py);
  }
  overlayCtx.closePath();
  overlayCtx.fill();
}

async function loadDetectors() {
  const vision = await import(
    /* @vite-ignore */
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18"
  );
  const filesetResolver = await vision.FilesetResolver.forVisionTasks(WASM_PATH);

  const handOptions = {
    baseOptions: {
      modelAssetPath: HAND_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  };
  const faceOptions = {
    baseOptions: {
      modelAssetPath: FACE_MODEL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  };

  try {
    handLandmarker = await vision.HandLandmarker.createFromOptions(filesetResolver, handOptions);
    faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, faceOptions);
  } catch {
    handLandmarker = await vision.HandLandmarker.createFromOptions(filesetResolver, {
      ...handOptions,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" },
    });
    faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
      ...faceOptions,
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
    });
  }

  modeLabel.textContent = "마법 감지";
}

function ensureFallbackCanvas() {
  if (fallbackCanvas) return;
  fallbackCanvas = document.createElement("canvas");
  fallbackCanvas.width = 96;
  fallbackCanvas.height = 72;
  fallbackCtx = fallbackCanvas.getContext("2d", { willReadFrequently: true });
}

function fallbackMotion() {
  ensureFallbackCanvas();
  fallbackCtx.drawImage(video, 0, 0, fallbackCanvas.width, fallbackCanvas.height);
  const frame = fallbackCtx.getImageData(0, 0, fallbackCanvas.width, fallbackCanvas.height);

  if (!previousFrame) {
    previousFrame = frame;
    return { point: lastPoint, pinch: false, label: "움직임" };
  }

  let total = 0;
  let weightedX = 0;
  let weightedY = 0;
  for (let i = 0; i < frame.data.length; i += 4) {
    const diff =
      Math.abs(frame.data[i] - previousFrame.data[i]) +
      Math.abs(frame.data[i + 1] - previousFrame.data[i + 1]) +
      Math.abs(frame.data[i + 2] - previousFrame.data[i + 2]);
    if (diff > 58) {
      const pixel = i / 4;
      const x = pixel % fallbackCanvas.width;
      const y = Math.floor(pixel / fallbackCanvas.width);
      total += diff;
      weightedX += x * diff;
      weightedY += y * diff;
    }
  }

  previousFrame = frame;
  if (total < 12000) {
    return { point: lastPoint, pinch: false, label: "손 찾는 중" };
  }

  return {
    point: {
      x: 1 - weightedX / total / fallbackCanvas.width,
      y: weightedY / total / fallbackCanvas.height,
    },
    pinch: true,
    label: "움직임 충전",
  };
}

function frameUrl(index) {
  return `/effects/powercore-optimized/powercore-${String(index).padStart(2, "0")}.png`;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

function loadPowercoreFrames() {
  powercoreLoadingPromise ||= Promise.all(
    Array.from({ length: FRAME_COUNT }, (_, index) => loadImage(frameUrl(index + 1))),
  )
    .then((frames) => {
      powercoreFrames = frames;
      return frames;
    })
    .catch((error) => {
      console.warn(error);
      powercoreFrames = [];
      return [];
    });

  return powercoreLoadingPromise;
}

function spawnSpriteEffect(point) {
  const rect = fxCanvas.getBoundingClientRect();
  spriteEffects = [
    {
      x: point.x * rect.width,
      y: point.y * rect.height,
      age: 0,
      fps: 30,
      life: 1,
      scale: 1.12,
    },
  ];
}

function explode(point) {
  armed = false;
  hiddenUntil = performance.now() + 760;
  spawnSpriteEffect(point);
  playExplosionSound();
  chargeLabel.textContent = "100%";
}

function setOrb(point, pinch) {
  const now = performance.now();
  lastPoint = {
    x: lerp(lastPoint.x, point.x, 0.36),
    y: lerp(lastPoint.y, point.y, 0.36),
  };

  const hidden = now < hiddenUntil;
  orb.style.setProperty("--x", `${lastPoint.x * 100}%`);
  orb.style.setProperty("--y", `${lastPoint.y * 100}%`);
  orb.classList.toggle("is-pinching", pinch && !hidden);
  orb.classList.toggle("is-charging", pinch && armed && !hidden);
  orb.classList.toggle("is-exploding", hidden);

  if (pinch && !fistWasActive) {
    fistStartedAt = now;
    playChargeTick(0);
  }

  if (pinch && armed && !hidden) {
    const charge = clamp((now - fistStartedAt) / PINCH_MS, 0, 1);
    orb.style.setProperty("--charge", `${Math.round(charge * 360)}deg`);
    chargeLabel.textContent = `${Math.round(charge * 100)}%`;

    if (charge >= 1) {
      explode(lastPoint);
    }
  }

  if (!pinch) {
    armed = true;
    fistStartedAt = 0;
    orb.style.setProperty("--charge", "0deg");
    chargeLabel.textContent = "0%";
  }

  fistWasActive = pinch;
}

function drawEffects(dt) {
  const rect = fxCanvas.getBoundingClientRect();
  fxCtx.clearRect(0, 0, rect.width, rect.height);

  if (powercoreFrames.length && performance.now() >= hiddenUntil) {
    idleFrameIndex = (idleFrameIndex + dt * 18) % powercoreFrames.length;
    const frame = powercoreFrames[Math.floor(idleFrameIndex)];
    const idleWidth = Math.min(rect.width, rect.height) * 0.44;
    const idleHeight = idleWidth * 0.68;
    const x = lastPoint.x * rect.width;
    const y = lastPoint.y * rect.height;

    fxCtx.save();
    fxCtx.globalAlpha = 0.9;
    fxCtx.filter = "saturate(1.18) brightness(1.04)";
    fxCtx.drawImage(frame, x - idleWidth / 2, y - idleHeight / 2, idleWidth, idleHeight);
    fxCtx.restore();
    orb.classList.add("has-vfx-core");
  } else {
    orb.classList.remove("has-vfx-core");
  }

  spriteEffects = spriteEffects.filter((effect) => {
    effect.age += dt;
    if (effect.age >= effect.life || !powercoreFrames.length) return false;

    const frameIndex = Math.min(powercoreFrames.length - 1, Math.floor(effect.age * effect.fps));
    const frame = powercoreFrames[frameIndex];
    const progress = effect.age / effect.life;
    const alpha = 1 - Math.max(0, progress - 0.8) / 0.2;
    const burst = 1 + Math.sin(Math.min(progress, 1) * Math.PI) * 0.35;
    const width = Math.min(rect.width, rect.height) * effect.scale * burst;
    const height = width * 0.68;

    fxCtx.save();
    fxCtx.globalAlpha = alpha;
    fxCtx.filter = `saturate(1.45) brightness(${1.25 + (1 - progress) * 0.75}) contrast(1.12)`;
    fxCtx.drawImage(frame, effect.x - width / 2, effect.y - height / 2, width, height);
    fxCtx.restore();
    return true;
  });
}

function detectFrame() {
  const now = performance.now();
  const dt = clamp((now - lastFrameAt) / 1000, 0.001, 0.033);
  lastFrameAt = now;
  const rect = overlay.getBoundingClientRect();
  overlayCtx.clearRect(0, 0, rect.width, rect.height);

  let point = lastPoint;
  let pinch = false;
  let label = "손 찾는 중";
  let detected = false;

  if (video.readyState >= 2) {
    if (handLandmarker && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const handResult = handLandmarker.detectForVideo(video, now);
      const faceResult = faceLandmarker?.detectForVideo(video, now);

      if (faceResult?.faceLandmarks?.length) {
        drawWizardFilter(faceResult.faceLandmarks[0]);
        lastFaceAt = now;
      }

      if (handResult.landmarks?.length) {
        const landmarks = handResult.landmarks[0];
        drawHand(landmarks);
        const spell = handSpellPoint(landmarks);
        trackedPoint = spell;
        trackedLabel = spell.pinchScore > 0.32 ? "핀치 준비" : "손";
        lastDetectionAt = now;
        detected = true;
        fistConfidence = lerp(
          fistConfidence,
          spell.pinch ? Math.max(0.72, spell.pinchScore) : spell.pinchScore * 0.82,
          0.42,
        );
      }
    } else if (!handLandmarker) {
      const motion = fallbackMotion();
      trackedPoint = motion.point;
      trackedLabel = motion.label;
      lastDetectionAt = now;
      detected = true;
      fistConfidence = clamp(fistConfidence + (motion.pinch ? 0.24 : -0.2), 0, 1);
    }
  }

  if (!detected) {
    fistConfidence = clamp(fistConfidence - 0.04, 0, 1);
  }

  if (now - lastDetectionAt < 350) {
    point = trackedPoint;
    pinch = fistWasActive ? fistConfidence > 0.36 : fistConfidence > 0.5;
    label = pinch ? "핀치" : trackedLabel;
  }

  gestureLabel.textContent = label;
  faceLabel.textContent = now - lastFaceAt < 500 ? "마법사" : "대기";
  setOrb(point, pinch);
  drawEffects(dt);
  requestAnimationFrame(runFrame);
}

function runFrame() {
  try {
    detectFrame();
  } catch (error) {
    console.error(error);
    setStatus("인식 중 잠깐 오류가 났어요. 다시 이어서 볼게요.", true);
    requestAnimationFrame(runFrame);
  }
}

function startLoop() {
  if (loopStarted) return;
  loopStarted = true;
  lastFrameAt = performance.now();
  runFrame();
}

function showHint() {
  hintToast.classList.remove("is-visible");
  void hintToast.offsetWidth;
  hintToast.classList.add("is-visible");
}

async function startCamera() {
  startButton.disabled = true;
  setStatus("카메라 권한을 기다리는 중이에요.");
  getAudioContext();
  loadPowercoreFrames();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    sizeCanvas();
    setStatus("카메라가 켜졌어요. 엄지와 검지를 1초 동안 잡으면 마법이 폭발해요.", true);
    showHint();

    try {
      setStatus("손과 얼굴 인식 모델을 불러오는 중이에요.", true);
      await loadDetectors();
      setStatus("마법사 필터가 켜졌어요. 엄지와 검지를 잡으면 마법이 충전돼요.", true);
    } catch (error) {
      modeLabel.textContent = "움직임 감지";
      setStatus("모델을 못 불러와서 움직임 충전 모드로 전환했어요.", true);
      console.warn(error);
    }

    startLoop();
  } catch (error) {
    setStatus("카메라 권한이 필요해요. 브라우저 권한을 확인해 주세요.");
    startButton.disabled = false;
    console.error(error);
  }
}

window.addEventListener("resize", sizeCanvas);
loadPowercoreFrames();
sizeCanvas();
startLoop();
startButton.addEventListener("click", startCamera);
