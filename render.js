// =======================
// Config & Globals
// =======================
const GAMMA = 0.9;           // 할인율 (0<gamma<1)
const A = 4;                  // numActions
const SOFTMAX_TAU = 2500;     // 로컬 정책 softmax 온도
const SOURCE_FLOW_RATE = 1.0; // 시작 노드 지속 유입(시각화용)

const POLICY_EPS = 1e-3;      // 정책 변화 감지 임계값
const FADE_EPS = 1e-3;        // γ^t 페이드아웃 임계값 (기여량에만 사용)
const VISUAL_FLOOR = 0.1;    // 입자 시각 알파 하한 (이동 중 선명도 유지)

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let flowPaused = true;

let nodes = [];
let edges = [];
let particles = [];
let nodeRadius;
let animationFrameId = null;

let mouseX = 0, mouseY = 0;
let isControllingPolicy = false;
let controlledNode = null;

let policyVersion = 0;        // 현재 정책 버전
let currentDSA = null;        // 최신 d(s,a)
let piCached = null;          // 최신 π


let showQ = true;   // 엣지에 Q(s,a) 표시
let showV = true;   // 노드 아래에 V(s) 표시
window.addEventListener('keydown', (e) => {
  if (e.key === 'q' || e.key === 'Q') showQ = !showQ;
  if (e.key === 'v' || e.key === 'V') showV = !showV;
});



// 크로스페이드 상태 (이전 정책→새 정책 부드럽게 전환)
let xf = {
  active: false,
  t: 0,         // 진행 프레임
  dur: 45,      // 전환 프레임 수 (30~90 권장)
  oldVersion: 0,
  lastDSA: null // 직전 정책의 d(s,a)
};
const smooth = x => (x<=0?0 : x>=1?1 : x*x*(3-2*x)); // smoothstep
function wNew() { return xf.active ? smooth(xf.t/xf.dur) : 1; }
function wOld() { return xf.active ? (1 - smooth(xf.t/xf.dur)) : 0; }

// 프레임 독립 속도: 화면 크기에 비례
function getParticleSpeed() {
  return Math.max(2, Math.min(canvas.width, canvas.height) * 0.006); // px/frame
}

// =======================
// 그래프(시각화) 정의
// =======================
const nodePositions = [
  { id: 0,  x: 0.10, y: 0.50 }, { id: 1,  x: 0.25, y: 0.25 },
  { id: 2,  x: 0.25, y: 0.75 }, { id: 3,  x: 0.40, y: 0.40 },
  { id: 4,  x: 0.40, y: 0.60 }, { id: 5,  x: 0.45, y: 0.10 },
  { id: 6,  x: 0.60, y: 0.25 }, { id: 7,  x: 0.55, y: 0.50 },
  { id: 8,  x: 0.45, y: 0.90 }, { id: 9,  x: 0.60, y: 0.75 },
  { id:10,  x: 0.75, y: 0.15 }, { id:11,  x: 0.75, y: 0.45 },
  { id:12,  x: 0.70, y: 0.60 }, { id:13,  x: 0.80, y: 0.85 },
  { id:14,  x: 0.90, y: 0.50 },
];

const undirectedEdges = [
  [0,1],[0,2],[1,2],[1,3],[1,5],[2,4],[2,8],
  [3,4],[3,6],[4,7],[4,9],[5,6],[6,10],[6,11],
  [7,3],[7,11],[7,12],[8,9],[9,12],[9,13],[10,11],
  [11,14],[12,14],[13,14],[12,9],[8,13],[14,11],[14,13]
];

// 상태 보상(노드 숫자)
const rewardPresets = {
  1: [0,1,3,4,0,1,3,1,0,3,0,2,1,3,5],
  2: [0,2,4,3,0,5,4,1,2,2,0,3,2,4,1],
  3: [0,3,4,1,0,0,2,1,5,3,0,1,1,2,4],
};
let rewardPresetId = 1;
const stateRewards = rewardPresets[rewardPresetId].slice();

// 보상 적용: stateRewards, 노드 라벨, mdpForGraph.reward 갱신
function applyRewardPreset(presetKey) {
  document.getElementById(`btnR${rewardPresetId}`).style.backgroundColor = '#444';
  document.getElementById(`btnR${presetKey}`).style.backgroundColor = '#660';

  rewardPresetId = presetKey;
  const arr = rewardPresets[presetKey];
  if (!arr || arr.length !== nodes.length) {
    console.warn('Invalid reward preset', arr, arr.length, nodes.length);
    return;
  }

  // 1) 전역 stateRewards 갱신
  for (let i = 0; i < arr.length; i++) stateRewards[i] = arr[i];

  // 2) 노드 라벨 갱신
  for (const n of nodes) n.reward = stateRewards[n.id] || 0;

  // 3) MDP의 R 갱신 (self-transition 페널티 유지)
  const S = mdpForGraph.numStates;
  for (let s = 0; s < S; s++) {
    for (let a = 0; a < mdpForGraph.numActions; a++) {
      const isSelf = mdpForGraph.transition[s][a][s] === 1;
      mdpForGraph.reward[s][a] = isSelf ? -1000 : (stateRewards[s] || 0);
    }
  }

  // 4) 시각적 EMA는 그대로 두면 부드럽게 반영됨
  // 필요하면 아래 줄로 J 반영을 더 빠르게 하고 싶을 때 dSA_vis만 유지:
  // pi_vis = null; // <- 화살표 크기도 부드럽게 유지하려면 주석 유지

  // 버전만 올려서 내부 상태 변화 신호 (선택)
  policyVersion++;
}
document.getElementById('btnR1')?.addEventListener('click', () => applyRewardPreset(1));
document.getElementById('btnR2')?.addEventListener('click', () => applyRewardPreset(2));
document.getElementById('btnR3')?.addEventListener('click', () => applyRewardPreset(3));


// =======================
// 시각화용 클래스
// =======================
// 보상 스케일 (init이나 보상 preset 변경 시 갱신)
let REW_MIN = 0, REW_MAX = 5;
const clamp01 = x => x <= 0 ? 0 : x >= 1 ? 1 : x;
const lerp = (a,b,t) => a + (b-a)*t;

class Node {
  constructor(id, x, y) {
    this.id = id; this.x = x; this.y = y;
    this.type = (id === 0) ? 'start' : 'normal';
    this.inFlow = 0;
    this.outFlowBuffer = 0;
    this.actions = [];             // Edge들이 actionIndex에 맞게 저장
    this.policyDistribution = {};  // a -> prob
    this.reward = stateRewards[id] || 0;
  }
  draw() {
    const r = this.reward ?? 0;         // 보상값
    const maxR = 5;                     // 최대 보상
    const t = Math.max(0, Math.min(1, r / maxR)); // 0~1 정규화
  
    // start: 회색 (#888888), end: 밝은 파랑 (#4da6ff)
    const startColor = { r: 136, g: 136, b: 136 };
    const endColor   = { r: 30,  g: 50, b: 255 };
  
    const rr = Math.round(startColor.r + (endColor.r - startColor.r) * t);
    const gg = Math.round(startColor.g + (endColor.g - startColor.g) * t);
    const bb = Math.round(startColor.b + (endColor.b - startColor.b) * t);
  
    const fillColor = `rgb(${rr},${gg},${bb})`;
  
    const radius = nodeRadius * (1 + Math.min(this.outFlowBuffer / 15, 0.5));
  
    // 노드 채우기
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.arc(this.x, this.y, radius, 0, Math.PI * 2);
    ctx.fill();
  
    // 선택 노드 강조
    if (this === controlledNode) {
      ctx.lineWidth = Math.max(2.0, nodeRadius * 0.16);
      ctx.strokeStyle = 'rgba(0,200,200,0.9)';
      ctx.stroke();
    }
  
    // 라벨(검은 스트로크 + 흰색 채움)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `${nodeRadius * 0.9}px sans-serif`;
    const text = (this.type === 'start') ? 'S' : r.toString();
  
    ctx.lineWidth = Math.max(2, nodeRadius * 0.12);
    ctx.strokeStyle = 'rgba(0,0,0,0.65)';
    ctx.strokeText(text, this.x, this.y);
  
    ctx.fillStyle = 'white';
    ctx.fillText(text, this.x, this.y);
  }
}

class Edge {
  constructor(sourceNode, targetNode, actionIndex = null) {
    this.source = sourceNode;
    this.target = targetNode;
    this.actionIndex = actionIndex; // 시각화/MDP 동일 인덱스
    if (actionIndex !== null) sourceNode.actions[actionIndex] = this;
  }
  drawBase() {
    ctx.strokeStyle = `rgba(160,160,180,0.15)`;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(this.source.x, this.source.y);
    ctx.lineTo(this.target.x, this.target.y);
    ctx.stroke();
  }

  // ▼ d^π(s,a)에 비례하는 굵은 선
  drawFlowLine(color, lineWidth) {
    const dx = this.target.x - this.source.x;
    const dy = this.target.y - this.source.y;
    const angle = Math.atan2(dy, dx);

    // 노드 원 경계를 살짝 비켜서 선을 그리기
    const fromX = this.source.x + nodeRadius * Math.cos(angle);
    const fromY = this.source.y + nodeRadius * Math.sin(angle);
    const toX   = this.target.x - nodeRadius * Math.cos(angle);
    const toY   = this.target.y - nodeRadius * Math.sin(angle);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
    ctx.restore();
  }

  drawArrowhead(color, lineWidth) {
    const dx = this.target.x - this.source.x;
    const dy = this.target.y - this.source.y;
    const angle = Math.atan2(dy, dx);
    const headlen = 3 + lineWidth * 1.2;
    const fromX = this.source.x + nodeRadius * Math.cos(angle);
    const fromY = this.source.y + nodeRadius * Math.sin(angle);

    const rectWidth = lineWidth * 2;
    const rectLength = headlen * 1.5;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.fillStyle = color;
    ctx.translate(fromX, fromY); ctx.rotate(angle);
    ctx.fillRect(0, -rectWidth / 2, rectLength, rectWidth);
    ctx.strokeRect(0, -rectWidth / 2, rectLength, rectWidth);
    ctx.restore();

    const tipX = fromX + (rectWidth / 2 + rectLength) * Math.cos(angle);
    const tipY = fromY + (rectWidth / 2 + rectLength) * Math.sin(angle);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - headlen * Math.cos(angle - Math.PI / 4), tipY - headlen * Math.sin(angle - Math.PI / 4));
    ctx.lineTo(tipX - headlen * Math.cos(angle + Math.PI / 4), tipY - headlen * Math.sin(angle + Math.PI / 4));
    ctx.closePath(); ctx.stroke(); ctx.fill();
    ctx.restore();
  }
}

class Particle {
  constructor(sourceNode, targetNode, flowValue) {
    this.version = policyVersion;
    this.x = sourceNode.x; this.y = sourceNode.y;
    this.size = 1 + Math.random() * 1.5;
    const spray = nodeRadius * 2.0;
    this.targetX = targetNode.x + (Math.random() - 0.5) * spray;
    this.targetY = targetNode.y + (Math.random() - 0.5) * spray;
    this.targetNode = targetNode;
    this.flowValue = flowValue;
  }
  update() {
    const dx = this.targetX - this.x, dy = this.targetY - this.y;
    const dist = Math.hypot(dx, dy);
    const depositR = Math.max(12, nodeRadius * 0.9);

    if (dist < depositR) {
      const vw = (xf.active
        ? ((this.version === xf.oldVersion) ? wOld()
           : (this.version === policyVersion) ? wNew() : 0)
        : 1);
      // ★ 엣지 1회 전이당 γ 한 번만 적용
      this.targetNode.inFlow += this.flowValue * GAMMA * vw;
      return false;
    }
    const step = Math.min(getParticleSpeed(), dist);
    const jitterX = (Math.random() - 0.5) * 0.3;
    const jitterY = (Math.random() - 0.5) * 0.3;
    this.x += (dx / dist) * step + jitterX;
    this.y += (dy / dist) * step + jitterY;
    return true;
  }
  draw() {
    // 이동 중 밝기는 유지: 버전 가중치만 반영, 하한으로 또렷함 보장
    const vw = (xf.active
                ? ((this.version === xf.oldVersion) ? wOld()
                   : (this.version === policyVersion) ? wNew() : 0)
                : 1);
    const alpha = 0.85 * Math.max(VISUAL_FLOOR, vw);
    if (alpha <= 0.01) return;

    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

// =======================
// 그래프 빌드 → actionIndex 결정
// =======================
function buildGraphForViz(canvasWidth, canvasHeight) {
  const ns = nodePositions.map(p => new Node(p.id, p.x * canvasWidth, p.y * canvasHeight));
  ns[0].type = 'start';

  // 인접 리스트(무방향)
  const adj = Array.from({ length: ns.length }, () => new Set());
  for (const [u, v] of undirectedEdges) { adj[u].add(v); adj[v].add(u); }

  // 각 상태에서 이웃을 각도 순으로 정렬 → actionIndex 0..A-1
  const es = [];
  for (const n of ns) {
    const nbrs = [...adj[n.id]];
    const withAngle = nbrs.map(tid => {
      const t = ns[tid];
      return { tid, angle: Math.atan2(t.y - n.y, t.x - n.x) };
    }).sort((a, b) => a.angle - b.angle);

    const real = Math.min(A, withAngle.length);
    for (let a = 0; a < real; a++) {
      const t = ns[withAngle[a].tid];
      es.push(new Edge(n, t, a));
    }

    // 초기 정책: real actions 균등, 나머지 0
    n.policyDistribution = {};
    for (let a = 0; a < real; a++) n.policyDistribution[a] = 1 / real;
    for (let a = real; a < A; a++) n.policyDistribution[a] = 0;
  }

  return { nodes: ns, edges: es };
}

// =======================
// 그래프 → MDP (네 MDP 함수 사용)
// =======================
function buildMDPFromGraph(nodes, gamma) {
  const S = nodes.length;

  // T[s][a][s'] (deterministic; 없는 액션은 self-loop)
  const T = math.zeros([S, A, S]);
  for (const n of nodes) {
    for (let a = 0; a < A; a++) {
      const e = n.actions[a];
      const sp = (e && e.target) ? e.target.id : n.id;
      T[n.id][a][sp] = 1;
    }
  }

  // R[s][a] : 상태 보상을 모든 액션에 동일하게 복사
  const R = math.zeros([S, A]);
  for (let s = 0; s < S; s++) {
    for (let a = 0; a < A; a++) {
      R[s][a] = stateRewards[s] || 0;
      if (T[s][a][s] === 1) {
        R[s][a] = -1000;
      }
    }
  }

  // p0: 시작 상태 1-hot
  const p0 = math.zeros([S]); p0[0] = 1;

  return new MDP(S, A, T, R, gamma, p0);
}

// =======================
// 정책/분포 유틸
// =======================
function policyFromViz(nodes) {
  const S = nodes.length;
  const pi = math.zeros([S, A]);
  for (const n of nodes) {
    for (let a = 0; a < A; a++) {
      pi[n.id][a] = n.policyDistribution[a] || 0;
    }
  }
  return pi;
}

function computeExactD_and_Return(mdp, pi) {
  // 상태 분포 d(s): 네가 구현한 함수 호출
  const dState = compute_state_stationary_distribution(mdp, pi); // [S]
  // 상태-행동 분포 d(s,a) = d(s) * pi(a|s)
  const dSA = math.zeros([mdp.numStates, mdp.numActions]);
  for (let s = 0; s < mdp.numStates; s++) {
    for (let a = 0; a < mdp.numActions; a++) dSA[s][a] = dState[s] * pi[s][a];
  }
  // 기대 할인 보상 J = Σ_{s,a} d(s,a) R(s,a)
  let J = 0;
  for (let s = 0; s < mdp.numStates; s++) {
    for (let a = 0; a < mdp.numActions; a++) J += dSA[s][a] * mdp.reward[s][a];
  }
  return { dState, dSA, J };
}

// 거리 기반 로컬 정책 업데이트
// ===== 각도 기반 + 거리-의존 temperature =====
const UNIFORM_RADIUS_FACTOR = 4.0; // nodeRadius * 이 값까지는 거의 균등
const EPS_NORM = 1e-8;

function updateLocalPolicy() {
  const clamp01 = x => x <= 0 ? 0 : x >= 1 ? 1 : x;
  const smooth  = x => x*x*(3 - 2*x); // smoothstep
    
  if (!controlledNode) return;
  const node = controlledNode;
  const mx = mouseX, my = mouseY;

  // 유효 액션 & 타깃 수집
  const validIdx = [];
  const targets  = [];
  for (let a = 0; a < A; a++) {
    const e = node.actions[a];
    if (!e) continue;
    validIdx.push(a);
    targets.push(e.target);
  }
  if (validIdx.length === 0) return;

  // 현재 노드→마우스 벡터
  const vx = mx - node.x, vy = my - node.y;
  const vnorm = Math.hypot(vx, vy);

  // 거리→temperature: 가까울수록 tau=크게, 멀수록 작게
  const R = nodeRadius * UNIFORM_RADIUS_FACTOR;
  const t = smooth(clamp01(vnorm / R));            // 0(가깝다)→1(멀다)

  // 각도 기반 점수: score = cos(angle) = (v·u)/(|v||u|)
  // vnorm이 아주 작으면 (노드 근처) 균등으로 처리
  let newPi = {};
  if (vnorm < nodeRadius) {
    const uni = 1 / validIdx.length;
    for (const a of validIdx) newPi[a] = uni;
  } else {
    const inv_v = 1 / (vnorm + EPS_NORM);
    const scores = targets.map(tg => {
      const ux = tg.x - node.x, uy = tg.y - node.y;
      const unorm = Math.hypot(ux, uy) + EPS_NORM;
      const cos = (vx * ux + vy * uy) * (inv_v / unorm); // [-1,1]
      return cos; // cos가 클수록 (각도 작을수록) 점수↑
    });

    // temperature softmax
    const m = Math.max(...scores);
    const exps = scores.map(s => Math.exp((s - m) / Math.exp(-vnorm / 50)));
    const Z = exps.reduce((a, b) => a + b, 0) || 1;

    for (let i = 0; i < validIdx.length; i++) newPi[validIdx[i]] = exps[i] / Z;
  }

  // 없는 액션은 0
  for (let a = 0; a < A; a++) if (!(a in newPi)) newPi[a] = 0;

  // 변화 감지 & 크로스페이드
  let changed = false;
  for (let a = 0; a < A; a++) {
    const oldv = node.policyDistribution[a] ?? 0;
    if (Math.abs(oldv - newPi[a]) > POLICY_EPS) { changed = true; break; }
  }
  node.policyDistribution = newPi;

  if (changed) {
    xf.oldVersion = policyVersion;
    xf.t = 0; xf.active = true;
    xf.lastDSA = currentDSA ? currentDSA.map(r => r.slice()) : null;
    policyVersion++;
  }
}

// nodes ←π— matrix 로 반영
function applyPolicyToNodes(pi) {
  const S = nodes.length;
  for (let s = 0; s < S; s++) {
    const n = nodes[s];
    const dist = {};
    for (let a = 0; a < A; a++) {
      // 실제 엣지가 없으면 0, 있으면 pi값 반영
      dist[a] = (n.actions[a] ? (pi[s][a] ?? 0) : 0);
    }
    n.policyDistribution = dist;
  }
}

// 현재 nodes의 π를 균등으로 설정
function setUniformPolicy() {
  // 크로스페이드 준비
  xf.oldVersion = policyVersion;
  xf.t = 0; xf.active = true;
  xf.lastDSA = currentDSA ? currentDSA.map(r => r.slice()) : null;

  // 각 state에서 유효 액션 수만큼 균등
  const S = nodes.length;
  const pi = math.zeros([S, A]);
  for (const n of nodes) {
    const valid = [];
    for (let a = 0; a < A; a++) if (n.actions[a]) valid.push(a);
    const p = valid.length ? 1 / valid.length : 0;
    for (let a = 0; a < A; a++) pi[n.id][a] = valid.includes(a) ? p : 0;
  }
  applyPolicyToNodes(pi);

  policyVersion++;
}

function setRandomPolicy() {
  if (!mdpForGraph) return;
  const pi = math.zeros([mdpForGraph.numStates, mdpForGraph.numActions]);
  for (let s = 0; s < mdpForGraph.numStates; s++) {
    // 하나의 action을 선택한후 그 action의 확률 1로. 나머지는 0. 
    const valid = [];
    for (let a = 0; a < A; a++) if (nodes[s].actions[a]) valid.push(a);
    const a = valid[Math.floor(Math.random() * valid.length)];
    pi[s][a] = 1;
  }
  applyPolicyToNodes(pi);
  policyVersion++;
}

function setGreedyPolicy() {
  if (!mdpForGraph) return;

  // (선택) 기존 크로스페이드 흐름과 맞추려면 유지
  xf.oldVersion = policyVersion;
  xf.t = 0; xf.active = true;
  xf.lastDSA = currentDSA ? currentDSA.map(r => r.slice()) : null;

  // 현재 π에서 Q^π 계산
  const piNow = policyFromViz(nodes);
  const [vNow, qNow] = policy_evaluation_mdp(mdpForGraph, piNow);

  // argmax_a Q^π(s,a) (유효 액션만, self-loop 제외)
  const S = mdpForGraph.numStates, A = mdpForGraph.numActions;
  const piG = math.zeros([S, A]);

  for (let s = 0; s < S; s++) {
    let bestA = null, bestVal = -Infinity;
    for (let a = 0; a < A; a++) {
      const e = nodes[s].actions[a];
      if (!e) continue;                       // 없는 액션 제외
      if (e.target && e.target.id === s) continue; // self-loop 제외
      const qsa = qNow[s][a];
      if (qsa > bestVal) { bestVal = qsa; bestA = a; }
    }
    // 모든 유효 액션이 self-loop였거나 없으면: 있는 액션 중 하나 fallback
    if (bestA === null) {
      for (let a = 0; a < A; a++) if (nodes[s].actions[a]) { bestA = a; break; }
    }
    if (bestA !== null) piG[s][bestA] = 1;
  }

  // 화면 반영
  applyPolicyToNodes(piG);
  policyVersion++;
}

// DP(정책 반복)로 optimal policy 구해 반영 (결정적 π*)
function setOptimalPolicy() {
  if (!mdpForGraph) return;

  // 크로스페이드 준비
  xf.oldVersion = policyVersion;
  xf.t = 0; xf.active = true;
  xf.lastDSA = currentDSA ? currentDSA.map(r => r.slice()) : null;

  // 1) DP로 최적 정책 계산
  const [piStar] = solve_mdp_dp(mdpForGraph); // [S,A], 보통 one-hot
  console.log(piStar);

  // 2) self-transition 금지 후 재정규화
  const S = nodes.length;
  for (let s = 0; s < S; s++) {
    // 유효 액션: e가 존재하고 e.target.id !== s
    const valid = [];
    for (let a = 0; a < A; a++) {
      const e = nodes[s].actions[a];
      if (e && e.target && e.target.id !== s) valid.push(a);
    }

    // 유효 액션이 없으면(고립 노드) 기존 분포 유지
    let Z = 0;
    if (valid.length > 0) {
      // self/없는 액션 0으로, 유효 액션만 집계
      for (let a = 0; a < A; a++) {
        const e = nodes[s].actions[a];
        if (!e || e.target.id === s) {
          piStar[s][a] = 0;
        }
      }
      // 재정규화
      for (const a of valid) Z += piStar[s][a];
      if (Z <= 0) {
        // DP가 self에 몰아줬다면 유효 액션에 균등 배분
        const p = 1 / valid.length;
        for (let a = 0; a < A; a++) piStar[s][a] = 0;
        for (const a of valid) piStar[s][a] = p;
      } else {
        for (const a of valid) piStar[s][a] /= Z;
      }
    } else {
      // 아무 이웃도 없으면 그대로 둠 (self-loop만 존재)
      // 필요하면 여기서도 균등(모두 0) 유지
    }
  }

  // 3) 화면의 노드 정책에 반영
  applyPolicyToNodes(piStar);

  policyVersion++;
}

// 버튼 클릭 핸들러
const btnUniform = document.getElementById('btnUniform');
const btnOptimal = document.getElementById('btnOptimal');
const btnRandom = document.getElementById('btnRandom');
const btnGreedy = document.getElementById('btnGreedy');
if (btnUniform) btnUniform.addEventListener('click', setUniformPolicy);
if (btnOptimal) btnOptimal.addEventListener('click', setOptimalPolicy);
if (btnRandom) btnRandom.addEventListener('click', setRandomPolicy);
if (btnGreedy) btnGreedy.addEventListener('click', setGreedyPolicy);


// =======================
// 메인 루프
// =======================

// --- Score FX state ---
let lastJ = null;
const scoreFx = { active:false, t:0, dur:550, delta:0, sparks:[] };

// 라벨 이펙트 트리거 (증가면 초록, 감소면 빨강 느낌)
function triggerScoreFx(delta, cx, y) {
  scoreFx.active = true;
  scoreFx.t = 0;
  scoreFx.delta = delta;

  // 작은 스파클들 생성
  const N = 28;
  scoreFx.sparks = Array.from({length:N}, () => {
    const ang = Math.random()*Math.PI*2;
    const spd = 1.2 + Math.random()*1.8;
    return {
      x: cx + (Math.random()-0.5)*30,
      y: y + 12 + (Math.random()-0.5)*8,
      vx: Math.cos(ang)*spd,
      vy: Math.sin(ang)*spd - 0.6,
      a: 1,  // alpha
      r: 1 + Math.random()*1.5
    };
  });
}

function drawScoreLabel(text) {
  const clamp01 = x => x<0?0 : x>1?1 : x;
  const easeOutCubic = x => 1 - Math.pow(1-x, 3);

  const cx = canvas.width/2, ty = 20;

  // 진행도 0..1
  let k = 0;
  if (scoreFx.active) {
    k = 1 - easeOutCubic(clamp01(scoreFx.t / scoreFx.dur)); // 1→0
  }

  // 팝/글로우 강도 (변화량 클수록 살짝 더 강하게)
  const mag = 0.5;  //Math.min(1, Math.abs(scoreFx.delta) * 1);
  const scale = 1 + 0.16 * k * mag;

  // 색/그림자 (증가: 초록, 감소: 빨강)
  const glow = scoreFx.delta >= 0 ? 'rgba(80,255,120,0.9)'
                                  : 'rgba(255,80,80,0.9)';

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.font = '40px sans-serif';

  // 글로우
  if (scoreFx.active) {
    ctx.shadowColor = glow;
    ctx.shadowBlur = 20 + 40*k*mag;
  }

  // 스케일 팝
  ctx.translate(cx, ty);
  ctx.scale(scale, scale);

  // 본문 텍스트
  ctx.fillStyle = 'rgba(255,255,0,0.95)'; // 기본 노란색
  ctx.fillText(text, 0, 0);
  ctx.restore();

  // 스파클 업데이트/그리기
  if (scoreFx.active) {
    const col = scoreFx.delta >= 0 ? 'rgba(80,255,120,' : 'rgba(255,80,80,';
    for (const s of scoreFx.sparks) {
      s.x += s.vx; s.y += s.vy; s.vy += 0.02; // 중력 약간
      s.a *= 0.95;
      if (s.a < 0.03) continue;
      ctx.fillStyle = `${col}${(0.8*s.a).toFixed(3)})`;
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI*2); ctx.fill();
    }
    scoreFx.t += (typeof lastTs==='number' ? 16 : 16); // dt를 쓰고 있으면 그 값 사용
    if (scoreFx.t >= scoreFx.dur) scoreFx.active = false;
  }
}

function roundRectPath(x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y, x+w,y+h, r);
  ctx.arcTo(x+w,y+h, x,y+h, r);
  ctx.arcTo(x,y+h, x,y, r);
  ctx.arcTo(x,y, x+w,y, r);
  ctx.closePath();
}

// 시작 노드 근방(화살표 시작 쪽)에 Q(s,a) 라벨 표시
function drawQOnEdges(Q){
  if (!showQ || !Q) return;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `${Math.max(11, nodeRadius*0.4)}px monospace`;

  edges.forEach(e => {
    const s = e.source.id, a = e.actionIndex;
    const val = Q[s]?.[a];
    if (val == null) return;

    // 방향/법선
    const sx = e.source.x, sy = e.source.y;
    const tx = e.target.x, ty = e.target.y;
    const dx = tx - sx, dy = ty - sy;
    const L  = Math.hypot(dx, dy) || 1;
    const ux = dx / L,  uy = dy / L;   // 단위 방향벡터
    const nx = -uy,     ny = ux;       // 법선

    // 소스 원 경계에서 조금 나간 지점 + 법선으로 살짝 띄우기
    const along   = nodeRadius + Math.max(10, nodeRadius * 1); // 화살표 시작쪽
    const normal  = Math.max(8, nodeRadius*0.5);

    // 화면 중심에서 바깥쪽으로 밀어내기(겹침 감소용)
    const cx = canvas.width/2, cy = canvas.height/2;
    const sign = ((sx-cx)*nx + (sy-cy)*ny) >= 0 ? 1 : -1;

    const px = sx + ux*along + nx*normal*sign;
    const py = sy + uy*along + ny*normal*sign;

    const text = val.toFixed(2);
    const pad  = 4;
    const tw   = ctx.measureText(text).width;
    const th   = Math.max(14, nodeRadius*0.7);

    // 배경 버블(반투명 어두운색)
    // max Q action이면 파란색으로.
    const maxQ = Math.max(...Q[s]);
    const maxQAction = Q[s].indexOf(maxQ);
    const isMaxQAction = a === maxQAction;
    const color = isMaxQAction ? 'rgba(150,150,0,0.55)' : 'rgba(0,0,0,0.55)';
    ctx.fillStyle = color;
    roundRectPath(px - (tw/2 + pad), py - th/2, tw + pad*2, th, 6);
    ctx.fill();

    // 텍스트
    ctx.fillStyle = 'white';
    ctx.fillText(text, px, py + 0.5);
  });

  ctx.restore();
}


// 노드 아래에 V(s) 표시
function drawVOnNodes(V){
  if (!showV || !V) return;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `${Math.max(11, nodeRadius*0.55)}px monospace`;

  nodes.forEach(n => {
    const val = V[n.id];
    if (val == null) return;

    const text = val.toFixed(2);
    const pad  = 3;
    const tw   = ctx.measureText(text).width;
    const th   = Math.max(14, nodeRadius*0.7);
    const x    = n.x;
    const y    = n.y + nodeRadius + th*0.8;  // 노드 아래쪽에 배치

    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    // roundRectPath(x - (tw/2 + pad), y - th/2, tw + pad*2, th, 6);
    ctx.fill();

    ctx.fillStyle = 'yellow';
    ctx.fillText(text, x, y + 0.5);
  });

  ctx.restore();
}


function animate() {
  animationFrameId = requestAnimationFrame(animate);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (isControllingPolicy) updateLocalPolicy();

  // 1) 정확한 d^π와 J(π) 계산
  const pi = policyFromViz(nodes);
  piCached = pi;
  const res = computeExactD_and_Return(mdpForGraph, pi);
  currentDSA = res.dSA;

  // ★ 현재 정책의 V, Q 계산 (policy_evaluation_mdp 사용)
  const [v_now, q_now] = policy_evaluation_mdp(mdpForGraph, pi);

  // 블렌드된 d(s,a): d_blend = w_old*d_old + w_new*d_new
  let dSA = currentDSA;
  if (xf.active && xf.lastDSA) {
    dSA = currentDSA.map((row, s) =>
      row.map((v, a) => wOld()*xf.lastDSA[s][a] + wNew()*v)
    );
  }

  // 기대 할인 보상도 블렌드된 분포 기준으로 표시(시각 일관)
  let J = 0;
  for (let s = 0; s < mdpForGraph.numStates; s++)
    for (let a = 0; a < mdpForGraph.numActions; a++)
      J += dSA[s][a] * mdpForGraph.reward[s][a];

  // --- J 변화 감지 후 트리거 ---
  const J_now = J;                                // 방금 계산한 값
  if (lastJ == null) lastJ = J_now;
  const dJ = J_now - lastJ;
  if (Math.abs(dJ) > 1e-2) {                      // 임계값(튜닝 가능)
    const cx = canvas.width / 2, ty = 20;         // 라벨 위치
    triggerScoreFx(dJ, cx, ty);
  }
  lastJ = J_now;

  // 2) 플로우(입자): 정책에 따라 방출, 도착 시 기여량에 γ^t 반영
  nodes.forEach(node => {
    node.outFlowBuffer += node.inFlow; // 이번 프레임 도착분 누적
    node.inFlow = 0;
  
    // ★ 한 스텝당 γ만큼만 내보내기
    const discounted = node.outFlowBuffer * GAMMA;
    let totalOut = discounted + (node.type === 'start' ? SOURCE_FLOW_RATE : 0);
  
    if (totalOut > 1e-2) {
      const numParticles = Math.ceil(totalOut * 2);
      const flowPer = totalOut / numParticles;
      for (let i = 0; i < numParticles; i++) {
        let r = Math.random(), cum = 0;
        for (let a = 0; a < A; a++) {
          const p = node.policyDistribution[a] || 0;
          cum += p;
          if (r < cum) {
            const e = node.actions[a];
            if (e) particles.push(new Particle(node, e.target, flowPer));
            break;
          }
        }
      }
    }
    // 이번에 방출한 만큼은 제거 (백로그는 없앤다)
    node.outFlowBuffer = 0;
  }); 

  particles = particles.filter(p => p.update());

  // 3) 배경 엣지
  edges.forEach(e => e.drawBase());

  // 4) Edge 렌더링: 선 굵기 = d^π(s,a), 화살표 크기 = π(a|s)
  edges.forEach(e => {
    const dsa  = dSA[e.source.id][e.actionIndex];                             // d^π(s,a)
    // (a) d^π 기반 굵은 라인
    if (dsa > 1e-4) {
      const flowLW   = Math.min(Math.log(1 + dsa) * 300, 50);                 // 굵기 스케일은 취향대로
      const flowAlph = Math.min(0.65, 0.18 + dsa * 4); // 투명도(선택)
      e.drawFlowLine(`rgba(255,255,255,${flowAlph})`, flowLW);
    }

  });
  edges.forEach(e => {
    const prob = nodes[e.source.id].policyDistribution[e.actionIndex] || 0;  // π(a|s)
    if (prob > 1e-4) {
      const arrowLW = 5 + prob * 10;                 // 화살표 "크기" 조절 값
      e.drawArrowhead(`rgba(255,85,85,0.9)`, arrowLW);
    }
  });

  // 5) 노드/입자
  if (!flowPaused) {
    particles.forEach(p => p.draw());
  }
  nodes.forEach(n => n.draw());

  // drawVOnNodes(v_now);
  drawQOnEdges(q_now);

  // 6) 상단 점수
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.font = '40px sans-serif';
  ctx.fillStyle = 'rgba(255,255,0,0.9)';
  drawScoreLabel(`R${rewardPresetId}: (Normalized) Expected Discounted Return: ${J.toFixed(2)}`);


  // 7) 크로스페이드 진행/종료
  if (xf.active) {
    xf.t++;
    if (xf.t >= xf.dur) {
      xf.active = false;
      xf.lastDSA = null;
    }
  }
}

// =======================
// 초기화 & 이벤트
// =======================
function findClosestNode(x, y) {
  let best = null, bestd = Infinity;
  for (const n of nodes) {
    const d = (n.x - x) ** 2 + (n.y - y) ** 2;
    if (d < bestd) { bestd = d; best = n; }
  }
  bestd = Math.sqrt(bestd);
  if (bestd > nodeRadius * 2) return null;
  return best;
}

function buildGraphForViz(canvasWidth, canvasHeight) {
  // (이미 위에 정의되어 있지만, 스코프 분리 필요하면 유지)
  const ns = nodePositions.map(p => new Node(p.id, p.x * canvasWidth, p.y * canvasHeight));
  ns[0].type = 'start';
  const adj = Array.from({ length: ns.length }, () => new Set());
  for (const [u, v] of undirectedEdges) { adj[u].add(v); adj[v].add(u); }
  const es = [];
  for (const n of ns) {
    const nbrs = [...adj[n.id]];
    const withAngle = nbrs.map(tid => {
      const t = ns[tid];
      return { tid, angle: Math.atan2(t.y - n.y, t.x - n.x) };
    }).sort((a, b) => a.angle - b.angle);
    const real = Math.min(A, withAngle.length);
    for (let a = 0; a < real; a++) {
      const t = ns[withAngle[a].tid];
      es.push(new Edge(n, t, a));
    }
    n.policyDistribution = {};
    for (let a = 0; a < real; a++) n.policyDistribution[a] = 1 / real;
    for (let a = real; a < A; a++) n.policyDistribution[a] = 0;
  }
  return { nodes: ns, edges: es };
}

function policyFromViz(nodes) {
  const S = nodes.length;
  const pi = math.zeros([S, A]);
  for (const n of nodes) for (let a = 0; a < A; a++) pi[n.id][a] = n.policyDistribution[a] || 0;
  return pi;
}

function setupCanvas() {
  // 1) CSS 크기 확정 (전체 화면 쓰는 예)
  canvas.style.width  = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;

  // 2) 현재 CSS 크기 측정
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  // 3) 백버퍼(물리 픽셀) 설정
  canvas.width  = Math.round(rect.width  * dpr);
  canvas.height = Math.round(rect.height * dpr);

  // 4) 컨텍스트 스케일 (중복 스케일 방지 초기화 필수)
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);

  return { w: rect.width, h: rect.height, dpr };
}

let mdpForGraph = null;
function init() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  const { w, h } = setupCanvas();   // ← CSS px 기준
  nodes = []; edges = []; particles = [];
  nodeRadius = Math.min(w, h) * 0.045;
  const built = buildGraphForViz(w, h);
  nodes = built.nodes; edges = built.edges;
  mdpForGraph = buildMDPFromGraph(nodes, GAMMA);
  currentDSA = null; piCached = null; xf.active = false; xf.lastDSA = null;
  animate();
}
window.addEventListener('resize', init);

// 마우스 이벤트
function updateMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  mouseX = e.clientX - rect.left;
  mouseY = e.clientY - rect.top;
}
canvas.addEventListener('mousedown', (e) => {
  updateMousePos(e);
  controlledNode = findClosestNode(mouseX, mouseY);
  isControllingPolicy = true;
});
canvas.addEventListener('mousemove', (e) => { if (isControllingPolicy) updateMousePos(e); });
canvas.addEventListener('mouseup',   () => {
  isControllingPolicy = false; controlledNode = null;
  // 드래그 종료 시 강제 리셋은 하지 않음(자연스런 크로스페이드 유지)
});
canvas.addEventListener('mouseleave',() => {
  isControllingPolicy = false; controlledNode = null;
});

window.addEventListener('resize', init);

// 시작
init();



function updateFlowButton() {
  const b = document.getElementById('btnFlow');
  if (!b) return;
  b.textContent = flowPaused ? 'Start flow' : 'Stop flow';
}

function toggleFlow() {
  flowPaused = !flowPaused;
  particles = [];
  updateFlowButton();
}

// 버튼/스페이스바로 토글
document.getElementById('btnFlow')?.addEventListener('click', toggleFlow);
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); toggleFlow(); }
});
