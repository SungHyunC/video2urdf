// The page's backdrop: what a spinning LiDAR sees on a floor, concentric rings of points (denser near the
// sensor, with the gaps a box on the floor leaves behind it), lit by the rotating beam. Animates only while
// the hero is on screen and the tab is visible; one still frame for prefers-reduced-motion; the device
// pixel ratio is capped. It never touches the viewer.
import * as THREE from "three";

const VERT = /* glsl */ `
  uniform float uTime, uSweep, uDpr, uSize, uBeam;
  attribute float aRand, aKind;
  varying float vGlow, vFade, vRand, vKind;
  const float TAU = 6.2831853;
  void main() {
    vec3 p = position;
    p.y += sin(uTime * 0.6 + aRand * TAU) * 0.006;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float dist = -mv.z;
    float ang = atan(p.z, p.x);                       // the sensor is at the origin
    float d = mod(uSweep - ang, TAU);
    vGlow = uBeam * exp(-d * 3.2) * (1.0 - smoothstep(20.0, 36.0, length(p.xz)));
    vFade = (1.0 - smoothstep(8.0, 64.0, dist)) * smoothstep(0.8, 3.0, dist);
    vRand = aRand;
    vKind = aKind;
    float size = uSize * (aKind > 0.5 ? 1.15 : 1.0) * (1.0 + vGlow * 1.3);
    gl_PointSize = clamp(size * uDpr * (0.5 + 6.5 / dist), 1.0, 5.0 * uDpr);
  }`;

const FRAG = /* glsl */ `
  uniform vec3 uBlue, uViolet, uAmber, uCyan;
  varying float vGlow, vFade, vRand, vKind;
  void main() {
    float r = length(gl_PointCoord - 0.5);
    if (r > 0.5) discard;
    float soft = 1.0 - smoothstep(0.28, 0.5, r);
    vec3 base = mix(uBlue, uViolet, vRand * 0.85);
    if (vKind > 0.5) base = mix(base, uCyan, 0.4);
    vec3 col = mix(base, uCyan, smoothstep(0.1, 0.55, vGlow));
    col = mix(col, uAmber, smoothstep(0.6, 0.97, vGlow));
    float a = ((vKind > 0.5 ? 0.5 : 0.42) + vGlow * 0.6) * vFade * soft;
    gl_FragColor = vec4(col * a, a);
  }`;

// boxes standing on the floor (x, z centre; half sizes; height): their faces facing the sensor get scan
// lines, and the floor rings behind them are left empty, as a real scan's shadow
const BOXES = [
  { x: 7.3, z: 0.4, hx: 0.75, hz: 0.65, h: 1.9 },
  { x: -8.4, z: -1.4, hx: 1.2, hz: 0.65, h: 0.75 },
];

function field() {
  const pos = [], rand = [], kind = [];
  const add = (x, y, z, k) => { pos.push(x, y, z); rand.push(Math.random()); kind.push(k); };
  const FLOOR = -1.4;
  const shadows = BOXES.map((b) => {  // angular interval each box hides, and its nearest distance
    const cs = [[b.x - b.hx, b.z - b.hz], [b.x + b.hx, b.z - b.hz], [b.x - b.hx, b.z + b.hz], [b.x + b.hx, b.z + b.hz]];
    const angs = cs.map(([x, z]) => Math.atan2(z, x));
    return { a0: Math.min(...angs), a1: Math.max(...angs), r: Math.min(...cs.map(([x, z]) => Math.hypot(x, z))) };
  });
  const hidden = (a, r) => shadows.some((s) => a > s.a0 && a < s.a1 && r > s.r);
  for (let k = 0; k < 36; k++) {
    const r = 0.9 * Math.pow(1.1, k) + 0.6;                  // ring radii grow with the beam's elevation
    const n = Math.round(Math.min(900, Math.max(260, (2 * Math.PI * r) / 0.085)));
    const wobble = (Math.random() - 0.5) * 0.06;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 - Math.PI + (Math.random() - 0.5) * 0.002;
      const rr = r * (1 + wobble * Math.sin(a * 3 + k)) + (Math.random() - 0.5) * 0.02;
      if (hidden(a, rr)) continue;
      add(rr * Math.cos(a), FLOOR + (Math.random() - 0.5) * 0.012, rr * Math.sin(a), 0);
    }
  }
  for (const b of BOXES) {  // scan lines on the faces the sensor sees
    const faces = [];
    if (b.z - b.hz > 0 || b.z + b.hz < 0) faces.push({ fixed: "z", v: b.z + (b.z > 0 ? -b.hz : b.hz), lo: b.x - b.hx, hi: b.x + b.hx });
    if (b.x - b.hx > 0 || b.x + b.hx < 0) faces.push({ fixed: "x", v: b.x + (b.x > 0 ? -b.hx : b.hx), lo: b.z - b.hz, hi: b.z + b.hz });
    for (const f of faces) for (let y = FLOOR + 0.05; y < FLOOR + b.h; y += 0.09) for (let t = f.lo; t <= f.hi; t += 0.055) {
      if (f.fixed === "z") add(t, y, f.v, 1); else add(f.v, y, t, 1);
    }
    for (let x = b.x - b.hx; x <= b.x + b.hx; x += 0.07) for (let z = b.z - b.hz; z <= b.z + b.hz; z += 0.07) add(x, FLOOR + b.h, z, 1);  // its top
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("aRand", new THREE.Float32BufferAttribute(rand, 1));
  g.setAttribute("aKind", new THREE.Float32BufferAttribute(kind, 1));
  return g;
}

export function start(canvas, { reduced = false, hero = null } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: "low-power" });
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  renderer.setPixelRatio(dpr);
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 140);
  const uniforms = {
    uTime: { value: 0 }, uSweep: { value: 0 }, uDpr: { value: dpr }, uSize: { value: 2.7 }, uBeam: { value: 1 },
    uBlue: { value: new THREE.Color("#3d6ae0") }, uViolet: { value: new THREE.Color("#7b52e6") },
    uAmber: { value: new THREE.Color("#ffb84d") }, uCyan: { value: new THREE.Color("#5fd4ff") },
  };
  const points = new THREE.Points(field(), new THREE.ShaderMaterial({
    uniforms, vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  points.frustumCulled = false;
  scene.add(points);

  let narrow = false;
  function resize() {
    const w = innerWidth, h = innerHeight;
    narrow = w < 760;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = narrow ? 70 : 50;
    camera.updateProjectionMatrix();
  }
  resize();
  let rt;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { resize(); if (!running) draw(last); }, 120); });

  let last = 7.3;  // seconds: the still frame for reduced motion
  function draw(t) {
    last = t;
    uniforms.uTime.value = t;
    uniforms.uSweep.value = ((t * 0.7) % (Math.PI * 2)) - Math.PI;
    const drift = Math.sin(t * 0.04);
    camera.position.set(drift * 2.4, 3.1 + Math.sin(t * 0.07) * 0.2, 9.5 - Math.cos(t * 0.05) * 0.6);
    camera.lookAt(drift * 0.8, -2.4, -9);
    renderer.render(scene, camera);
  }

  let running = false, heroVisible = true;
  const t0 = performance.now() - last * 1000;
  const should = () => !reduced && heroVisible && !document.hidden;
  function loop(now) {
    if (!should()) { running = false; rest(); return; }
    draw((now - t0) / 1000);
    requestAnimationFrame(loop);
  }
  // Off the hero the field rests: no beam, dimmed, so it never runs behind the text of later sections.
  function rest() { uniforms.uBeam.value = 0; draw(last); canvas.classList.toggle("dim", !heroVisible); }
  function kick() {
    if (!running && should()) { uniforms.uBeam.value = 1; canvas.classList.remove("dim"); running = true; requestAnimationFrame(loop); }
    else if (!should() && !running) rest();
  }
  draw(last);
  canvas.classList.add("on");
  if (hero) new IntersectionObserver(([e]) => { heroVisible = e.isIntersecting; kick(); }, { rootMargin: "160px 0px" }).observe(hero);
  document.addEventListener("visibilitychange", kick);
  kick();
}
