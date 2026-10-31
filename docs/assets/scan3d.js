// "Measured, not guessed": the published laptop model (docs/models/laptop_lidar) drawn as a scan, its
// measured meshes only, the hinge axis lit, the lid driven by the joint angle measured on each video frame,
// and the lid's measured outline (model.json's plate_size_mm) as dimension lines. Runs only while visible.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const BASE = "models/laptop_lidar/";
const AMBER = 0xffb84d;

const PV = /* glsl */ `
  uniform float uScan, uDpr;
  varying float vGlow;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    float d = wp.x - uScan;                       // a scan plane sweeping across the model
    vGlow = exp(-d * d * 900.0) + 0.35 * exp(-max(-d, 0.0) * 18.0) * step(d, 0.0);
    vec4 mv = viewMatrix * wp;
    gl_Position = projectionMatrix * mv;
    gl_PointSize = clamp((1.4 + vGlow * 1.8) * uDpr * 0.9 / -mv.z, 1.0, 5.0 * uDpr);
  }`;
const PF = /* glsl */ `
  varying float vGlow;
  void main() {
    float r = length(gl_PointCoord - 0.5);
    if (r > 0.5) discard;
    vec3 col = mix(vec3(0.42, 0.6, 1.0), vec3(1.0, 0.75, 0.35), smoothstep(0.35, 0.95, vGlow));
    float a = (0.38 + vGlow * 0.6) * (1.0 - smoothstep(0.28, 0.5, r));
    gl_FragColor = vec4(col * a, a);
  }`;

function holdFill(values) {
  const out = []; let last = values.find((v) => v !== null) ?? 0;
  for (const v of values) { if (v !== null) last = v; out.push(last); }
  return out;
}

function samplePoints(geo, n) {  // area-weighted points on a mesh's triangles
  const p = geo.attributes.position, idx = geo.index;
  const tri = (i) => idx ? [idx.getX(3 * i), idx.getX(3 * i + 1), idx.getX(3 * i + 2)] : [3 * i, 3 * i + 1, 3 * i + 2];
  const count = (idx ? idx.count : p.count) / 3;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const areas = []; let total = 0;
  for (let i = 0; i < count; i++) {
    const [i0, i1, i2] = tri(i);
    a.fromBufferAttribute(p, i0); b.fromBufferAttribute(p, i1); c.fromBufferAttribute(p, i2);
    total += b.clone().sub(a).cross(c.clone().sub(a)).length() / 2;
    areas.push(total);
  }
  const out = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const r = Math.random() * total;
    let lo = 0, hi = areas.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (areas[m] < r) lo = m + 1; else hi = m; }
    const [i0, i1, i2] = tri(lo);
    a.fromBufferAttribute(p, i0); b.fromBufferAttribute(p, i1); c.fromBufferAttribute(p, i2);
    let u = Math.random(), v = Math.random();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const q = a.clone().add(b.clone().sub(a).multiplyScalar(u)).add(c.clone().sub(a).multiplyScalar(v));
    out.set([q.x, q.y, q.z], 3 * k);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(out, 3));
  return g;
}

function dimension(from, to, normal, tick, mat) {  // a CAD-style dimension line with end ticks
  const n = normal.clone().normalize().multiplyScalar(tick);
  const pts = [from, to, from.clone().sub(n), from.clone().add(n), to.clone().sub(n), to.clone().add(n)];
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineSegments(g, mat);
}

export async function start({ card, reduced = false, finePointer = true }) {
  const stage = card.querySelector("#scanStage"), canvas = card.querySelector("#scanCanvas");
  const spec = await (await fetch(BASE + "model.json")).json();
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
  renderer.setPixelRatio(dpr);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 16 / 11, 0.01, 20);
  scene.add(new THREE.HemisphereLight(0xc8d6ff, 0x101522, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.2); key.position.set(1, 2, 1.2); scene.add(key);
  const rim = new THREE.DirectionalLight(0x9b6dff, 1.0); rim.position.set(-1.4, 0.8, -1.6); scene.add(rim);

  const pointMat = new THREE.ShaderMaterial({ uniforms: { uScan: { value: 0 }, uDpr: { value: dpr } }, vertexShader: PV, fragmentShader: PF,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const solidMat = new THREE.MeshStandardMaterial({ color: 0x18213a, roughness: 0.55, metalness: 0.25, transparent: true, opacity: 0.92,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x8fb0ff, transparent: true, opacity: 0.75 });
  const dimMat = new THREE.LineBasicMaterial({ color: AMBER, transparent: true, opacity: 0.95 });
  const loader = new OBJLoader();

  let lid = null;  // { node, moving, axis, verts: [Vector3] }
  async function body(b) {
    const node = new THREE.Group(); node.position.fromArray(b.pos);
    const moving = new THREE.Group(); node.add(moving);
    const verts = [];
    for (const g of b.geoms) {
      if (!g.measured || !g.mesh) continue;
      const geo = (await loader.loadAsync(BASE + g.mesh)).children[0].geometry;
      geo.deleteAttribute("uv"); geo.deleteAttribute("normal");
      const merged = geo.index ? geo : geo;  // OBJLoader gives non-indexed triangles
      merged.computeVertexNormals();
      moving.add(new THREE.Mesh(merged, solidMat));
      moving.add(new THREE.LineSegments(new THREE.EdgesGeometry(merged, 28), edgeMat));
      moving.add(new THREE.Points(samplePoints(merged, 2600), pointMat));
      const p = merged.attributes.position;
      for (let i = 0; i < p.count; i++) verts.push(new THREE.Vector3().fromBufferAttribute(p, i));
    }
    if (b.joint && b.joint.type === "hinge") lid = { node, moving, axis: new THREE.Vector3(...b.joint.axis).normalize(), verts, name: b.joint.name };
    for (const c of b.children) moving.add(await body(c));
    return node;
  }
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;  // MuJoCo z-up -> three.js y-up
  root.add(await body(spec.tree));
  scene.add(root);
  if (!lid) return;

  // the hinge axis, lit, and the lid's measured outline as dimension lines (in the lid's frame)
  const ax = lid.axis;
  const along = lid.verts.map((v) => v.dot(ax));
  const lo = Math.min(...along), hi = Math.max(...along), mid = (lo + hi) / 2, width = hi - lo;
  const perp = (v) => v.clone().sub(ax.clone().multiplyScalar(v.dot(ax)));
  let far = lid.verts[0];
  for (const v of lid.verts) if (perp(v).length() > perp(far).length()) far = v;
  const u = perp(far).normalize(), reach = perp(far).length();
  const axisGroup = new THREE.Group();
  const len = width * 1.45;
  const orient = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), ax);
  const glow = [];
  for (const [r, o] of [[0.0016, 1], [0.005, 0.28], [0.011, 0.1]]) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 20, 1, true),
      new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: o, depthWrite: false, blending: THREE.AdditiveBlending }));
    m.quaternion.copy(orient); m.position.copy(ax.clone().multiplyScalar(mid));
    axisGroup.add(m); glow.push(m);
  }
  for (const s of [-1, 1]) {
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.0045, 16, 12), new THREE.MeshBasicMaterial({ color: AMBER }));
    cap.position.copy(ax.clone().multiplyScalar(mid + s * len / 2)); axisGroup.add(cap);
  }
  lid.node.add(axisGroup);  // fixed in the parent: the lid turns about it
  const off = 0.022;
  const wA = ax.clone().multiplyScalar(lo).add(u.clone().multiplyScalar(reach + off));
  const wB = ax.clone().multiplyScalar(hi).add(u.clone().multiplyScalar(reach + off));
  lid.moving.add(dimension(wA, wB, u, 0.008, dimMat));
  const sideN = ax.clone();
  const dA = ax.clone().multiplyScalar(hi + off);
  const dB = ax.clone().multiplyScalar(hi + off).add(u.clone().multiplyScalar(reach));
  lid.moving.add(dimension(dA, dB, sideN, 0.008, dimMat));
  const anchors = {
    dimW: { obj: lid.moving, p: wA.clone().add(wB).multiplyScalar(0.5).add(u.clone().multiplyScalar(0.03)) },
    dimD: { obj: lid.moving, p: dA.clone().add(dB).multiplyScalar(0.5).add(ax.clone().multiplyScalar(0.045)) },
    dimAxis: { obj: lid.node, p: ax.clone().multiplyScalar(lo - width * 0.12) },
  };
  const m = spec.measured || {};
  if (m.plate_size_mm) {
    card.querySelector("#dimW span").textContent = `${m.plate_size_mm[0].toFixed(0)} mm`;
    card.querySelector("#dimD span").textContent = `${m.plate_size_mm[1].toFixed(0)} mm`;
  }

  // floor grid under the base
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  const grid = new THREE.GridHelper(1.2, 24, 0x41558a, 0x1c2640);
  grid.material.transparent = true; grid.material.opacity = 0.55;
  grid.position.y = box.min.y - 0.002;
  scene.add(grid);

  // camera from the model's front, a little to the side and above (as the viewer frames it)
  const f = spec.front ? new THREE.Vector3(spec.front[0], 0, -spec.front[1]) : new THREE.Vector3(0.6, 0, 0.8);
  const side = new THREE.Vector3(-f.z, 0, f.x).multiplyScalar(0.8);
  camera.position.copy(center).add(f.add(side).normalize().multiplyScalar(1.25 * size).add(new THREE.Vector3(0, 0.62 * size, 0)));
  camera.near = size / 100; camera.far = size * 20; camera.updateProjectionMatrix();
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(center).add(new THREE.Vector3(0, 0.04 * size, 0));
  controls.enableDamping = !reduced; controls.enableZoom = false; controls.enablePan = false;
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.55;
  controls.minPolarAngle = 0.35; controls.maxPolarAngle = 1.45;
  if (!finePointer) { controls.enabled = false; canvas.style.touchAction = "pan-y"; }  // don't trap a phone's scroll
  controls.update();

  // the measured joint angle, frame by frame
  const track = holdFill(spec.joints[lid.name] || [0]);
  const fps = spec.fps || 30, frames = track.length;
  const qMin = Math.min(...track), qMax = Math.max(...track);
  const path = track.map((q, i) => `${i ? "L" : "M"}${(i / (frames - 1) * 300).toFixed(1)} ${(37 - (q - qMin) / (qMax - qMin || 1) * 34).toFixed(1)}`).join("");
  card.querySelector("#scanPath").setAttribute("d", path);
  const cursor = card.querySelector("#scanCursor"), readQ = card.querySelector("#scanQ"), readF = card.querySelector("#scanF");
  const els = Object.fromEntries(Object.keys(anchors).map((k) => [k, card.querySelector("#" + k)]));

  const radius = size / 2;
  function resize() {  // keep the whole model in view at any aspect (a phone's card is narrow)
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const vf = camera.fov * Math.PI / 180, hf = 2 * Math.atan(Math.tan(vf / 2) * camera.aspect);
    const dist = radius / Math.sin(Math.min(vf, hf) / 2) * 0.8;
    const off = camera.position.clone().sub(controls.target).setLength(dist);
    camera.position.copy(controls.target).add(off);
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  const v = new THREE.Vector3();
  let frame = 0;
  function draw(t, fromControls = false) {
    frame = reduced ? Math.floor(frames * 0.12) : Math.floor(t * fps) % frames;
    const q = track[frame];
    lid.moving.quaternion.setFromAxisAngle(lid.axis, q);
    pointMat.uniforms.uScan.value = box.min.x - 0.1 + ((t * 0.16) % 1) * (box.max.x - box.min.x + 0.2);
    const pulse = 0.85 + 0.15 * Math.sin(t * 3.0);
    glow[1].material.opacity = 0.28 * pulse; glow[2].material.opacity = 0.1 * pulse;
    if (!fromControls) controls.update();  // update() emits "change", which redraws in reduced motion
    renderer.render(scene, camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    for (const [k, a] of Object.entries(anchors)) {
      v.copy(a.p); a.obj.localToWorld(v); v.project(camera);
      els[k].style.transform = `translate(${((v.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v.y) / 2 * h).toFixed(1)}px)`;
      els[k].style.visibility = v.z < 1 ? "visible" : "hidden";
    }
    readQ.textContent = `${(q * 180 / Math.PI).toFixed(0)}°`;
    readF.textContent = `${frame + 1} / ${frames}`;
    cursor.setAttribute("x1", (frame / (frames - 1) * 300).toFixed(1));
    cursor.setAttribute("x2", (frame / (frames - 1) * 300).toFixed(1));
  }

  let visible = false, running = false;
  const t0 = performance.now();
  const should = () => visible && !document.hidden && !reduced;
  function loop(now) {
    if (!should()) { running = false; return; }
    draw((now - t0) / 1000);
    requestAnimationFrame(loop);
  }
  const kick = () => { if (!running && should()) { running = true; requestAnimationFrame(loop); } };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; kick(); }).observe(stage);
  document.addEventListener("visibilitychange", kick);
  if (reduced) controls.addEventListener("change", () => draw(0, true));
  draw(0);
  card.classList.add("ready");
  kick();
}
