import React from 'react';
import {
  AbsoluteFill,
  Easing,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {loadFont as loadInter} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

const {fontFamily: SANS} = loadInter('normal', {weights: ['500', '700', '800'], subsets: ['latin']});
const {fontFamily: MONO} = loadMono('normal', {weights: ['500', '700'], subsets: ['latin']});

const BG = '#07090d';
const YELLOW = '#ffc93c';
const VIOLET = '#8b7cff';
const GREEN = '#3ddc97';
const RED = '#ff5a5f';
const OUT = Easing.bezier(0.16, 1, 0.3, 1);

// Scene timeline (frames at 30 fps).
const SCENES = [
  {id: 'hook', len: 150},
  {id: 'film', len: 105},
  {id: 'track', len: 165},
  {id: 'measure', len: 210},
  {id: 'check', len: 180},
  {id: 'real', len: 150},
  {id: 'physics', len: 150},
  {id: 'more', len: 180},
  {id: 'knows', len: 180},
  {id: 'robots', len: 165},
  {id: 'urdf', len: 90},
  {id: 'end', len: 150},
] as const;
export const PROMO_FRAMES = SCENES.reduce((n, s) => n + s.len, 0);

const ease = (f: number, a: number, b: number, from = 0, to = 1) =>
  interpolate(f, [a, b], [from, to], {easing: OUT, extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

// Fades and settles each scene in, and fades it out at the end.
const Scene: React.FC<{len: number; children: React.ReactNode; bg?: string}> = ({len, children, bg = BG}) => {
  const f = useCurrentFrame();
  const opacity = Math.min(ease(f, 0, 8), interpolate(f, [len - 6, len], [1, 0], {extrapolateLeft: 'clamp'}));
  const scale = ease(f, 0, 20, 1.04, 1);
  return (
    <AbsoluteFill style={{background: bg, opacity, transform: `scale(${scale})`, fontFamily: SANS, color: 'white'}}>
      {children}
    </AbsoluteFill>
  );
};

// Words rise in one after another.
const Words: React.FC<{text: string; delay?: number; stagger?: number; style?: React.CSSProperties; hi?: string[]; hiColor?: string}> = ({
  text,
  delay = 0,
  stagger = 3,
  style,
  hi = [],
  hiColor = YELLOW,
}) => {
  const f = useCurrentFrame();
  return (
    <div style={{display: 'flex', flexWrap: 'wrap', columnGap: '0.28em', ...style}}>
      {text.split(' ').map((w, i) => {
        const t = ease(f, delay + i * stagger, delay + i * stagger + 14);
        return (
          <span
            key={i}
            style={{
              display: 'inline-block',
              opacity: t,
              transform: `translateY(${(1 - t) * 40}px)`,
              filter: `blur(${(1 - t) * 8}px)`,
              color: hi.includes(w.replace(/[.,]/g, '')) ? hiColor : undefined,
            }}
          >
            {w}
          </span>
        );
      })}
    </div>
  );
};

const Pop: React.FC<{delay?: number; children: React.ReactNode; style?: React.CSSProperties}> = ({delay = 0, children, style}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - delay, fps, config: {damping: 14, stiffness: 160}});
  return <div style={{opacity: Math.min(1, s * 1.5), transform: `scale(${0.6 + 0.4 * s})`, ...style}}>{children}</div>;
};

const Chip: React.FC<{color: string; children: React.ReactNode; size?: number}> = ({color, children, size = 30}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 14,
      padding: '14px 26px',
      borderRadius: 999,
      background: 'rgba(255,255,255,0.07)',
      border: `2px solid ${color}`,
      fontSize: size,
      fontWeight: 700,
    }}
  >
    <span style={{width: 14, height: 14, borderRadius: 7, background: color, boxShadow: `0 0 18px ${color}`}} />
    {children}
  </div>
);

const Step: React.FC<{n: string; delay?: number}> = ({n, delay = 0}) => {
  const f = useCurrentFrame();
  return (
    <div style={{fontFamily: MONO, fontSize: 26, letterSpacing: 4, color: YELLOW, opacity: ease(f, delay, delay + 10)}}>{n}</div>
  );
};

const Count: React.FC<{to: number; delay: number; dur?: number; decimals?: number}> = ({to, delay, dur = 30, decimals = 0}) => {
  const f = useCurrentFrame();
  return <>{(to * ease(f, delay, delay + dur)).toFixed(decimals)}</>;
};

const Clip: React.FC<{src: string; startFrom?: number; rate?: number; style?: React.CSSProperties}> = ({src, startFrom = 0, rate = 1, style}) => (
  <OffthreadVideo
    src={staticFile(src)}
    startFrom={startFrom}
    playbackRate={rate}
    muted
    style={{width: '100%', height: '100%', objectFit: 'cover', ...style}}
  />
);

const Card: React.FC<{x: number; y: number; w: number; h: number; delay?: number; children: React.ReactNode; border?: string}> = ({
  x,
  y,
  w,
  h,
  delay = 0,
  children,
  border = 'rgba(255,255,255,0.12)',
}) => {
  const f = useCurrentFrame();
  const t = ease(f, delay, delay + 18);
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width: w,
        height: h,
        borderRadius: 28,
        overflow: 'hidden',
        border: `2px solid ${border}`,
        boxShadow: '0 30px 80px rgba(0,0,0,0.55)',
        opacity: t,
        transform: `translateY(${(1 - t) * 60}px)`,
      }}
    >
      {children}
    </div>
  );
};

const Badge: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <div
    style={{
      position: 'absolute',
      fontFamily: MONO,
      fontSize: 22,
      fontWeight: 700,
      padding: '8px 16px',
      borderRadius: 10,
      background: 'rgba(0,0,0,0.65)',
      color: 'white',
      ...style,
    }}
  >
    {children}
  </div>
);

// 1. Hook: the robot closing the measured laptop.
const Hook: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <Scene len={150} bg="white">
      <AbsoluteFill style={{transform: 'scale(1.1)', transformOrigin: '50% 80%'}}>
        <Clip src="robot.mp4" startFrom={45} rate={1.15} />
      </AbsoluteFill>
      <AbsoluteFill style={{background: 'linear-gradient(180deg, rgba(7,9,13,0) 45%, rgba(7,9,13,0.92) 100%)'}} />
      <Badge style={{left: 60, top: 50, background: 'rgba(7,9,13,0.8)'}}>MuJoCo simulation</Badge>
      <div style={{position: 'absolute', left: 90, bottom: 80, width: 1500}}>
        <Words text="This robot is closing a laptop" delay={4} style={{fontSize: 84, fontWeight: 800, lineHeight: 1.05}} />
        <Words
          text="it only knows from one phone video."
          delay={22}
          style={{fontSize: 84, fontWeight: 800, lineHeight: 1.05}}
          hi={['one', 'phone', 'video']}
        />
      </div>
      <div style={{position: 'absolute', right: 70, bottom: 92, opacity: ease(f, 60, 75)}}>
        <Chip color={GREEN}>hinge measured · 116°</Chip>
      </div>
    </Scene>
  );
};

// 2. The input: one handheld iPhone video.
const Film: React.FC = () => {
  const f = useCurrentFrame();
  const tilt = interpolate(f, [0, 105], [-14, -6]);
  return (
    <Scene len={105}>
      <div style={{position: 'absolute', left: 90, top: 150, perspective: 1600}}>
        <div style={{width: 1040, height: 780, transform: `rotateY(${-tilt}deg)`, borderRadius: 34, overflow: 'hidden', border: '10px solid #1c2030', boxShadow: '0 40px 120px rgba(0,0,0,0.7)'}}>
          <Clip src="phone.mp4" rate={2} />
          <Badge style={{right: 18, top: 18}}>2×</Badge>
        </div>
      </div>
      <div style={{position: 'absolute', left: 1230, top: 300, width: 620}}>
        <Step n="STEP 1 · INPUT" />
        <Words text="Film it moving." delay={4} style={{fontSize: 96, fontWeight: 800, marginTop: 18}} />
        <div style={{display: 'flex', flexDirection: 'column', gap: 18, marginTop: 40, alignItems: 'flex-start'}}>
          <Pop delay={24}><Chip color={YELLOW}>one iPhone video · 18 s</Chip></Pop>
          <Pop delay={34}><Chip color={VIOLET}>handheld, no markers</Chip></Pop>
          <Pop delay={44}><Chip color={GREEN}>LiDAR depth for scale</Chip></Pop>
        </div>
      </div>
    </Scene>
  );
};

// 3. Muse Spark picks the parts, SAM 3.1 tracks them.
const Track: React.FC = () => (
  <Scene len={165}>
    <Card x={80} y={110} w={1146} h={860}>
      <Clip src="masks.mp4" startFrom={0} rate={1.5} />
    </Card>
    <div style={{position: 'absolute', left: 1300, top: 220, width: 560}}>
      <Step n="STEP 2 · AGENT" />
      <Words text="An AI agent finds every part." delay={4} style={{fontSize: 78, fontWeight: 800, marginTop: 18, lineHeight: 1.05}} hi={['every', 'part']} />
      <div style={{display: 'flex', flexDirection: 'column', gap: 18, marginTop: 44, alignItems: 'flex-start'}}>
        <Pop delay={30}><Chip color={VIOLET}>Muse Spark watches &amp; plans</Chip></Pop>
        <Pop delay={48}><Chip color={YELLOW}>SAM 3.1 tracks each part</Chip></Pop>
        <Pop delay={66}><Chip color={GREEN}>on every frame</Chip></Pop>
      </div>
    </div>
  </Scene>
);

// 4. Measure: a wipe between the phone video and the measured model replayed over it.
const Measure: React.FC = () => {
  const f = useCurrentFrame();
  const w = 1080;
  const h = 810;
  const wipe = interpolate(f, [10, 70, 130, 190], [0.02, 0.98, 0.35, 0.65], {
    easing: Easing.inOut(Easing.cubic),
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const rows: [string, React.ReactNode, string, string][] = [
    ['Lid', <><Count to={309} delay={30} /> × <Count to={205} delay={30} /> mm</>, 'Apple spec: 304 × 212', YELLOW],
    ['Hinge range', <><Count to={116} delay={50} />°</>, 'opened and shut on video', VIOLET],
    ['Lid thickness', <><Count to={3.8} delay={70} decimals={1} /> mm</>, 'from depth, no ruler', GREEN],
    ['Replay match', <><Count to={95} delay={90} />%</>, 'lid, from the phone’s own camera', 'white'],
  ];
  return (
    <Scene len={210}>
      <Card x={80} y={135} w={w} h={h}>
        <Clip src="phone.mp4" startFrom={60} rate={1.3} />
        <AbsoluteFill style={{clipPath: `inset(0 0 0 ${wipe * 100}%)`}}>
          <Clip src="replay.mp4" startFrom={60} rate={1.3} />
        </AbsoluteFill>
        <div style={{position: 'absolute', left: `${wipe * 100}%`, top: 0, bottom: 0, width: 4, marginLeft: -2, background: 'white', boxShadow: '0 0 24px white'}} />
        <Badge style={{left: 18, bottom: 18}}>phone video</Badge>
        <Badge style={{right: 18, bottom: 18, background: 'rgba(139,124,255,0.85)'}}>measured model</Badge>
      </Card>
      <div style={{position: 'absolute', left: 1240, top: 130, width: 620}}>
        <Step n="STEP 3 · MEASURE" />
        <Words text="Measured, not guessed." delay={4} style={{fontSize: 72, fontWeight: 800, marginTop: 14, lineHeight: 1.05}} />
        <div style={{marginTop: 34, display: 'flex', flexDirection: 'column', gap: 22}}>
          {rows.map(([k, v, sub, c], i) => (
            <div key={k} style={{opacity: ease(f, 24 + i * 20, 38 + i * 20), transform: `translateX(${(1 - ease(f, 24 + i * 20, 38 + i * 20)) * 40}px)`, borderLeft: `6px solid ${c}`, paddingLeft: 22}}>
              <div style={{fontSize: 24, color: 'rgba(255,255,255,0.6)', fontWeight: 600}}>{k}</div>
              <div style={{fontFamily: MONO, fontSize: 52, fontWeight: 700}}>{v}</div>
              <div style={{fontSize: 22, color: 'rgba(255,255,255,0.5)'}}>{sub}</div>
            </div>
          ))}
        </div>
      </div>
    </Scene>
  );
};

// 5. The agent catches its own bad fit and fixes it.
const Check: React.FC = () => {
  const f = useCurrentFrame();
  const panel = (src: string, ok: boolean, delay: number, x: number) => (
    <Card x={x} y={300} w={800} h={600} delay={delay} border={ok ? GREEN : RED}>
      <Img src={staticFile(src)} style={{width: 1600, height: 600, objectFit: 'cover', objectPosition: 'left'}} />
      <div style={{position: 'absolute', left: 20, bottom: 20}}>
        <Pop delay={delay + 14}>
          <Chip color={ok ? GREEN : RED} size={28}>{ok ? '✓ after its own fix' : '✗ first fit: base is a sliver'}</Chip>
        </Pop>
      </div>
    </Card>
  );
  const msg = 'Spark watches the replay → re-segments the base → re-tracks';
  const typed = msg.slice(0, Math.floor(interpolate(f, [60, 110], [0, msg.length], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})));
  return (
    <Scene len={180}>
      <div style={{position: 'absolute', left: 0, right: 0, top: 80, display: 'flex', justifyContent: 'center'}}>
        <Words text="It checks its own work." delay={2} style={{fontSize: 96, fontWeight: 800}} hi={['checks']} hiColor={GREEN} />
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 220, textAlign: 'center', fontFamily: MONO, fontSize: 30, color: VIOLET}}>
        {typed}
        <span style={{opacity: f % 20 < 10 ? 1 : 0}}>▌</span>
      </div>
      {panel('agent_first_fit.jpg', false, 14, 120)}
      {panel('agent_refit.jpg', true, 110, 1000)}
    </Scene>
  );
};

// 6. The model is painted with the same video.
const Real: React.FC = () => (
  <Scene len={150} bg="white">
    <Clip src="turntable.mp4" rate={1} />
    <AbsoluteFill style={{background: 'linear-gradient(180deg, rgba(7,9,13,0.85) 0%, rgba(7,9,13,0) 30%)'}} />
    <div style={{position: 'absolute', left: 90, top: 60}}>
      <Step n="STEP 4 · LOOK" />
      <Words text="Painted with the same video." delay={4} style={{fontSize: 76, fontWeight: 800, marginTop: 10}} hi={['same', 'video']} />
    </div>
  </Scene>
);

// 7. Physics: friction measured where the lid stands on its own.
const Physics: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <Scene len={150} bg="white">
      <AbsoluteFill style={{top: 60}}>
        <Clip src="physics.mp4" rate={0.6} style={{objectFit: 'contain'}} />
      </AbsoluteFill>
      <AbsoluteFill style={{background: 'linear-gradient(0deg, rgba(7,9,13,0.92) 0%, rgba(7,9,13,0) 38%)'}} />
      <div style={{position: 'absolute', left: 90, bottom: 70}}>
        <Step n="STEP 5 · PHYSICS" />
        <Words text="Physics too: mass and hinge friction." delay={4} style={{fontSize: 70, fontWeight: 800, marginTop: 10}} hi={['friction']} />
      </div>
      <div style={{position: 'absolute', right: 80, bottom: 84, display: 'flex', gap: 18, opacity: ease(f, 40, 55)}}>
        <Chip color={GREEN}>
          <span style={{fontFamily: MONO}}><Count to={0.28} delay={40} decimals={2} /> N·m</span>
        </Chip>
        <Chip color={YELLOW}>1.4 kg</Chip>
      </div>
    </Scene>
  );
};

// 8. Other objects: a drawer slide and a two-hinge desk lamp.
const More: React.FC = () => (
  <Scene len={180}>
    <div style={{position: 'absolute', left: 0, right: 0, top: 70, display: 'flex', justifyContent: 'center'}}>
      <Words text="Laptops. Drawers. Desk lamps." delay={2} stagger={6} style={{fontSize: 88, fontWeight: 800}} hi={['Drawers', 'lamps']} />
    </div>
    <Card x={96} y={250} w={680} h={510} delay={10}>
      <Clip src="drawer_replay.mp4" rate={1} />
    </Card>
    <Card x={816} y={250} w={287} h={510} delay={20}>
      <Clip src="stand_replay.mp4" rate={1.2} />
    </Card>
    <Card x={1143} y={250} w={680} h={510} delay={30}>
      <Clip src="stand_physics.mp4" rate={0.5} style={{objectFit: 'contain', background: 'white'}} />
    </Card>
    <div style={{position: 'absolute', top: 800, left: 96, width: 680, display: 'flex', justifyContent: 'center'}}>
      <Pop delay={40}><Chip color={YELLOW}>slide · 248 mm travel</Chip></Pop>
    </div>
    <div style={{position: 'absolute', top: 800, left: 716, width: 487, display: 'flex', justifyContent: 'center'}}>
      <Pop delay={50}><Chip color={VIOLET}>2 hinges</Chip></Pop>
    </div>
    <div style={{position: 'absolute', top: 800, left: 1143, width: 680, display: 'flex', justifyContent: 'center'}}>
      <Pop delay={60}><Chip color={GREEN}>lamp holds its pose in sim</Chip></Pop>
    </div>
  </Scene>
);

// It knows what it is: the measured meshes next to what Muse Spark / SAM 3D made of them.
const Knows: React.FC = () => (
  <Scene len={180}>
    <div style={{position: 'absolute', left: 0, right: 0, top: 70, display: 'flex', justifyContent: 'center'}}>
      <Words text="It knows what it is." delay={0} style={{fontSize: 84, fontWeight: 800}} hi={['knows']} />
    </div>
    <div style={{position: 'absolute', left: 0, right: 0, top: 185, display: 'flex', justifyContent: 'center'}}>
      <Words text="measured from the video, then rebuilt by Meta models" delay={10} stagger={2}
             style={{fontSize: 38, color: 'rgba(255,255,255,0.7)'}} />
    </div>
    <Card x={60} y={290} w={880} h={500} delay={12} border={YELLOW}>
      <Clip src="drawer_compare.mp4" />
      <Badge style={{left: 20, bottom: 20}}>cabinet · Muse Spark design</Badge>
    </Card>
    <Card x={980} y={290} w={880} h={500} delay={22} border={VIOLET}>
      <Clip src="mouse_compare.mp4" />
      <Badge style={{left: 20, bottom: 20}}>mouse · SAM 3D Objects</Badge>
    </Card>
    <div style={{position: 'absolute', left: 0, right: 0, top: 850, display: 'flex', justifyContent: 'center', gap: 20}}>
      <Pop delay={40}><Chip color={YELLOW}>3 Meta models propose, the video picks</Chip></Pop>
      <Pop delay={52}><Chip color={GREEN}>0.5 mm from the measured depth</Chip></Pop>
    </div>
  </Scene>
);

// The robot uses it: four measured objects, gripped and worked in MuJoCo physics.
const Robots: React.FC = () => (
  <Scene len={165} bg="white">
    <AbsoluteFill>
      <Clip src="robot_grid.mp4" rate={1.4} />
    </AbsoluteFill>
    <AbsoluteFill style={{background: 'linear-gradient(180deg, rgba(7,9,13,0) 50%, rgba(7,9,13,0.92) 100%)'}} />
    <div style={{position: 'absolute', left: 90, bottom: 150}}>
      <Words text="Then a robot uses it." delay={4} style={{fontSize: 84, fontWeight: 800}} hi={['robot']} />
    </div>
    <div style={{position: 'absolute', left: 90, bottom: 70, display: 'flex', gap: 16}}>
      <Pop delay={20}><Chip color={YELLOW} size={26}>pull a drawer by its handle</Chip></Pop>
      <Pop delay={28}><Chip color={VIOLET} size={26}>carry a mouse</Chip></Pop>
      <Pop delay={36}><Chip color={GREEN} size={26}>swing a lamp arm</Chip></Pop>
      <Pop delay={44}><Chip color={RED} size={26}>close a laptop</Chip></Pop>
    </div>
  </Scene>
);

// 9. Output: the URDF itself.
const Urdf: React.FC = () => {
  const f = useCurrentFrame();
  const lines: [string, string][] = [
    ['<joint name="lid_joint" type="revolute">', 'white'],
    ['  <parent link="base"/>  <child link="lid"/>', 'rgba(255,255,255,0.6)'],
    ['  <axis xyz="1 0 0"/>', VIOLET],
    ['  <limit lower="-1.861" upper="0.164"/>', YELLOW],
    ['  <dynamics friction="0.284"/>', GREEN],
    ['</joint>', 'white'],
  ];
  const chars = Math.floor(interpolate(f, [8, 60], [0, lines.reduce((n, [l]) => n + l.length, 0)], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  let left = chars;
  return (
    <Scene len={90}>
      <div style={{position: 'absolute', left: 0, right: 0, top: 110, display: 'flex', justifyContent: 'center'}}>
        <Words text="Out comes a URDF." delay={0} style={{fontSize: 84, fontWeight: 800}} hi={['URDF']} />
      </div>
      <div style={{position: 'absolute', left: 330, top: 300, width: 1260, padding: '44px 54px', borderRadius: 28, background: '#11141c', border: '2px solid rgba(255,255,255,0.12)', fontFamily: MONO, fontSize: 40, lineHeight: 1.6}}>
        {lines.map(([l, c], i) => {
          const shown = l.slice(0, Math.max(0, left));
          left -= l.length;
          return (
            <div key={i} style={{color: c, whiteSpace: 'pre', minHeight: '1.6em'}}>
              {shown}
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 860, display: 'flex', justifyContent: 'center', gap: 20, opacity: ease(f, 50, 62)}}>
        <Chip color={VIOLET} size={26}>URDF</Chip>
        <Chip color={YELLOW} size={26}>MJCF</Chip>
        <Chip color={GREEN} size={26}>meshes + textures</Chip>
      </div>
    </Scene>
  );
};

// 10. End card.
const End: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <Scene len={150}>
      <AbsoluteFill style={{background: `radial-gradient(circle at 50% 40%, rgba(139,124,255,0.25), ${BG} 60%)`}} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
        <Pop delay={2}>
          <div style={{fontSize: 150, fontWeight: 800, letterSpacing: -4}}>
            Video<span style={{color: YELLOW}}>2</span>URDF
          </div>
        </Pop>
        <Words text="One video in. A robot-ready model out." delay={14} style={{fontSize: 52, fontWeight: 600, marginTop: 6, color: 'rgba(255,255,255,0.85)'}} />
        <div style={{display: 'flex', gap: 18, marginTop: 50, opacity: ease(f, 36, 50)}}>
          <Chip color={GREEN} size={28}>open source</Chip>
          <Chip color={VIOLET} size={28}>runs on open weights</Chip>
        </div>
        <div style={{fontFamily: MONO, fontSize: 40, marginTop: 56, color: YELLOW, opacity: ease(f, 50, 64)}}>github.com/SungHyunC/video2urdf</div>
        <div style={{fontSize: 26, marginTop: 40, color: 'rgba(255,255,255,0.55)', opacity: ease(f, 60, 74)}}>
          Built with Muse Spark · SAM 3.1 · SAM 3D · MuJoCo
        </div>
      </AbsoluteFill>
    </Scene>
  );
};

const VIEWS: Record<(typeof SCENES)[number]['id'], React.FC> = {
  hook: Hook,
  film: Film,
  track: Track,
  measure: Measure,
  check: Check,
  real: Real,
  physics: Physics,
  more: More,
  knows: Knows,
  robots: Robots,
  urdf: Urdf,
  end: End,
};

const Progress: React.FC = () => {
  const f = useCurrentFrame();
  return <div style={{position: 'absolute', left: 0, top: 0, height: 6, width: `${(f / PROMO_FRAMES) * 100}%`, background: YELLOW}} />;
};

export const Promo: React.FC = () => {
  let at = 0;
  return (
    <AbsoluteFill style={{background: BG}}>
      {SCENES.map((s) => {
        const View = VIEWS[s.id];
        const from = at;
        at += s.len;
        return (
          <Sequence key={s.id} from={from} durationInFrames={s.len}>
            <View />
          </Sequence>
        );
      })}
      <Progress />
    </AbsoluteFill>
  );
};
