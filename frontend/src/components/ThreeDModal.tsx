import React, { useState, useEffect, useRef } from 'react';
import { 
  X, 
  Box, 
  ExternalLink, 
  Loader2, 
  RotateCw, 
  Layers, 
  Compass, 
  ZoomIn, 
  ZoomOut, 
  Sparkles,
  Maximize2,
  RefreshCw,
  AlertCircle
} from 'lucide-react';

interface ThreeDModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelUrl: string | null;
  monumentName: string;
}

// Verified live Sketchfab 3D models with active embed authorizations
const VERIFIED_MODELS: Record<string, { embed: string; source: string; style: string; height: string }> = {
  taj_mahal: {
    embed: 'https://sketchfab.com/models/33149233cefd492b9abdd50fe5a8c921/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/taj-mahal-33149233cefd492b9abdd50fe5a8c921',
    style: 'Mughal Architecture',
    height: '73 meters'
  },
  qutub_minar: {
    embed: 'https://sketchfab.com/models/e165e25ec8f1497aa83ab72419f40ddc/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/qutub-minar-e165e25ec8f1497aa83ab72419f40ddc',
    style: 'Indo-Islamic (Ghurid / Mamluk)',
    height: '72.5 meters'
  },
  red_fort: {
    embed: 'https://sketchfab.com/models/74ff6d703a174f9fb1ba266003f5c4fc/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/red-fort-delhi-74ff6d703a174f9fb1ba266003f5c4fc',
    style: 'Mughal Fortified Architecture',
    height: '33 meters (Walls)'
  },
  india_gate: {
    embed: 'https://sketchfab.com/models/59fe55328271479d82acb65310178d99/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/india-gate-59fe55328271479d82acb65310178d99',
    style: 'Neo-Classical Triumphal Arch',
    height: '42 meters'
  },
  charminar: {
    embed: 'https://sketchfab.com/models/82603f1fe3ed4c31b37786e2bc2b63b0/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/charminar-hyderabad-82603f1fe3ed4c31b37786e2bc2b63b0',
    style: 'Indo-Islamic (Qutb Shahi)',
    height: '48.7 meters'
  },
  hawa_mahal: {
    embed: 'https://sketchfab.com/models/47c0ea2a759544729c137d8b86ba5b9a/embed?autostart=1',
    source: 'https://sketchfab.com/3d-models/the-hawa-mahal-wall-47c0ea2a759544729c137d8b86ba5b9a',
    style: 'Rajput Architecture',
    height: '15 meters (5 Storeys)'
  }
};

function getMonumentKey(name: string): string {
  const norm = name.toLowerCase().replace(/[^a-z0-9]/g, '_');
  if (norm.includes('taj')) return 'taj_mahal';
  if (norm.includes('qutub') || norm.includes('qutb')) return 'qutub_minar';
  if (norm.includes('red') || norm.includes('qila') || norm.includes('lal')) return 'red_fort';
  if (norm.includes('india') || norm.includes('gate')) return 'india_gate';
  if (norm.includes('charminar')) return 'charminar';
  if (norm.includes('hawa') || norm.includes('wind')) return 'hawa_mahal';
  return 'taj_mahal';
}

function resolveModelUrl(rawUrl: string | null, name: string): { embedUrl: string; sourceUrl: string } {
  const key = getMonumentKey(name);
  const info = VERIFIED_MODELS[key];

  // If rawUrl is missing, or is one of the old broken 404 dummy hashes, use verified live model
  const isOldBroken = !rawUrl || 
    rawUrl.includes('31f47285') || 
    rawUrl.includes('bf041ca4') || 
    rawUrl.includes('a5113eb7') || 
    rawUrl.includes('ca6c729e') || 
    rawUrl.includes('2d2fe62f') || 
    rawUrl.includes('8333bc7b');

  if (isOldBroken && info) {
    return { embedUrl: info.embed, sourceUrl: info.source };
  }

  let finalEmbed = rawUrl || info?.embed || '';
  if (!finalEmbed.includes('autostart=')) {
    finalEmbed += (finalEmbed.includes('?') ? '&' : '?') + 'autostart=1';
  }
  const finalSource = finalEmbed.replace('/embed', '').split('?')[0];
  return { embedUrl: finalEmbed, sourceUrl: finalSource };
}

// =========================================================================
// Native 3D CAD Architectural Explorer (Pure HTML5 Canvas 3D Engine)
// Works 100% offline with zero dependencies and no third-party iframe errors.
// =========================================================================

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface Face3D {
  pts: Point3D[];
  color: string;
}

function generateMonumentGeometry(key: string): { faces: Face3D[]; name: string; info: string } {
  const faces: Face3D[] = [];

  const addBox = (
    cx: number, cy: number, cz: number,
    w: number, h: number, d: number,
    colorTop: string, colorSide: string
  ) => {
    const hw = w / 2, hh = h / 2, hd = d / 2;
    const p = [
      { x: cx - hw, y: cy - hh, z: cz - hd }, // 0: LBF
      { x: cx + hw, y: cy - hh, z: cz - hd }, // 1: RBF
      { x: cx + hw, y: cy + hh, z: cz - hd }, // 2: RTF
      { x: cx - hw, y: cy + hh, z: cz - hd }, // 3: LTF
      { x: cx - hw, y: cy - hh, z: cz + hd }, // 4: LBB
      { x: cx + hw, y: cy - hh, z: cz + hd }, // 5: RBB
      { x: cx + hw, y: cy + hh, z: cz + hd }, // 6: RTB
      { x: cx - hw, y: cy + hh, z: cz + hd }, // 7: LTB
    ];

    // Front, Back, Left, Right, Top, Bottom
    faces.push({ pts: [p[0], p[1], p[2], p[3]], color: colorSide });
    faces.push({ pts: [p[5], p[4], p[7], p[6]], color: colorSide });
    faces.push({ pts: [p[4], p[0], p[3], p[7]], color: colorSide });
    faces.push({ pts: [p[1], p[5], p[6], p[2]], color: colorSide });
    faces.push({ pts: [p[3], p[2], p[6], p[7]], color: colorTop });
    faces.push({ pts: [p[4], p[5], p[1], p[0]], color: '#2a2622' });
  };

  const addDome = (
    cx: number, cy: number, cz: number,
    r: number, h: number, segs: number,
    color: string
  ) => {
    const rings = 4;
    for (let i = 0; i < rings; i++) {
      const phi1 = (i / rings) * (Math.PI / 2);
      const phi2 = ((i + 1) / rings) * (Math.PI / 2);
      const y1 = cy + Math.sin(phi1) * h;
      const y2 = cy + Math.sin(phi2) * h;
      const r1 = Math.cos(phi1) * r;
      const r2 = Math.cos(phi2) * r;

      for (let j = 0; j < segs; j++) {
        const theta1 = (j / segs) * Math.PI * 2;
        const theta2 = ((j + 1) / segs) * Math.PI * 2;

        const p1 = { x: cx + r1 * Math.cos(theta1), y: y1, z: cz + r1 * Math.sin(theta1) };
        const p2 = { x: cx + r1 * Math.cos(theta2), y: y1, z: cz + r1 * Math.sin(theta2) };
        const p3 = { x: cx + r2 * Math.cos(theta2), y: y2, z: cz + r2 * Math.sin(theta2) };
        const p4 = { x: cx + r2 * Math.cos(theta1), y: y2, z: cz + r2 * Math.sin(theta1) };

        faces.push({ pts: [p1, p2, p3, p4], color });
      }
    }
  };

  const addColumn = (
    cx: number, cy: number, cz: number,
    r: number, h: number, segs: number,
    color: string
  ) => {
    for (let j = 0; j < segs; j++) {
      const th1 = (j / segs) * Math.PI * 2;
      const th2 = ((j + 1) / segs) * Math.PI * 2;
      const p1 = { x: cx + r * Math.cos(th1), y: cy - h / 2, z: cz + r * Math.sin(th1) };
      const p2 = { x: cx + r * Math.cos(th2), y: cy - h / 2, z: cz + r * Math.sin(th2) };
      const p3 = { x: cx + r * Math.cos(th2), y: cy + h / 2, z: cz + r * Math.sin(th2) };
      const p4 = { x: cx + r * Math.cos(th1), y: cy + h / 2, z: cz + r * Math.sin(th1) };
      faces.push({ pts: [p1, p2, p3, p4], color });
    }
  };

  if (key === 'taj_mahal') {
    // Marble plinth
    addBox(0, -60, 0, 150, 10, 150, '#E8E4DC', '#D4CEBE');
    // Central mausoleum block
    addBox(0, -10, 0, 80, 90, 80, '#F5F2EA', '#DDD7C7');
    // Arched portal insets (iwans)
    addBox(0, -15, 41, 35, 60, 4, '#38322B', '#2B2620');
    addBox(0, -15, -41, 35, 60, 4, '#38322B', '#2B2620');
    addBox(41, -15, 0, 4, 60, 35, '#38322B', '#2B2620');
    addBox(-41, -15, 0, 4, 60, 35, '#38322B', '#2B2620');
    // Main central dome & drum
    addColumn(0, 40, 0, 26, 15, 12, '#EFECE2');
    addDome(0, 47, 0, 30, 42, 12, '#FAF8F2');
    // 4 Corner Chattris
    const dOff = 26;
    [[-dOff, dOff], [dOff, dOff], [dOff, -dOff], [-dOff, -dOff]].forEach(([cx, cz]) => {
      addColumn(cx, 40, cz, 7, 10, 8, '#E5E0D2');
      addDome(cx, 45, cz, 8, 12, 8, '#F7F5EE');
    });
    // 4 Corner Minarets with balconies
    const mOff = 62;
    [[-mOff, mOff], [mOff, mOff], [mOff, -mOff], [-mOff, -mOff]].forEach(([mx, mz]) => {
      addColumn(mx, -5, mz, 6, 100, 8, '#ECE7DC');
      // Balcony rings
      addBox(mx, -30, mz, 16, 3, 16, '#DDD7C7', '#C8C1B0');
      addBox(mx, 10, mz, 14, 3, 14, '#DDD7C7', '#C8C1B0');
      addBox(mx, 40, mz, 12, 3, 12, '#DDD7C7', '#C8C1B0');
      // Minaret cupola
      addDome(mx, 46, mz, 6, 12, 8, '#FAF8F2');
    });
    return { faces, name: 'Taj Mahal', info: 'Bilateral Symmetry • Onion Dome • 4 Freestanding Minarets' };
  }

  if (key === 'qutub_minar') {
    // Multi-storey tapering fluted minaret
    const storeys = [
      { y: -70, rB: 28, rT: 24, h: 35, col: '#B34A32', bW: 34 },
      { y: -35, rB: 23, rT: 19, h: 35, col: '#C25539', bW: 29 },
      { y: 0, rB: 18, rT: 14, h: 35, col: '#B34A32', bW: 24 },
      { y: 35, rB: 13, rT: 10, h: 35, col: '#E0D8C8', bW: 19 },
      { y: 70, rB: 9, rT: 7, h: 35, col: '#EFEAE0', bW: 15 }
    ];

    addBox(0, -90, 0, 80, 10, 80, '#8C3D2B', '#733223');

    storeys.forEach(s => {
      addColumn(0, s.y, 0, (s.rB + s.rT) / 2, s.h, 16, s.col);
      // Projecting cantilevered balcony
      addBox(0, s.y + s.h / 2, 0, s.bW, 4, s.bW, '#5C2417', '#4A1C12');
    });
    // Crown lantern
    addDome(0, 88, 0, 7, 14, 10, '#E0D8C8');
    return { faces, name: 'Qutub Minar', info: '72.5m Height • 5 Tapering Storeys • Cantilevered Balconies' };
  }

  if (key === 'india_gate') {
    // Base plinth
    addBox(0, -65, 0, 130, 10, 80, '#D9A066', '#B8824D');
    // Main Arch Piers (Left and Right)
    addBox(-38, -10, 0, 36, 100, 60, '#E6AF7A', '#C7935F');
    addBox(38, -10, 0, 36, 100, 60, '#E6AF7A', '#C7935F');
    // Top Arch Architrave & Inscribed Attic
    addBox(0, 45, 0, 114, 25, 58, '#DF9F63', '#C7884B');
    // Stepped Cornice Top
    addBox(0, 62, 0, 95, 12, 48, '#D99352', '#B87535');
    addBox(0, 72, 0, 70, 8, 36, '#C47E3D', '#A6662B');
    // Cenotaph bowl finial on roof
    addDome(0, 76, 0, 14, 8, 12, '#9E5D24');
    // Central Vault Void Backdrop
    addBox(0, -10, 0, 40, 70, 4, '#241B12', '#1A130B');
    // Amar Jawan Jyoti Plinth
    addBox(0, -56, 0, 18, 8, 18, '#1A1817', '#11100F');
    return { faces, name: 'India Gate', info: '42m Triumphal Arch • Sandstone Vault • Stepped Attic' };
  }

  if (key === 'charminar') {
    // Main cubic gateway base
    addBox(0, -35, 0, 100, 50, 100, '#C9B8A3', '#AA9780');
    // Grand arches on all 4 sides (void insets)
    addBox(0, -35, 51, 48, 42, 4, '#2B2620', '#1F1B17');
    addBox(0, -35, -51, 48, 42, 4, '#2B2620', '#1F1B17');
    addBox(51, -35, 0, 4, 42, 48, '#2B2620', '#1F1B17');
    addBox(-51, -35, 0, 4, 42, 48, '#2B2620', '#1F1B17');
    // Upper Mosque level
    addBox(0, -5, 0, 96, 20, 96, '#BAA892', '#9C8974');
    // 4 Corner Minarets (48.7m)
    const cOff = 48;
    [[-cOff, cOff], [cOff, cOff], [cOff, -cOff], [-cOff, -cOff]].forEach(([cx, cz]) => {
      addColumn(cx, 0, cz, 8, 120, 10, '#D4C4AF');
      // Double carved balconies
      addBox(cx, -10, cz, 22, 4, 22, '#998670', '#85735D');
      addBox(cx, 25, cz, 20, 4, 20, '#998670', '#85735D');
      addBox(cx, 55, cz, 18, 4, 18, '#998670', '#85735D');
      // Minaret dome
      addDome(cx, 61, cz, 9, 14, 10, '#E6D7C3');
    });
    return { faces, name: 'Charminar', info: '4 Grand Arches • 48.7m Corner Minarets • Qutb Shahi Style' };
  }

  if (key === 'red_fort') {
    // Massive fortified sandstone ramparts
    addBox(0, -50, 0, 170, 45, 60, '#A63321', '#8C2616');
    // Left & Right Octagonal Bastions (Lahori Gate towers)
    addColumn(-50, -20, 32, 22, 85, 8, '#B83B26');
    addColumn(50, -20, 32, 22, 85, 8, '#B83B26');
    // Central entrance archway recess
    addBox(0, -45, 31, 36, 40, 6, '#24100C', '#170A07');
    // Crenellated battlements top
    addBox(0, -25, 0, 160, 6, 56, '#C4412B', '#992B19');
    // Chattri pavilions atop towers
    addDome(-50, 23, 32, 16, 18, 8, '#EAE1CE');
    addDome(50, 23, 32, 16, 18, 8, '#EAE1CE');
    // Central decorative pavilion
    addBox(0, -18, 15, 45, 12, 18, '#B83B26', '#8C2616');
    addDome(0, -11, 15, 12, 12, 8, '#EAE1CE');
    return { faces, name: 'Red Fort', info: 'Red Sandstone Ramparts • Octagonal Bastions • Lahori Gate' };
  }

  // Default: Hawa Mahal (Honeycomb 5-storey pyramid)
  addBox(0, -65, 0, 150, 15, 30, '#D95E50', '#BF4B3D');
  const tiers = [
    { y: -45, w: 140, h: 25, d: 24, col: '#E06859' },
    { y: -22, w: 115, h: 22, d: 22, col: '#E87566' },
    { y: -2, w: 90, h: 20, d: 20, col: '#ED8072' },
    { y: 16, w: 65, h: 18, d: 18, col: '#F28E80' },
    { y: 32, w: 40, h: 16, d: 16, col: '#F79C8F' }
  ];
  tiers.forEach(t => {
    addBox(0, t.y, 0, t.w, t.h, t.d, t.col, '#C75142');
    // Projecting jharokha casements
    const numJ = Math.floor(t.w / 22);
    for (let k = -numJ / 2 + 0.5; k < numJ / 2; k++) {
      addBox(k * 22, t.y, t.d / 2 + 3, 10, t.h - 6, 6, '#FFAEA1', '#D45C4D');
    }
  });
  // Crown finials (Krishna crown peak)
  addDome(0, 41, 0, 14, 16, 8, '#FFE6E0');
  return { faces, name: 'Hawa Mahal', info: '5-Storey Honeycomb Facade • 953 Jharokhas • Venturi Airflow' };
}

export const ThreeDModal: React.FC<ThreeDModalProps> = ({
  isOpen,
  onClose,
  modelUrl,
  monumentName
}) => {
  const [activeTab, setActiveTab] = useState<'sketchfab' | 'cad'>('sketchfab');
  const [iframeLoading, setIframeLoading] = useState(true);
  const [iframeError, setIframeError] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [rotation, setRotation] = useState({ yaw: 0.6, pitch: 0.35 });
  const [zoom, setZoom] = useState(1.0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });

  const resolved = resolveModelUrl(modelUrl, monumentName);
  const monumentKey = getMonumentKey(monumentName);

  // Reset loading state when modal opens
  useEffect(() => {
    if (isOpen) {
      setIframeLoading(true);
      setIframeError(false);
    }
  }, [isOpen, modelUrl]);

  // 3D Canvas Rendering Loop for Native CAD Explorer
  useEffect(() => {
    if (!isOpen || activeTab !== 'cad') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    const { faces } = generateMonumentGeometry(monumentKey);

    const render = () => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      // Background gradient
      const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
      bgGrad.addColorStop(0, '#1C1917');
      bgGrad.addColorStop(1, '#0C0A09');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // Grid ground plane
      ctx.strokeStyle = 'rgba(214, 180, 140, 0.12)';
      ctx.lineWidth = 1;
      const gridSize = 180;
      const step = 30;
      for (let gx = -gridSize; gx <= gridSize; gx += step) {
        const p1 = project({ x: gx, y: -65, z: -gridSize }, rotation, zoom, W, H);
        const p2 = project({ x: gx, y: -65, z: gridSize }, rotation, zoom, W, H);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
      for (let gz = -gridSize; gz <= gridSize; gz += step) {
        const p1 = project({ x: -gridSize, y: -65, z: gz }, rotation, zoom, W, H);
        const p2 = project({ x: gridSize, y: -65, z: gz }, rotation, zoom, W, H);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // Sort faces by depth (Painter's algorithm)
      const sortedFaces = faces.map(face => {
        const projPts = face.pts.map(pt => projectWithDepth(pt, rotation, zoom, W, H));
        const avgZ = projPts.reduce((acc, p) => acc + (p?.depth || 0), 0) / face.pts.length;
        return { face, projPts, avgZ };
      }).sort((a, b) => b.avgZ - a.avgZ);

      // Render 3D polygons
      sortedFaces.forEach(({ face, projPts }) => {
        if (projPts.some(p => p === null)) return;
        const valid = projPts as { x: number; y: number; depth: number }[];

        ctx.beginPath();
        ctx.moveTo(valid[0].x, valid[0].y);
        for (let i = 1; i < valid.length; i++) {
          ctx.lineTo(valid[i].x, valid[i].y);
        }
        ctx.closePath();

        if (!wireframe) {
          ctx.fillStyle = face.color;
          ctx.fill();
        }

        ctx.strokeStyle = wireframe ? '#E6A15C' : 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = wireframe ? 1.2 : 0.6;
        ctx.stroke();
      });

      // Auto-rotation update
      if (autoRotate && !isDraggingRef.current) {
        setRotation(prev => ({ ...prev, yaw: prev.yaw + 0.008 }));
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [isOpen, activeTab, rotation, zoom, wireframe, autoRotate, monumentKey]);

  if (!isOpen) return null;

  // Mouse / Touch Drag Handlers for Native 3D CAD
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - lastMouseRef.current.x;
    const dy = e.clientY - lastMouseRef.current.y;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    setRotation(prev => ({
      yaw: prev.yaw + dx * 0.012,
      pitch: Math.max(-0.9, Math.min(1.1, prev.pitch + dy * 0.01))
    }));
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.4, Math.min(2.5, prev - e.deltaY * 0.0015)));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-stone-900/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="w-full max-w-5xl bg-white rounded-2xl shadow-2xl border border-[#E5DFD7] overflow-hidden flex flex-col h-[88vh]">
        {/* Header Bar */}
        <div className="px-5 py-3.5 border-b border-[#E5DFD7] flex flex-wrap items-center justify-between gap-3 bg-[#FAF8F5]">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-amber-100 text-[#C85A32] shadow-2xs">
              <Box className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="font-serif font-bold text-base text-[#1F2923]">
                  {monumentName}
                </h3>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-200 uppercase tracking-wider">
                  3D Spatial Model
                </span>
              </div>
              <p className="text-[11px] text-stone-500">
                {VERIFIED_MODELS[monumentKey]?.style || 'Architectural Geometry'} • Rotatable 360° Inspection
              </p>
            </div>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex items-center space-x-2">
            <div className="inline-flex rounded-lg border border-[#E5DFD7] p-0.5 bg-stone-100 text-xs">
              <button
                onClick={() => setActiveTab('sketchfab')}
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-md font-medium transition-all ${
                  activeTab === 'sketchfab'
                    ? 'bg-white text-stone-900 shadow-2xs'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5 text-[#C85A32]" />
                <span>Sketchfab Model</span>
              </button>
              <button
                onClick={() => setActiveTab('cad')}
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-md font-medium transition-all ${
                  activeTab === 'cad'
                    ? 'bg-white text-stone-900 shadow-2xs'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                <Compass className="w-3.5 h-3.5 text-amber-700" />
                <span>Native 3D CAD Explorer</span>
              </button>
            </div>

            <a
              href={resolved.sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-stone-500 hover:text-stone-800 hover:bg-stone-200 transition-colors"
              title="Open Source Model in New Tab"
            >
              <ExternalLink className="w-4 h-4" />
            </a>

            <button
              onClick={onClose}
              className="p-2 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-200 transition-colors"
              title="Close 3D Viewport"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Viewport Area */}
        <div className="flex-1 bg-stone-950 relative overflow-hidden flex items-center justify-center">
          {activeTab === 'sketchfab' ? (
            /* Sketchfab Embed Viewport */
            <div className="w-full h-full relative">
              {iframeLoading && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-stone-950/90 text-white space-y-3">
                  <Loader2 className="w-8 h-8 animate-spin text-[#C85A32]" />
                  <p className="text-xs text-stone-400 font-medium">
                    Initializing Sketchfab 3D WebGL runtime...
                  </p>
                </div>
              )}

              {iframeError && (
                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-stone-900 p-6 text-center space-y-3">
                  <AlertCircle className="w-10 h-10 text-amber-500" />
                  <h4 className="font-semibold text-sm text-stone-200">
                    Network issue connecting to Sketchfab
                  </h4>
                  <p className="text-xs text-stone-400 max-w-sm">
                    External 3D embeds may be blocked by your network or adblocker. You can explore the monument in the Native 3D CAD Explorer.
                  </p>
                  <button
                    onClick={() => setActiveTab('cad')}
                    className="px-4 py-2 rounded-lg text-xs font-semibold bg-[#C85A32] text-white hover:bg-[#b04d28] transition-colors"
                  >
                    Switch to Native 3D CAD Explorer →
                  </button>
                </div>
              )}

              <iframe
                title={`3D Model of ${monumentName}`}
                src={resolved.embedUrl}
                className="w-full h-full border-0"
                allow="autoplay; fullscreen; xr-spatial-tracking; execution-while-out-of-viewport; execution-while-not-rendered; web-share"
                allowFullScreen
                onLoad={() => setIframeLoading(false)}
                onError={() => {
                  setIframeLoading(false);
                  setIframeError(true);
                }}
              />
            </div>
          ) : (
            /* Native 3D CAD Interactive Canvas Viewport */
            <div 
              className="w-full h-full relative cursor-grab active:cursor-grabbing select-none"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
            >
              <canvas
                ref={canvasRef}
                width={1024}
                height={680}
                className="w-full h-full block"
              />

              {/* HUD Overlay Info */}
              <div className="absolute top-4 left-4 pointer-events-none space-y-1">
                <div className="px-3 py-1.5 rounded-lg bg-stone-900/80 backdrop-blur-sm border border-stone-800 text-xs text-stone-200 space-y-0.5">
                  <p className="font-semibold text-amber-400">
                    {VERIFIED_MODELS[monumentKey]?.style || 'Architectural View'}
                  </p>
                  <p className="text-[11px] text-stone-400">
                    Elevation: {VERIFIED_MODELS[monumentKey]?.height || 'Monument Scale'}
                  </p>
                </div>
              </div>

              {/* HUD Controls */}
              <div className="absolute bottom-4 right-4 flex items-center space-x-2 bg-stone-900/85 backdrop-blur-sm border border-stone-800 rounded-xl p-1.5 text-xs text-stone-300">
                <button
                  onClick={() => setAutoRotate(!autoRotate)}
                  className={`p-2 rounded-lg transition-colors ${
                    autoRotate ? 'bg-amber-600/30 text-amber-300 border border-amber-500/40' : 'hover:bg-stone-800'
                  }`}
                  title="Toggle Auto Rotation"
                >
                  <RotateCw className={`w-3.5 h-3.5 ${autoRotate ? 'animate-spin' : ''}`} style={{ animationDuration: '6s' }} />
                </button>
                <button
                  onClick={() => setWireframe(!wireframe)}
                  className={`p-2 rounded-lg transition-colors ${
                    wireframe ? 'bg-amber-600/30 text-amber-300 border border-amber-500/40' : 'hover:bg-stone-800'
                  }`}
                  title="Toggle Wireframe CAD Mesh"
                >
                  <Layers className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoom(prev => Math.min(2.2, prev + 0.15))}
                  className="p-2 rounded-lg hover:bg-stone-800 transition-colors"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoom(prev => Math.max(0.5, prev - 0.15))}
                  className="p-2 rounded-lg hover:bg-stone-800 transition-colors"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => { setRotation({ yaw: 0.6, pitch: 0.35 }); setZoom(1.0); }}
                  className="p-2 rounded-lg hover:bg-stone-800 transition-colors"
                  title="Reset Camera View"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer Bar */}
        <div className="px-5 py-3 border-t border-[#E5DFD7] bg-[#FAF8F5] flex flex-wrap items-center justify-between gap-3 text-xs text-stone-600">
          <div className="flex items-center space-x-2">
            <span className="font-medium text-stone-800">Navigation:</span>
            <span>Click & drag to rotate (Yaw/Pitch) • Scroll wheel to zoom • Switch between photorealistic & CAD view</span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setActiveTab(activeTab === 'sketchfab' ? 'cad' : 'sketchfab')}
              className="text-xs text-[#C85A32] hover:underline font-medium"
            >
              Switch to {activeTab === 'sketchfab' ? 'Native 3D CAD' : 'Sketchfab Photorealistic'} View →
            </button>
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-stone-200 hover:bg-stone-300 text-stone-800 transition-colors"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// =========================================================================
// Math Projection Utilities
// =========================================================================

function project(
  pt: Point3D,
  rot: { yaw: number; pitch: number },
  zoom: number,
  W: number,
  H: number
): { x: number; y: number } | null {
  const res = projectWithDepth(pt, rot, zoom, W, H);
  return res ? { x: res.x, y: res.y } : null;
}

function projectWithDepth(
  pt: Point3D,
  rot: { yaw: number; pitch: number },
  zoom: number,
  W: number,
  H: number
): { x: number; y: number; depth: number } | null {
  const cosY = Math.cos(rot.yaw);
  const sinY = Math.sin(rot.yaw);
  const cosP = Math.cos(rot.pitch);
  const sinP = Math.sin(rot.pitch);

  // Rotate around Y-axis (Yaw)
  const x1 = pt.x * cosY + pt.z * sinY;
  const z1 = -pt.x * sinY + pt.z * cosY;

  // Rotate around X-axis (Pitch)
  const y1 = pt.y * cosP - z1 * sinP;
  const z2 = pt.y * sinP + z1 * cosP + 350; // Camera distance

  if (z2 <= 20) return null; // Behind camera

  const fov = 380 * zoom;
  const s = fov / z2;
  const screenX = x1 * s + W / 2;
  const screenY = -y1 * s + H / 2 + 20;

  return { x: screenX, y: screenY, depth: z2 };
}
