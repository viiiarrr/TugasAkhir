/**
 * processor.js — Port algoritma Python ke JavaScript
 * Algoritma: EMA, Outlier Filter, Split-and-Merge, RANSAC, Phantom Detection
 */

const DEFAULTS = {
  ema_alpha:       0.25,
  max_dist:        250.0,
  min_dist:        2.0,
  outlier_sigma:   2.0,
  outlier_window:  10,
  min_count:       4,
  split_threshold: 8.0,
  min_segment_pts: 6,
  ransac_iter:     60,
  ransac_inlier_thr: 6.0,
  phantom_dist_thr:  8.0,
  num_sensor:      8,
  sensor_step:     45.0,
  snapshot_interval: 30,  // ambil snapshot setiap N paket
};

class SensorProcessor {
  constructor(params = {}) {
    this.p = { ...DEFAULTS, ...params };
    this.rows = [];
    this.snapshots = [];
    this.reset();
  }

  // ────────────────────────────────────────────
  //  STATE MANAGEMENT
  // ────────────────────────────────────────────

  reset() {
    this.stableMap = new Float64Array(360);
    this.countMap  = new Int32Array(360);
    this.histMap   = Array.from({ length: 360 }, () => []);
    this.currentYaw  = 0;
    this.packetCount = 0;
  }

  _takeSnapshot(rowIndex) {
    return {
      rowIndex,
      stableMap: new Float64Array(this.stableMap),
      countMap:  new Int32Array(this.countMap),
      histMap:   this.histMap.map(h => h.slice()),
      currentYaw:  this.currentYaw,
      packetCount: this.packetCount,
    };
  }

  _loadSnapshot(snap) {
    this.stableMap   = new Float64Array(snap.stableMap);
    this.countMap    = new Int32Array(snap.countMap);
    this.histMap     = snap.histMap.map(h => h.slice());
    this.currentYaw  = snap.currentYaw;
    this.packetCount = snap.packetCount;
  }

  // ────────────────────────────────────────────
  //  CSV PARSING
  // ────────────────────────────────────────────

  parseCSV(text) {
    const lines = text.replace(/\r/g, '').trim().split('\n');
    const rows  = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',');
      if (cols.length < 10) continue;
      const yaw  = parseFloat(cols[1]);
      const dists = cols.slice(2, 10).map(Number);
      if (isNaN(yaw)) continue;
      rows.push({ time: cols[0].trim(), yaw, distances: dists });
    }
    return rows;
  }

  /**
   * Load CSV text: parse → pre-process semua baris → ambil snapshot berkala.
   * Setelah selesai, state = snapshot terakhir (full data).
   */
  loadCSV(text) {
    this.rows      = this.parseCSV(text);
    this.snapshots = [];
    this.reset();

    const N = this.rows.length;
    const interval = this.p.snapshot_interval;

    // Snapshot awal (row -1, state kosong)
    this.snapshots.push(this._takeSnapshot(-1));

    for (let i = 0; i < N; i++) {
      this.processRow(this.rows[i]);
      if ((i + 1) % interval === 0 || i === N - 1) {
        this.snapshots.push(this._takeSnapshot(i));
      }
    }
  }

  /**
   * Seek ke rowIndex: cari snapshot terdekat ≤ rowIndex, load, lanjut proses.
   */
  seekTo(rowIndex) {
    rowIndex = Math.max(0, Math.min(rowIndex, this.rows.length - 1));

    // Cari snapshot terbesar ≤ rowIndex
    let snapIdx = 0;
    for (let i = 0; i < this.snapshots.length; i++) {
      if (this.snapshots[i].rowIndex <= rowIndex) snapIdx = i;
      else break;
    }

    this._loadSnapshot(this.snapshots[snapIdx]);

    // Proses sisa baris dari snapshot ke rowIndex
    const start = this.snapshots[snapIdx].rowIndex + 1;
    for (let i = start; i <= rowIndex; i++) {
      this.processRow(this.rows[i]);
    }
  }

  // ────────────────────────────────────────────
  //  CORE: PROSES SATU BARIS DATA
  // ────────────────────────────────────────────

  processRow(row) {
    this.currentYaw = row.yaw;
    this.packetCount++;
    const p = this.p;

    for (let i = 0; i < p.num_sensor; i++) {
      const dist = row.distances[i];
      if (isNaN(dist) || dist < p.min_dist || dist > p.max_dist) continue;

      // Rumus kunci dari Python: physical_deg = (-yaw + i*sensor_step) % 360
      const physDeg = ((-row.yaw + i * p.sensor_step) % 360 + 360) % 360;
      const idx     = Math.floor(physDeg) % 360;

      // Outlier filter (median + sigma)
      const hist = this.histMap[idx];
      if (hist.length >= p.outlier_window) {
        const sorted = hist.slice().sort((a, b) => a - b);
        const med    = sorted[Math.floor(sorted.length / 2)];
        const mean   = hist.reduce((s, x) => s + x, 0) / hist.length;
        const std    = Math.sqrt(hist.reduce((s, x) => s + (x - mean) ** 2, 0) / hist.length);
        if (std > 0 && Math.abs(dist - med) > p.outlier_sigma * std) continue;
      }

      hist.push(dist);
      if (hist.length > p.outlier_window) hist.shift();

      // EMA update
      if (this.stableMap[idx] === 0) {
        this.stableMap[idx] = dist;
      } else {
        this.stableMap[idx] = (1 - p.ema_alpha) * this.stableMap[idx] + p.ema_alpha * dist;
      }
      this.countMap[idx]++;
    }
  }

  // ────────────────────────────────────────────
  //  BANGUN TITIK DARI STABLE MAP
  // ────────────────────────────────────────────

  getMapPoints() {
    const sx = [], sy = [], rawX = [], rawY = [];
    for (let i = 0; i < 360; i++) {
      const d = this.stableMap[i];
      if (d > 0) {
        const rad = i * Math.PI / 180;
        const x   = d * Math.cos(rad);
        const y   = d * Math.sin(rad);
        rawX.push(x); rawY.push(y);
        if (this.countMap[i] >= this.p.min_count) {
          sx.push(x); sy.push(y);
        }
      }
    }
    return { sx, sy, rawX, rawY };
  }

  getStats() {
    const filled = 0 | this.stableMap.reduce((s, v) => s + (v > 0 ? 1 : 0), 0);
    const stable = 0 | [...this.countMap].filter(v => v >= this.p.min_count).length;
    return {
      filled, stable, total: 360,
      packets: this.packetCount,
      yaw: this.currentYaw,
    };
  }

  // ────────────────────────────────────────────
  //  WALL FITTING & PHANTOM DETECTION
  // ────────────────────────────────────────────

  _ptLineDist(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return Math.hypot(px - x1, py - y1);
    const t = ((px - x1) * dx + (py - y1) * dy) / len2;
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  _splitAndMerge(ptsIdx, points, depth = 0) {
    const p = this.p;
    if (ptsIdx.length < 2)
      return ptsIdx.length >= p.min_segment_pts ? [ptsIdx] : [];

    const seg = ptsIdx.map(i => points[i]);
    const [x1, y1] = seg[0];
    const [x2, y2] = seg[seg.length - 1];

    let maxD = 0, maxI = 0;
    for (let j = 0; j < seg.length; j++) {
      const d = this._ptLineDist(seg[j][0], seg[j][1], x1, y1, x2, y2);
      if (d > maxD) { maxD = d; maxI = j; }
    }

    if (maxD > p.split_threshold && depth < 12) {
      const left  = this._splitAndMerge(ptsIdx.slice(0, maxI + 1), points, depth + 1);
      const right = this._splitAndMerge(ptsIdx.slice(maxI),         points, depth + 1);
      return [...left, ...right];
    }
    return ptsIdx.length >= p.min_segment_pts ? [ptsIdx] : [];
  }

  _ransacLine(pts) {
    const p = this.p;
    if (pts.length < 2) return null;

    let bestInliers = null, bestCount = 0;

    for (let iter = 0; iter < p.ransac_iter; iter++) {
      let i = Math.floor(Math.random() * pts.length);
      let j = Math.floor(Math.random() * (pts.length - 1));
      if (j >= i) j++;

      const [x1, y1] = pts[i], [x2, y2] = pts[j];
      const dx = x2 - x1, dy = y2 - y1;
      if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) continue;

      const a = dy, b = -dx, c = dx * y1 - dy * x1;
      const norm = Math.hypot(a, b);

      const inliers = pts.map(pp => Math.abs(a * pp[0] + b * pp[1] + c) / norm < p.ransac_inlier_thr);
      const cnt = inliers.filter(Boolean).length;
      if (cnt > bestCount) { bestCount = cnt; bestInliers = inliers; }
    }

    if (!bestInliers || bestCount < 2) return null;

    // Re-fit dengan inlier via SVD 2×2
    const inPts = pts.filter((_, i) => bestInliers[i]);
    const cx = inPts.reduce((s, pp) => s + pp[0], 0) / inPts.length;
    const cy = inPts.reduce((s, pp) => s + pp[1], 0) / inPts.length;
    let sxx = 0, sxy = 0, syy = 0;
    for (const [x, y] of inPts) {
      sxx += (x - cx) ** 2; sxy += (x - cx) * (y - cy); syy += (y - cy) ** 2;
    }
    const tr   = sxx + syy;
    const det  = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
    const lam  = tr / 2 + disc;
    let dx2, dy2;
    if (Math.abs(sxy) > 1e-9) { dx2 = lam - syy; dy2 = sxy; }
    else { dx2 = sxx >= syy ? 1 : 0; dy2 = sxx >= syy ? 0 : 1; }

    const nAb = Math.hypot(dy2, -dx2);
    if (nAb === 0) return null;
    const a = dy2 / nAb, b = -dx2 / nAb;
    const c = -(a * cx + b * cy);
    return { line: [a, b, c] };
  }

  /**
   * Deteksi dinding & phantom point dari titik stabil.
   * @returns { wallSegs, inlierMask, phantomMask }
   */
  detectWalls(sx, sy) {
    const n = sx.length;
    const p = this.p;

    if (n < p.min_segment_pts * 2) {
      return {
        wallSegs:    [],
        inlierMask:  new Array(n).fill(true),
        phantomMask: new Array(n).fill(false),
      };
    }

    const pts    = sx.map((x, i) => [x, sy[i]]);
    const angles = pts.map(pp => Math.atan2(pp[1], pp[0]));
    const order  = angles.map((_, i) => i).sort((a, b) => angles[a] - angles[b]);
    const sorted = order.map(i => pts[i]);

    const allIdx   = Array.from({ length: sorted.length }, (_, i) => i);
    const segments = this._splitAndMerge(allIdx, sorted);

    const wallLines = [], wallSegs = [];
    const segmentsData = [];
    for (const segIdx of segments) {
      if (segIdx.length < 2) continue;
      const segPts = segIdx.map(i => sorted[i]);
      const res    = this._ransacLine(segPts);
      if (!res) continue;
      const [a, b, c] = res.line;
      wallLines.push(res.line);
      
      // Proyeksikan titik awal dan akhir ke garis RANSAC agar lurus sempurna
      const [px1, py1] = segPts[0];
      const [px2, py2] = segPts[segPts.length - 1];
      
      const dist1 = a * px1 + b * py1 + c;
      const x1 = px1 - a * dist1;
      const y1 = py1 - b * dist1;
      
      const dist2 = a * px2 + b * py2 + c;
      const x2 = px2 - a * dist2;
      const y2 = py2 - b * dist2;

      segmentsData.push({
        line: res.line,
        p1: [x1, y1],
        p2: [x2, y2]
      });
    }

    // Sambungkan ujung-ujung segmen yang berdekatan untuk membentuk sudut kotak yang rapi
    const N_segs = segmentsData.length;
    if (N_segs > 1) {
      for (let i = 0; i < N_segs; i++) {
        const seg1 = segmentsData[i];
        const seg2 = segmentsData[(i + 1) % N_segs];
        
        const [a1, b1, c1] = seg1.line;
        const [a2, b2, c2] = seg2.line;
        
        const det = a1 * b2 - a2 * b1;
        // Jika garis tidak sejajar (det tidak mendekati 0)
        if (Math.abs(det) > 0.1) {
          const ix = (b1 * c2 - b2 * c1) / det;
          const iy = (a2 * c1 - a1 * c2) / det;
          
          // Cek jarak perpotongan dari ujung segmen
          const distToP2 = Math.hypot(ix - seg1.p2[0], iy - seg1.p2[1]);
          const distToP1 = Math.hypot(ix - seg2.p1[0], iy - seg2.p1[1]);
          
          // Jika perpotongan tidak terlalu jauh, gabungkan (membentuk sudut tajam)
          if (distToP2 < 80 && distToP1 < 80) {
            seg1.p2 = [ix, iy];
            seg2.p1 = [ix, iy];
          }
        }
      }
    }

    for (const seg of segmentsData) {
      wallSegs.push([seg.p1[0], seg.p1[1], seg.p2[0], seg.p2[1]]);
    }

    // Phantom: titik yang jauh dari SEMUA wall line
    const isPhantom = new Array(n).fill(true);
    for (let gi = 0; gi < n; gi++) {
      const px = sx[gi], py = sy[gi];
      for (const [a, b, c] of wallLines) {
        if (Math.abs(a * px + b * py + c) <= p.phantom_dist_thr) {
          isPhantom[gi] = false;
          break;
        }
      }
    }

    return {
      wallSegs,
      inlierMask:  isPhantom.map(v => !v),
      phantomMask: isPhantom,
    };
  }
}
