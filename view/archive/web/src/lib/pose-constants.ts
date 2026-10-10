// 133-keypoint skeleton definitions from RTMPose Wholebody
// Full RTMPose layout: body(0-16) + feet(17-22) + face(23-90) + left_hand(91-111) + right_hand(112-132)
// ST-GCN uses a subset: body(9) + left_hand(21) + right_hand(21) + face(18) = 69 core points

export interface KeypointDef {
  id: number;
  name: string;
  group: "body" | "left_hand" | "right_hand" | "face";
  connections: number[]; // IDs this point connects to
}

export interface BoneConn {
  from: number;
  to: number;
  group: "body" | "left_hand" | "right_hand" | "face";
}

const BODY_BONES: [number, number][] = [
  // Face connections
  [0, 1], [0, 2], [1, 3], [2, 4], // nose → eyes → ears
  // Upper body
  [5, 6], // shoulders
  [5, 7], [7, 9], // left arm
  [6, 8], [8, 10], // right arm
  // Lower body
  [5, 11], [6, 12], // shoulders → hips
  [11, 12], // hips
  [11, 13], [13, 15], // left leg
  [12, 14], [14, 16], // right leg
];

const HAND_BONES: [number, number][] = [
  // Wrist to fingers
  [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
  // Thumb
  [1, 2], [2, 3], [3, 4],
  // Index
  [5, 6], [6, 7], [7, 8],
  // Middle
  [9, 10], [10, 11], [11, 12],
  // Ring
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [17, 18], [18, 19], [19, 20],
];

const FACE_BONES: [number, number][] = [
  // Face contour (oval)
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
  [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
  // Eyebrows
  [17, 18], [18, 19], [19, 20], [20, 21],
  [22, 23], [23, 24], [24, 25], [25, 26],
  // Nose
  [27, 28], [28, 29], [29, 30], [27, 31], [31, 32], [32, 33], [27, 34], [34, 35],
  // Eyes
  [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
  [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
  // Mouth
  [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
  [48, 54], [48, 55], [54, 55],
  [55, 56], [56, 57], [57, 58],
  [55, 59], [59, 60], [60, 61],
  [48, 62], [62, 63], [63, 64],
  [54, 65], [65, 66], [66, 67],
];

export const BODY_PART_COLORS: Record<string, { stroke: string; fill: string; glow: string; label: string }> = {
  body: { stroke: "#22d3ee", fill: "#06b6d4", glow: "shadow-cyan-400/30", label: "身体躯干" },
  left_hand: { stroke: "#c084fc", fill: "#a855f7", glow: "shadow-purple-400/30", label: "左手" },
  right_hand: { stroke: "#f472b6", fill: "#ec4899", glow: "shadow-pink-400/30", label: "右手" },
  face: { stroke: "#fbbf24", fill: "#f59e0b", glow: "shadow-amber-400/30", label: "面部" },
};

// Generate all bone connections with group info
export function getBodyBones(): BoneConn[] {
  return BODY_BONES.map(([from, to]) => ({ from, to, group: "body" as const }));
}

export function getLeftHandBones(): BoneConn[] {
  return HAND_BONES.map(([from, to]) => ({ from: from + 91, to: to + 91, group: "left_hand" as const }));
}

export function getRightHandBones(): BoneConn[] {
  return HAND_BONES.map(([from, to]) => ({ from: from + 112, to: to + 112, group: "right_hand" as const }));
}

export function getFaceBones(): BoneConn[] {
  return FACE_BONES.map(([from, to]) => ({ from: from + 23, to: to + 23, group: "face" as const }));
}

// Feet bones (indices 17-22): left big_toe, left small_toe, left heel, right big_toe, right small_toe, right heel
const FEET_BONES: [number, number][] = [
  [15, 17], [17, 18], [15, 19],
  [16, 20], [20, 21], [16, 22],
];

export function getFeetBones(): BoneConn[] {
  return FEET_BONES.map(([from, to]) => ({ from, to, group: "body" as const }));
}

export function getAllBones(): BoneConn[] {
  return [...getBodyBones(), ...getFeetBones(), ...getFaceBones(), ...getLeftHandBones(), ...getRightHandBones()];
}

// Body keypoint names
export const BODY_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];

// Generate a demo pose frame with RTMPose Wholebody 133-keypoint layout:
// body(0-16) + feet(17-22) + face(23-90) + left_hand(91-111) + right_hand(112-132)
export function generateDemoPoseFrame(
  frameIndex: number,
  totalFrames: number,
): { x: number; y: number; confidence: number }[] {
  const t = frameIndex / Math.max(totalFrames - 1, 1);
  const kp: { x: number; y: number; confidence: number }[] = [];
  const phase = t * Math.PI * 2;

  const bodyBaseY = 0.15;
  const bodySway = Math.sin(phase * 0.7) * 0.03;

  // ── Body keypoints (0-16): COCO-17 ──
  kp.push({ x: 0.5 + bodySway, y: bodyBaseY, confidence: 0.95 });           // 0  nose
  kp.push({ x: 0.48 + bodySway, y: bodyBaseY - 0.02, confidence: 0.92 });    // 1  left_eye
  kp.push({ x: 0.52 + bodySway, y: bodyBaseY - 0.02, confidence: 0.93 });    // 2  right_eye
  kp.push({ x: 0.46 + bodySway, y: bodyBaseY, confidence: 0.88 });           // 3  left_ear
  kp.push({ x: 0.54 + bodySway, y: bodyBaseY, confidence: 0.87 });           // 4  right_ear
  kp.push({ x: 0.42 + bodySway, y: bodyBaseY + 0.08, confidence: 0.96 });    // 5  left_shoulder
  kp.push({ x: 0.58 + bodySway, y: bodyBaseY + 0.08, confidence: 0.97 });    // 6  right_shoulder
  kp.push({ x: 0.35 + bodySway + Math.sin(phase * 1.3) * 0.04, y: bodyBaseY + 0.17, confidence: 0.9 }); // 7  left_elbow
  kp.push({ x: 0.65 + bodySway + Math.cos(phase * 1.5) * 0.05, y: bodyBaseY + 0.17, confidence: 0.91 });// 8  right_elbow
  kp.push({ x: 0.30 + bodySway + Math.sin(phase * 1.3) * 0.07, y: bodyBaseY + 0.25, confidence: 0.85 });// 9  left_wrist
  kp.push({ x: 0.70 + bodySway + Math.cos(phase * 1.5) * 0.08, y: bodyBaseY + 0.25, confidence: 0.86 });// 10 right_wrist
  kp.push({ x: 0.44 + bodySway, y: bodyBaseY + 0.22, confidence: 0.94 });   // 11 left_hip
  kp.push({ x: 0.56 + bodySway, y: bodyBaseY + 0.22, confidence: 0.95 });   // 12 right_hip
  kp.push({ x: 0.42 + bodySway, y: bodyBaseY + 0.38, confidence: 0.92 });   // 13 left_knee
  kp.push({ x: 0.58 + bodySway, y: bodyBaseY + 0.38, confidence: 0.93 });   // 14 right_knee
  kp.push({ x: 0.40 + bodySway, y: bodyBaseY + 0.52, confidence: 0.89 });   // 15 left_ankle
  kp.push({ x: 0.60 + bodySway, y: bodyBaseY + 0.52, confidence: 0.88 });   // 16 right_ankle

  // ── Feet keypoints (17-22): 6 points ──
  const la = kp[15]; const ra = kp[16];
  kp.push({ x: la.x - 0.01, y: la.y + 0.03, confidence: 0.82 });  // 17 left_big_toe
  kp.push({ x: la.x + 0.01, y: la.y + 0.03, confidence: 0.80 });  // 18 left_small_toe
  kp.push({ x: la.x, y: la.y + 0.05, confidence: 0.78 });          // 19 left_heel
  kp.push({ x: ra.x - 0.01, y: ra.y + 0.03, confidence: 0.82 });  // 20 right_big_toe
  kp.push({ x: ra.x + 0.01, y: ra.y + 0.03, confidence: 0.80 });  // 21 right_small_toe
  kp.push({ x: ra.x, y: ra.y + 0.05, confidence: 0.78 });          // 22 right_heel

  // ── Face keypoints (23-90): 68 points ──
  const fx = 0.5 + bodySway;
  const fy = bodyBaseY - 0.01;
  for (let i = 0; i < 68; i++) {
    const angle = (i / 68) * Math.PI * 2;
    const r = 0.04 + (i < 17 ? 0.03 : 0.005);
    kp.push({
      x: fx + Math.cos(angle) * r * (i < 17 ? 1 : 0.4),
      y: fy + Math.sin(angle) * r,
      confidence: 0.65 + Math.random() * 0.25,
    });
  }

  // ── Left hand keypoints (91-111): 21 points ──
  const lw = kp[9];
  const lhPhase = phase * 1.8;
  for (let i = 0; i < 21; i++) {
    const fingerIdx = Math.floor(i / 4);
    const angle = -0.5 + fingerIdx * 0.25 + Math.sin(lhPhase + fingerIdx * 0.5) * 0.08;
    const dist = (i === 0 ? 0 : 0.03 + (i % 5) * 0.025);
    kp.push({ x: lw.x + Math.cos(angle) * dist, y: lw.y + Math.sin(angle) * dist - 0.01, confidence: 0.7 + Math.random() * 0.2 });
  }

  // ── Right hand keypoints (112-132): 21 points ──
  const rw = kp[10];
  const rhPhase = phase * 2.0;
  for (let i = 0; i < 21; i++) {
    const fingerIdx = Math.floor(i / 4);
    const angle = Math.PI - 0.5 + fingerIdx * 0.25 + Math.cos(rhPhase + fingerIdx * 0.6) * 0.1;
    const dist = (i === 0 ? 0 : 0.03 + (i % 5) * 0.025);
    kp.push({ x: rw.x + Math.cos(angle) * dist, y: rw.y + Math.sin(angle) * dist - 0.01, confidence: 0.7 + Math.random() * 0.2 });
  }

  return kp;
}

export function getKeypointGroup(index: number): "body" | "left_hand" | "right_hand" | "face" {
  if (index < 23) return "body";   // COCO body(0-16) + feet(17-22)
  if (index < 91) return "face";   // face(23-90)
  if (index < 112) return "left_hand"; // left_hand(91-111)
  return "right_hand";             // right_hand(112-132)
}

export function generateMultiFramePose(
  numFrames: number,
): { x: number; y: number; confidence: number }[][] {
  return Array.from({ length: numFrames }, (_, i) => generateDemoPoseFrame(i, numFrames));
}
