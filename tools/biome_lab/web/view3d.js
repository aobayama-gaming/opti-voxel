// The studio's 3D view: a climate report's ground heights (heights.f32: n x n float32 metres, row-major, x right, z
// down, spacing_m apart) as a mesh, coloured per point by the caller. Drawn on demand (orbit, resize, new data).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SKY = new THREE.Color("#b4c8d8");

// The engine's slope blend of a ground's colours (page_shading_glsl.hpp ground_flat_weight).
export function flat_weight(p_up) {
    const t = Math.min(1, Math.max(0, (p_up - 0.8) / 0.15));
    return t * t * (3 - 2 * t);
}

export class View3D {
    constructor(p_canvas) {
        this.canvas = p_canvas;
        this.renderer = new THREE.WebGLRenderer({ canvas: p_canvas, antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.scene = new THREE.Scene();
        this.scene.background = SKY;
        this.camera = new THREE.PerspectiveCamera(45, 1, 1, 1e6);
        this.controls = new OrbitControls(this.camera, p_canvas);
        this.controls.maxPolarAngle = Math.PI * 0.49;
        this.controls.addEventListener("change", () => this.draw());
        const sun = new THREE.DirectionalLight(0xffffff, 2.4);
        sun.position.set(-0.6, 1, -0.4);
        this.scene.add(sun, new THREE.HemisphereLight(0xd6e4ff, 0x50483c, 1.1));
        this.water = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
            new THREE.MeshLambertMaterial({ color: 0x2e5f80, transparent: true, opacity: 0.7 }));
        this.water.visible = false;
        this.scene.add(this.water);
        this.mesh = null;
        this.exaggeration = 1;
        new ResizeObserver(() => this.resize()).observe(p_canvas.parentElement);
    }

    resize() {
        const box = this.canvas.parentElement;
        this.renderer.setSize(box.clientWidth, box.clientHeight, false);
        this.camera.aspect = box.clientWidth / Math.max(1, box.clientHeight);
        this.camera.updateProjectionMatrix();
        this.draw();
    }

    draw() { this.renderer.render(this.scene, this.camera); }

    // The heights as a mesh; p_color(i, up) the linear RGB of point i (up: its normal's y, the slope). p_frame: put the
    // camera back over the whole slice (a new place or size), else it stays where the user left it (an edit).
    set(p_h, p_n, p_spacing, p_color, p_frame) {
        const half = (p_n - 1) / 2;
        const pos = new Float32Array(p_n * p_n * 3);
        let lo = Infinity, hi = -Infinity, sum = 0;
        for (let j = 0; j < p_n; j++) {
            for (let i = 0; i < p_n; i++) {
                const k = j * p_n + i, h = p_h[k];
                pos.set([(i - half) * p_spacing, h, (j - half) * p_spacing], k * 3);
                lo = Math.min(lo, h); hi = Math.max(hi, h); sum += h;
            }
        }
        const index = [];
        for (let j = 0; j + 1 < p_n; j++) {
            for (let i = 0; i + 1 < p_n; i++) {
                const a = j * p_n + i, b = a + 1, c = a + p_n, d = c + 1;
                index.push(a, c, b, b, c, d);
            }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        geo.setIndex(index);
        geo.computeVertexNormals();
        const normal = geo.getAttribute("normal");
        const col = new Float32Array(p_n * p_n * 3);
        for (let k = 0; k < p_n * p_n; k++) {
            const c = p_color(k, normal.getY(k));
            col.set([c.r, c.g, c.b], k * 3);
        }
        geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.mesh.geometry = geo;
        } else {
            this.mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
            this.scene.add(this.mesh);
        }
        this.mesh.scale.y = this.exaggeration;
        this.mesh.userData = { n: p_n, spacing: p_spacing };
        this.stale(false);
        const span = (p_n - 1) * p_spacing;
        this.water.scale.set(span, 1, span);
        this.scene.fog = new THREE.Fog(SKY, span * 1.2, span * 4);
        this.camera.near = span / 5000;
        this.camera.far = span * 10;
        this.camera.updateProjectionMatrix();
        if (p_frame) {
            const mid = (sum / (p_n * p_n)) * this.exaggeration;
            this.controls.target.set(0, mid, 0);
            this.camera.position.set(0, Math.max(mid + span * 0.5, hi * this.exaggeration + span * 0.15), span * 0.85);
            this.controls.update();
        }
        this.draw();
        return { lo, hi, mean: sum / (p_n * p_n), span };
    }

    set_exaggeration(p_x) {
        this.exaggeration = p_x;
        if (this.mesh) this.mesh.scale.y = p_x;
        this.draw();
    }

    show_water(p_on) { this.water.visible = p_on; this.draw(); }

    // The last mesh greyed: it is no longer the table's (refused, or a run on its way).
    stale(p_on) {
        if (!this.mesh) return;
        this.mesh.material.color.set(p_on ? 0x777777 : 0xffffff);
        this.draw();
    }

    // The slice point under a pointer event, {i, j, x, z} (x, z: metres from the slice's centre), or null.
    pick(p_event) {
        if (!this.mesh) return null;
        const r = this.canvas.getBoundingClientRect();
        const ndc = new THREE.Vector2(((p_event.clientX - r.left) / r.width) * 2 - 1, -((p_event.clientY - r.top) / r.height) * 2 + 1);
        const ray = new THREE.Raycaster();
        ray.setFromCamera(ndc, this.camera);
        const hit = ray.intersectObject(this.mesh)[0];
        if (!hit) return null;
        const { n, spacing } = this.mesh.userData;
        const half = (n - 1) / 2;
        const i = Math.round(hit.point.x / spacing + half), j = Math.round(hit.point.z / spacing + half);
        return { i: Math.min(n - 1, Math.max(0, i)), j: Math.min(n - 1, Math.max(0, j)), x: hit.point.x, z: hit.point.z };
    }
}
