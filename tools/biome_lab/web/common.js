// What the lab's two pages share: DOM and server helpers, and the hand-off of an edited table between them.

export const $ = (s) => document.querySelector(s);

export function el(p_tag, p_attrs = {}, ...p_children) {
    const e = document.createElement(p_tag);
    for (const [k, v] of Object.entries(p_attrs)) {
        if (v === undefined || v === null || v === false) continue;
        if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
        else if (k === "class") e.className = v;
        else e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of p_children.flat()) {
        if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return e;
}

export async function api(p_path, p_body) {
    const r = await fetch(p_path, p_body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p_body) });
    const v = await r.json();
    if (!r.ok) throw new Error(v.error || r.statusText);
    return v;
}

export function toast(p_text, p_bad = false) {
    const t = $("#toast");
    t.textContent = p_text;
    t.className = p_bad ? "show bad" : "show";
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { t.className = ""; }, p_bad ? 6000 : 2500);
}

// The other page opens on the table as it is here, unsaved edits included (its saved text says what is unsaved).
const HANDOFF = "biome_lab_handoff";

let handed = false;
export const handed_off = () => handed; // the page is leaving with its table: no "unsaved changes" question

export function go_to(p_page, p_path, p_data, p_saved) {
    handed = true;
    try { sessionStorage.setItem(HANDOFF, JSON.stringify({ path: p_path, data: p_data, saved: p_saved })); } catch { /* private window */ }
    location.href = p_page;
}

export function take_handoff() {
    try {
        const h = JSON.parse(sessionStorage.getItem(HANDOFF) || "null");
        sessionStorage.removeItem(HANDOFF);
        return h;
    } catch { return null; }
}
