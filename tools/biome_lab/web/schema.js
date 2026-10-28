// The climate table's JSON format as data, from the comment above parse() in src/climate_parse.cpp (the one source:
// change it there first, then here). The page only uses this to lay out its forms and offer keys; the engine
// (climate_report.sh) parses and validates, so a key the engine refuses still shows its own message. The server diffs
// these keys against the parser's keys() lists at startup and the page shows any drift (server.py, format_drift).
//
// Value kinds: text (multi-line string), string, number, int, bool, enum (values), color (#rrggbb), ref (a name of
// the table: "fields", "fields+ground", "regions"), refs (n names), numbers (n numbers), points ([[x, y], ...]),
// object (of a type), list (of a type), json (anything).

const SHAPE_KEYS = {
    shape: { kind: "enum", values: ["plain", "billow", "ridged", "square"],
        help: "plain a n + o; billow 2 |a n| - 1 + o; ridged (o - |a n|)^2 (o default 1); square (a n)^2 + o. Inserted without its mean." },
    scale: { kind: "number", help: "a: the noise's scale inside the shape." },
    offset: { kind: "number", help: "o: the shape's offset (ridged: default 1; plain / billow / square: needs a multifractal, else refused)." },
    coeffs: { kind: "numbers", n: 4, help: "[c1, cn, cabs, csq]: c1 + cn n + cabs |n| + csq n^2 (instead of shape / scale / offset)." },
};

// A band's, a stretch's or a multiplier's octaves: one wavelength, or from_m .. to_m.
const range_title = (v) => v.wavelength_m !== undefined ? `${v.wavelength_m} m` : `${v.from_m ?? ""} .. ${v.to_m ?? ""} m`;
const edge_template = () => ({ field: "", min: 0, fade: 0.1 });

const RANGE_KEYS = {
    wavelength_m: { kind: "number", metres: true, help: "One octave, rounded to 2^k voxels." },
    from_m: { kind: "number", metres: true, help: "The finest octave of a range (every octave from from_m to to_m that the ground has)." },
    to_m: { kind: "number", metres: true, help: "The coarsest octave of a range." },
};

export const TYPES = {
    table: {
        label: "Table",
        keys: {
            comment: { kind: "text", help: "Ignored by the engine: what the table is and how it was made." },
            fields: { kind: "list", of: "field", help: "Climate fields (at most 24 groups with the multifractal state): noise the biomes' regions read." },
            regions: { kind: "list", of: "region", help: "Named products of edges, for several landforms in one table." },
            heights: { kind: "list", of: "height", help: "Height terms: height_m * curve(field) added to the ground at each sample." },
            ground_stretch: { kind: "list", of: "stretch", help: "The ground's octaves in range read their noise squeezed along the base's up (keeps steep landforms a height field)." },
            biomes: { kind: "list", of: "biome", help: "At most 16. Priority layering: each biome takes its region of what the earlier ones left; the last is the default (no \"where\")." },
        },
        required: ["fields", "biomes"],
    },
    field: {
        label: "Field",
        title: (v) => v.name,
        keys: {
            name: { kind: "string", help: "Referenced by edges, multipliers, terms, heights and warps." },
            comment: { kind: "text" },
            seed: { kind: "int", help: "Optional, else a hash of the name. Octave k's seed is make_octave(seed, k)." },
            noise: { kind: "enum", values: ["perlin", "cell"], help: "cell: Worley F1 (plain bands only), read through a height term (dunes, cones)." },
            bands: { kind: "list", of: "band", help: "The field's octaves." },
            terms: { kind: "list", of: "term", help: "Linear terms after the cascade, in list order: coef * the ground or an earlier field." },
        },
        required: ["name"],
        template: () => ({ name: "new_field", bands: [{ wavelength_m: 5000, amp: 1 }] }),
    },
    band: {
        label: "Band",
        title: range_title,
        keys: { ...RANGE_KEYS,
            amp: { kind: "number", help: "Amplitude (of the coarsest octave for a range)." },
            persistence: { kind: "number", help: "A range's amplitude factor per finer octave (default 1)." },
            ...SHAPE_KEYS,
            stretch_y: { kind: "number", help: "A power of two: the band's noise squeezed along the base's up." },
            comment: { kind: "text" } },
        required: ["amp"],
        template: () => ({ wavelength_m: 5000, amp: 1 }),
    },
    term: {
        label: "Term",
        title: (v) => `${v.source} x ${v.per_m ?? v.x}`,
        keys: {
            source: { kind: "ref", of: "fields+ground", help: "ground: its height above the datum (per metre); a field: an earlier field's value." },
            per_m: { kind: "number", help: "The coefficient on the ground's height, per metre (a lapse)." },
            x: { kind: "number", help: "The coefficient on the earlier field." },
            comment: { kind: "text" },
        },
        required: ["source"],
        template: () => ({ source: "ground", per_m: -0.002 }),
    },
    stretch: {
        label: "Stretch",
        title: (v) => `${range_title(v)} x${v.stretch_y}`,
        keys: { ...RANGE_KEYS,
            stretch_y: { kind: "number", help: "A power of two; octave k takes at most 2^(k - 3). No octave stretched twice." },
            comment: { kind: "text" } },
        required: ["stretch_y"],
        template: () => ({ from_m: 100, to_m: 30000000, stretch_y: 4 }),
    },
    region: {
        label: "Region",
        title: (v) => v.name,
        keys: {
            name: { kind: "string", help: "Named by a biome's or a height's \"region\"." },
            where: { kind: "list", of: "edge", help: "Its edges (a product of smoothsteps)." },
            comment: { kind: "text" },
        },
        required: ["name", "where"],
        template: () => ({ name: "new_region", where: [edge_template()] }),
    },
    edge: {
        label: "Edge",
        title: (v) => `${v.min ?? "-inf"} < ${v.field} < ${v.max ?? "inf"}`,
        keys: {
            field: { kind: "ref", of: "fields" },
            min: { kind: "number", help: "1 above min (either or both of min / max)." },
            max: { kind: "number", help: "1 below max." },
            fade: { kind: "number", help: "Fades to 0 over this width outside (or fade_min / fade_max)." },
            fade_min: { kind: "number" },
            fade_max: { kind: "number" },
        },
        required: ["field"],
        template: edge_template,
    },
    height: {
        label: "Height",
        title: (v) => `${v.height_m} m x ${v.field}${v.region ? " in " + v.region : ""}`,
        keys: {
            field: { kind: "ref", of: "fields" },
            height_m: { kind: "number", help: "The term's height (under the ground's amplitude cap)." },
            curve: { kind: "points", help: "libnoise's Curve, x increasing; the field must reach it. Without one: height_m * field." },
            region: { kind: "ref", of: "regions", help: "Gates the term (needs a curve): 1 in the region's core, 0 outside its fades." },
            comment: { kind: "text" },
        },
        required: ["field", "height_m"],
        template: () => ({ field: "", height_m: 100 }),
    },
    biome: {
        label: "Biome",
        title: (v) => v.name,
        color: (v) => v.debug_color || v.surface?.flat,
        keys: {
            name: { kind: "string", help: "Its ground material is biome:<name>." },
            comment: { kind: "text" },
            region: { kind: "ref", of: "regions", help: "The region's edges come before the biome's own." },
            where: { kind: "list", of: "edge", help: "Where the biome is (none on the last, default biome)." },
            amp: { kind: "list", of: "amp", help: "Multipliers on the ground's and fields' octaves (octave k reads the weights of coarser octaves only)." },
            terrace: { kind: "object", of: "terrace", help: "libnoise's Terrace of the ground height, blended by the weight." },
            surface: { kind: "object", of: "surface", help: "The ground colours (sRGB) on gentle and steep slopes; without it, the procedural ground's." },
            debug_color: { kind: "color", help: "Its colour in the biome maps and --view=12." },
        },
        required: ["name"],
        template: () => ({ name: "new_biome", where: [edge_template()], amp: [{ field: "ground", from_m: 100, to_m: 5000, x: 1 }], surface: { flat: "#6a8a3c", steep: "#7a7468" }, debug_color: "#40a040" }),
    },
    amp: {
        label: "Multiplier",
        title: (v) => `${v.field} ${range_title(v)} x${v.x ?? 1}${v.shape ? " " + v.shape : ""}`,
        keys: {
            field: { kind: "ref", of: "fields+ground" },
            ...RANGE_KEYS,
            x: { kind: "number", help: "The multiplier (default 1). The fitter keeps it under the 3.3 km cap." },
            ...SHAPE_KEYS,
            multifractal: { kind: "number", help: "Gain (ground only): weighted by clamp(gain (1 - c)). Musgrave ridged: 2 + ridged; hybrid: 1 + plain + offset." },
            warp_m: { kind: "number", help: "Ground only: the noise read displaced by warp_m times the three warp fields." },
            warp_fields: { kind: "refs", n: 3, of: "fields", help: "Three fields (x, y, z of the warp)." },
            comment: { kind: "text" },
        },
        required: ["field"],
        template: () => ({ field: "ground", wavelength_m: 1638.4, x: 1 }),
    },
    terrace: {
        label: "Terrace",
        keys: {
            step_m: { kind: "number", help: "A terrace every step_m of height." },
            invert: { kind: "bool" },
        },
        required: ["step_m"],
        template: () => ({ step_m: 40 }),
    },
    surface: {
        label: "Surface",
        keys: {
            flat: { kind: "color", help: "Gentle slopes." },
            steep: { kind: "color", help: "Steep slopes (else flat)." },
            roughness: { kind: "number", help: "0..1 (default 1)." },
        },
        required: ["flat"],
        template: () => ({ flat: "#6a8a3c" }),
    },
};

// The table's sections in the sidebar, in the format's order.
export const SECTIONS = [
    { key: "biomes", type: "biome", label: "Biomes" },
    { key: "fields", type: "field", label: "Fields" },
    { key: "regions", type: "region", label: "Regions" },
    { key: "heights", type: "height", label: "Heights" },
    { key: "ground_stretch", type: "stretch", label: "Ground stretch" },
];

// A default value for a key of a type, when the user adds it.
export function default_value(p_spec) {
    switch (p_spec.kind) {
        case "text": case "string": case "ref": return "";
        case "number": case "int": return 0;
        case "bool": return false;
        case "enum": return p_spec.values[0];
        case "color": return "#808080";
        case "refs": return Array(p_spec.n).fill("");
        case "numbers": return Array(p_spec.n).fill(0);
        case "points": return [[-1, 0], [1, 1]];
        case "list": return [];
        case "object": return TYPES[p_spec.of].template();
        default: return null;
    }
}
