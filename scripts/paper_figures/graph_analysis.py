import os, glob, sys, json, random
import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = r"C:\Users\Dillu\Desktop\EL\road-extraction\Satellite-Imagery-based-Road-Network-Extraction"
sys.path.insert(0, REPO)
from backend.src.utils.graph_postprocess import hysteresis_threshold, connect_canopy_gaps


OUT_FIG = os.path.join(REPO, "docs", "paper", "figures", "fig_graph_analysis.pdf")
OUT_JSON = os.path.join(REPO, "figures", "real", "measurements", "graph_analysis.json")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

so = ort.SessionOptions(); so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
sess = ort.InferenceSession(os.path.join(REPO, "models", "mobilevit_v2.onnx"), sess_options=so,
                            providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def predict(img):
    x = np.ascontiguousarray(((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None])
    s = None
    for ax in [None, (2,), (3,), (2, 3)]:
        xi = np.ascontiguousarray(np.flip(x, ax)) if ax else x
        p = sess.run(None, {inp: xi})[0]
        p = np.flip(p, ax) if ax else p
        s = p if s is None else s + p
    m = hysteresis_threshold((s / 4).squeeze(), 0.35, 0.12)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return connect_canopy_gaps(m)

def skeleton_graph(mask):
    """Road graph: junction pixel clusters and end points become nodes, skeleton paths become edges."""
    sk = skeletonize(mask > 127)
    P = nx.Graph()
    ys, xs = np.nonzero(sk)
    pts = set(zip(ys.tolist(), xs.tolist()))
    for y, x in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in pts:
                    P.add_edge((y, x), (y + dy, x + dx), w=float(np.hypot(dy, dx)))
    key = {n for n in P if P.degree(n) != 2}
    cid = {}
    for k, comp in enumerate(nx.connected_components(P.subgraph(key))):
        for n in comp:
            cid[n] = k
    centre = {}
    for n, k in cid.items():
        centre.setdefault(k, []).append(n)
    centre = {k: tuple(np.mean(v, axis=0).round().astype(int)) for k, v in centre.items()}
    G = nx.Graph()
    for k, c in centre.items():
        G.add_node(c)
    seen = set()
    for start in key:
        for nb in P.neighbors(start):
            if nb in key or (start, nb) in seen:
                continue
            path, length, prev, cur = [start, nb], P[start][nb]["w"], start, nb
            while cur not in key:
                nxt = [m for m in P.neighbors(cur) if m != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                length += P[prev][cur]["w"]; path.append(cur)
            seen.add((cur, path[-2])); seen.add((start, nb))
            if cur in key:
                a, b = centre[cid[start]], centre[cid[cur]]
                if a != b and length >= 5 and (not G.has_edge(a, b) or G[a][b]["length"] > length):
                    G.add_edge(a, b, length=length, pixels=path)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    return G

def efficiency(G):
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    tot = 0.0
    for src, d in nx.all_pairs_dijkstra_path_length(G, weight="length"):
        tot += sum(1.0 / v for t, v in d.items() if t != src and v > 0)
    return tot / (n * (n - 1))

FRACS = np.arange(0, 0.41, 0.05)

def attack(G, order):
    E0 = efficiency(G); n = G.number_of_nodes(); out = []
    for f in FRACS:
        H = G.copy(); H.remove_nodes_from(order[:int(round(f * n))])
        out.append(efficiency(H) / E0 if E0 > 0 else 0.0)
    return out

random.seed(0)
results, keep = {}, None
for path in sorted(glob.glob(os.path.join(REPO, "data", "samples", "*.jpg"))):
    name = os.path.basename(path).split("_")[0]
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    G = skeleton_graph(predict(img))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy() if G.number_of_nodes() else G
    if G.number_of_nodes() < 10:
        print(name, "skipped, nodes:", G.number_of_nodes()); continue
    bc = nx.betweenness_centrality(G, weight="length", normalized=True)
    targeted = attack(G, sorted(bc, key=bc.get, reverse=True))
    rand = np.mean([attack(G, random.sample(list(G.nodes()), G.number_of_nodes())) for _ in range(20)], axis=0)
    i10 = list(np.round(FRACS, 2)).index(0.1)
    results[name] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                     "R_targeted_10pct": round(targeted[i10], 3), "R_random_10pct": round(float(rand[i10]), 3),
                     "targeted": [round(v, 3) for v in targeted], "random": [round(float(v), 3) for v in rand]}
    print(name, results[name]["nodes"], results[name]["R_targeted_10pct"], results[name]["R_random_10pct"])
    if keep is None or G.number_of_nodes() > keep[2].number_of_nodes():
        keep = (name, img, G, bc)

json.dump({"note": "Largest connected component of the predicted road graph per sample tile (4 flip TTA, hysteresis 0.35/0.12, closing, gap bridging). Efficiency uses edge pixel length. R = E(G without top 10% nodes) / E(G). Random removal averaged over 20 trials.",
           "fractions": [round(float(f), 2) for f in FRACS], "tiles": results}, open(OUT_JSON, "w"), indent=2)

plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
name, img, G, bc = keep
ax = axs[0]; ax.imshow(img, alpha=0.6)
for u, v, d in G.edges(data=True):
    p = np.array(d["pixels"]); ax.plot(p[:, 1], p[:, 0], color="white", lw=1.2)
ys, xs = zip(*G.nodes()); c = [bc[n] for n in G.nodes()]
sc = ax.scatter(xs, ys, c=c, cmap="plasma", s=14, edgecolors="black", linewidths=0.3, zorder=3)
top = max(bc, key=bc.get); ax.scatter([top[1]], [top[0]], s=80, facecolors="none", edgecolors="red", lw=1.2, zorder=4)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"(a) Betweenness, tile {name}")
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)

ax = axs[1]; pct = FRACS * 100
T = np.mean([r["targeted"] for r in results.values()], axis=0)
Rr = np.mean([r["random"] for r in results.values()], axis=0)
ax.plot(pct, T, "o-", color="#c0392b", ms=3, label="Highest betweenness first")
ax.plot(pct, Rr, "s--", color="#2c3e50", ms=3, label="Random")
ax.set_xlabel("Nodes removed (%)"); ax.set_ylabel("Relative efficiency"); ax.set_ylim(0, 1.05)
ax.set_title(f"(b) Mean over {len(results)} tiles"); ax.legend(fontsize=6, frameon=False); ax.grid(alpha=0.3)

ax = axs[2]; names = list(results); y = np.arange(len(names))
ax.barh(y - 0.2, [results[n]["R_targeted_10pct"] for n in names], 0.4, color="#c0392b", label="Targeted")
ax.barh(y + 0.2, [results[n]["R_random_10pct"] for n in names], 0.4, color="#7f8c8d", label="Random")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6); ax.set_xlim(0, 1); ax.set_xlabel("Resilience index R")
ax.set_title("(c) R after removing 10% of nodes"); ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(OUT_FIG, bbox_inches="tight"); fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved", OUT_FIG)
