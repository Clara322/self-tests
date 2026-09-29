import sys, collections, numpy as np, generator as G

def vec(d):
    t = np.array(d["target"]); phi = np.arctan2(t[1], t[0]); R = np.linalg.norm(t)
    obs = d["obstacle"] or [0., 0., 0.]
    o = np.array(obs[:2]); r_o = np.linalg.norm(o)
    o_rel = [r_o, (np.arctan2(o[1], o[0]) - phi) if r_o > 0 else 0.0, obs[2]]
    lim = [0. if p[0] is None else p[0] for p in d["limits"]] + \
          [0. if p[1] is None else p[1] for p in d["limits"]]
    q = np.array(d["init_mean"]).copy(); q[0] -= phi
    return np.array([R] + d["noise_std"] + o_rel + lim + list(q))

def main(goals_path, splits_path):
    payload, records = G.load_goals(goals_path)
    sp = G.load_splits(splits_path)
    recs = {d["id"]: d for _, _, d in records}
    splits = sp["splits"]

    print(f"goal file {sp['goals_file']}  ({len(records)} goals)   sizes {sp['sizes']}\n")

    print("1. leakage")
    sets = {k: set(v) for k, v in splits.items()}
    print("   no goal in two splits :", not (sets["train"] & sets["test"]
          or sets["train"] & sets["val"] or sets["val"] & sets["test"]))
    print("   every goal assigned   :", len(set().union(*sets.values())) == len(records))
    print("   split file matches goal file:", sp["goals_file"] == goals_path)

    print("\n2. balance and representativeness (test should look like train)")
    print(f"   {'split':6s} {'n':>4s} | {'plateau':>13s} {'ceiling':>13s} {'|target|':>13s} {'noise':>13s}")
    for name in ("train", "val", "test"):
        ds = [recs[i] for i in splits[name]]
        f = lambda xs: f"{np.mean(xs):.2f}+-{np.std(xs):.2f}"
        print(f"   {name:6s} {len(ds):4d} | {f([d['plateau'] for d in ds]):>13s} "
              f"{f([d['ceiling'] for d in ds]):>13s} "
              f"{f([np.linalg.norm(d['target']) for d in ds]):>13s} "
              f"{f([max(d['noise_std']) for d in ds]):>13s}")
    for name in ("train", "val", "test"):
        c = collections.Counter(recs[i]["type"] for i in splits[name])
        print(f"   {name:6s} types: {dict(c)}")

    print("\n3. near-duplicates between test and train (same type)")
    worst = []
    for gtype in {d["type"] for d in recs.values()}:
        tr = np.array([vec(recs[i]) for i in splits["train"] if recs[i]["type"] == gtype])
        te_ids = [i for i in splits["test"] if recs[i]["type"] == gtype]
        if not len(tr) or not te_ids:
            continue
        scale = tr.std(0); scale[scale == 0] = 1.0
        for i in te_ids:
            d = np.linalg.norm((vec(recs[i]) - tr) / scale, axis=1).min()
            worst.append((d, i, gtype))
    worst.sort()
    med = np.median([w[0] for w in worst])
    print(f"   median nearest-train distance: {med:.2f} (normalised)")
    print("   five closest test goals:")
    for d, i, t in worst[:5]:
        print(f"     {i:22s} {d:.2f}" + ("   <-- suspiciously close" if d < 0.25 * med else ""))

    print("\n4. resolution")
    n_per = min(collections.Counter(recs[i]["type"] for i in splits["test"]).values())
    print(f"   smallest type in test: {n_per} goals")
    print(f"   a confusion-matrix cell near 60% carries +-{100*np.sqrt(.6*.4/n_per):.0f} percentage points")

if __name__ == "__main__":
    main(*(sys.argv[1:] or ["goals_v1.json", "splits_v1.json"]))