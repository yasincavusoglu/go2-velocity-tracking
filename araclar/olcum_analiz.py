import os

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AD = {
    (0.0, 0.0, 0.0): "dur",
    (0.8, 0.0, 0.0): "ileri",
    (-0.5, 0.0, 0.0): "geri",
    (0.0, 0.4, 0.0): "sola yan",
    (0.0, -0.4, 0.0): "sağa yan",
    (0.0, 0.0, 0.8): "yerinde sola",
    (0.0, 0.0, -0.8): "yerinde sağa",
    (0.5, 0.0, 0.8): "ileri+sola",
}


def segment_mesafe(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    pay = a * e - b * b
    s = np.where(pay > 1e-12, np.clip((b * f - c * e) / np.where(pay > 1e-12, pay, 1), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p1 + d1 * s[..., None] - (p2 + d2 * t[..., None]), axis=-1)


def bolumler(komut):
    sinir = [0] + [t for t in range(1, len(komut)) if (komut[t] != komut[t - 1]).any()] + [len(komut)]
    return list(zip(sinir[:-1], sinir[1:]))


def hareket_ozeti(yol):
    d = np.load(yol)
    q, eklemler, sinir, yumusak = d["eklem_aci"], list(d["eklemler"]), d["sinir"], d["yumusak"]
    z, temas, komut = d["ayak_z"], d["temas"].astype(bool), d["komut"]
    tork, eklem_hiz = d["tork"], d["eklem_hiz"]
    dizler = [eklemler.index(f"{b}_calf_joint") for b in ("FL", "FR", "RL", "RR")]
    yakin = [((q[:, :, j] - sinir[j, 0] < 0.05) | (sinir[j, 1] - q[:, :, j] < 0.05)).mean() * 100 for j in dizler]
    disinda = [((q[:, :, j] < yumusak[j, 0]) | (q[:, :, j] > yumusak[j, 1])).mean() * 100 for j in dizler]
    zemin = np.median(z[temas])
    tepeler = []
    for r in range(z.shape[1]):
        for f in range(z.shape[2]):
            havada = ~temas[:, r, f]
            t = 0
            while t < len(havada):
                if not havada[t]:
                    t += 1
                    continue
                son = t
                while son < len(havada) and havada[son]:
                    son += 1
                if t > 0 and son < len(havada):
                    tepeler.append(z[t:son, r, f].max() - zemin)
                t = son
    hareketli = np.array([any(k) for k in komut])
    guc = np.abs(tork[hareketli] * eklem_hiz[hareketli]).sum(-1).mean()
    doyma = (np.abs(tork) >= 0.95 * 23.5).any(-1).mean() * 100
    return [
        f"robot kütlesi ort {d['kutle'].mean():.1f} kg, düşme {int(d['dusme'].sum())}",
        f"gövde yüksekliği ort {d['govde_z'].mean() * 100:.1f} cm",
        "diz açısı ort (FL, FR, RL, RR): " + ", ".join(f"{q[:, :, j].mean():+.2f}" for j in dizler) + " rad",
        "dizin mekanik sınıra 0,05 rad'dan yakın olduğu zaman %: " + ", ".join(f"{v:.1f}" for v in yakin),
        "dizin yumuşak sınırın dışında olduğu zaman %: " + ", ".join(f"{v:.1f}" for v in disinda),
        f"ayağın en yüksek noktası medyan {np.median(tepeler) * 100:.1f} cm, "
        f"yüzde 90'ı {np.percentile(tepeler, 90) * 100:.1f} cm'nin altında",
        f"torkun sınırın %95'ini geçtiği adım %{doyma:.2f}, en büyük eklem hızı {np.abs(eklem_hiz).max():.1f} rad/s",
        f"yürürken mekanik güç ort {guc:.1f} W",
    ]


def bacak_ozeti(yol):
    d = np.load(yol)
    adlar, P, K, Q = list(d["adlar"]), d["konum"], d["kok_konum"], d["kok_quat"]
    komut, kuvvet = d["komut"], d["kuvvet"]
    w, x, y, z = (Q[..., i] for i in range(4))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    goreli = P - K[:, :, None, :]
    yan = -np.sin(yaw)[..., None] * goreli[..., 0] + np.cos(yaw)[..., None] * goreli[..., 1]
    i = {a: adlar.index(a) for a in adlar}

    def parca(bacak, u1, u2):
        return P[..., i[f"{bacak}_{u1}"], :], P[..., i[f"{bacak}_{u2}"], :]

    def en_yakin(sol, sag):
        parcalar = (("thigh", "calf"), ("calf", "foot"))
        return np.minimum.reduce(
            [segment_mesafe(*parca(sol, *a), *parca(sag, *b)) for a in parcalar for b in parcalar]
        )

    arka_aralik = yan[..., i["RL_foot"]] - yan[..., i["RR_foot"]]
    on_aralik = yan[..., i["FL_foot"]] - yan[..., i["FR_foot"]]
    arka, on = en_yakin("RL", "RR"), en_yakin("FL", "FR")
    baldir = (kuvvet[..., [i[f"{b}_calf"] for b in ("FL", "FR", "RL", "RR")]] > 1.0).any(-1)
    gecis = int(round(0.5 / float(d["dt"])))
    satirlar = ["bölüm          arka ayak aralığı    ön ayak aralığı    arka/ön bacak en yakın    baldır teması"]
    satirlar.append("               min / medyan (m)     min / medyan (m)   (cm)                      (%)")
    for a, b in bolumler(komut):
        s = slice(a + gecis, b)
        ad = AD.get(tuple(float(v) for v in komut[a]), str(komut[a]))
        satirlar.append(
            f"{ad:14s} {arka_aralik[s].min():+.3f} / {np.median(arka_aralik[s]):+.3f}      "
            f"{on_aralik[s].min():+.3f} / {np.median(on_aralik[s]):+.3f}    "
            f"{arka[s].min() * 100:5.1f} / {on[s].min() * 100:5.1f}             {baldir[s].mean() * 100:5.1f}"
        )
    satirlar.append(
        f"bütün kayıt: arka ayak aralığı medyan {np.median(arka_aralik) * 100:.1f} cm; bacaklar en çok "
        f"{arka.min() * 100:.1f} cm (arka) ve {on.min() * 100:.1f} cm (ön) yaklaşıyor; 4 cm'den yakın oldukları "
        f"zaman %{(np.minimum(arka, on) < 0.04).mean() * 100:.2f}; baldır teması %{baldir.mean() * 100:.2f}"
    )
    return satirlar


if __name__ == "__main__":
    klasor = os.path.join(KOK, "sonuclar")
    satirlar = ["eklemler ve duruş"] + hareket_ozeti(os.path.join(klasor, "hareket", "hareket.npz"))
    satirlar += ["", "bacakların birbirine uzaklığı"] + bacak_ozeti(os.path.join(klasor, "bacak", "bacak.npz"))
    with open(os.path.join(klasor, "olcumler.txt"), "w") as f:
        f.write("\n".join(satirlar) + "\n")
    print("\n".join(satirlar))
