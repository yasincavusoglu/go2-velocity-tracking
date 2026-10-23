import json
import os
import sys

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
EKSEN = ("vx", "vy", "wz")


def kayan(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(x)
    for i in range(len(x)):
        b = max(0, i - n + 1)
        out[i] = (c[i + 1] - c[b]) / (i + 1 - b)
    return out


def korelasyon(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def analiz(yol):
    d = np.load(yol)
    dt = float(d["dt"])
    komut, hiz, temas = d["komut"], d["hiz"], d["temas"].astype(float)
    ayak = list(d["ayaklar"])
    i_fl, i_fr, i_rl, i_rr = (ayak.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"))
    T, N, _ = hiz.shape
    gecis = int(round(0.5 / dt))
    pencere = int(round(0.2 / dt))
    sinir = [0] + [t for t in range(1, T) if (komut[t] != komut[t - 1]).any()] + [T]
    sonuc = []
    for a, b in zip(sinir[:-1], sinir[1:]):
        cmd = tuple(float(v) for v in komut[a])
        s = slice(a + gecis, b)
        h = hiz[s]
        kayan_h = kayan(hiz[a:b], pencere)
        satir = {"ad": AD.get(cmd, str(cmd)), "komut": cmd}
        eks = [i for i in range(3) if cmd[i] != 0]
        robot_ort = h.mean(axis=0)
        if eks:
            i = eks[0]
            hata = h[:, :, i] - cmd[i]
            satir["ort"] = float(robot_ort[:, i].mean())
            satir["sapma_yuzde"] = float(100 * (robot_ort[:, i].mean() - cmd[i]) / abs(cmd[i]))
            satir["en_kotu_robot"] = float(robot_ort[np.argmax(np.abs(robot_ort[:, i] - cmd[i])), i])
            satir["rms"] = float(np.sqrt((hata**2).mean()))
            satir["salinim"] = float(h[:, :, i].std(axis=0).mean())
            yakin = np.abs(kayan_h[:, :, i] - cmd[i]) <= 0.1 * abs(cmd[i])
            oturma = []
            for r in range(N):
                t0 = next((t for t in range(b - a) if yakin[t:, r].all()), None)
                oturma.append(np.nan if t0 is None else t0 * dt)
            satir["oturma_ort"] = float(np.nanmean(oturma))
            satir["oturma_en_kotu"] = float(np.nanmax(oturma)) if not np.isnan(oturma).all() else np.nan
            satir["oturamayan"] = int(np.isnan(oturma).sum())
            if len(eks) > 1:
                j = eks[1]
                satir["ort2"] = float(robot_ort[:, j].mean())
                satir["sapma2_yuzde"] = float(100 * (robot_ort[:, j].mean() - cmd[j]) / abs(cmd[j]))
        diger = [i for i in range(3) if i not in eks]
        satir["kayma"] = {EKSEN[i]: float(robot_ort[:, i].mean()) for i in diger}
        satir["kayma_en_kotu"] = max((abs(float(robot_ort[:, i].mean())) for i in diger), default=0.0)
        c = temas[s]
        capraz = [
            np.nanmean([korelasyon(c[:, r, i_fl], c[:, r, i_rr]), korelasyon(c[:, r, i_fr], c[:, r, i_rl])])
            for r in range(N)
        ]
        on = [korelasyon(c[:, r, i_fl], c[:, r, i_fr]) for r in range(N)]
        satir["capraz_uyum"] = float(np.nanmean(capraz)) if not np.isnan(capraz).all() else np.nan
        satir["capraz_en_kotu"] = float(np.nanmin(capraz)) if not np.isnan(capraz).all() else np.nan
        satir["on_ayak_uyum"] = float(np.nanmean(on)) if not np.isnan(on).all() else np.nan
        satir["ucus"] = float((c.sum(axis=2) == 0).mean())
        satir["dort_yerde"] = float((c.sum(axis=2) == 4).mean())
        satir["govde_z_std_mm"] = float(1000 * d["govde_z"][s].std(axis=0).mean())
        satir["egim"] = float(d["egim"][s].mean())
        satir["kayma_hizi"] = float(d["kayma"][s].mean())
        satir["tork"] = float(d["tork"][s].mean())
        sonuc.append(satir)
    return sonuc, N


def yazdir(ad, sonuc, N):
    print(f"\n=== {ad} ({N} robot) ===")
    print(
        f"{'bölüm':13s} {'ölçülen':>8s} {'sapma%':>7s} {'en kötü':>8s} {'rms':>6s} {'salınım':>7s} {'oturma':>7s}"
        f" {'en geç':>7s} {'kayma(en büyük)':>16s} {'çapraz':>7s} {'ç.en kötü':>9s} {'ön':>6s} {'uçuş%':>6s}"
        f" {'4yerde%':>8s} {'z mm':>5s} {'kayma':>6s}"
    )
    for s in sonuc:
        if "ort" in s:
            ek = f" ({s['ort2']:+.2f} {s['sapma2_yuzde']:+.1f}%)" if "ort2" in s else ""
            print(
                f"{s['ad']:13s} {s['ort']:+8.3f} {s['sapma_yuzde']:+6.1f}% {s['en_kotu_robot']:+8.3f} {s['rms']:6.3f}"
                f" {s['salinim']:7.3f} {s['oturma_ort']:6.2f}s {s['oturma_en_kotu']:6.2f}s {s['kayma_en_kotu']:16.3f}"
                f" {s['capraz_uyum']:7.2f} {s['capraz_en_kotu']:9.2f} {s['on_ayak_uyum']:6.2f} {100 * s['ucus']:6.1f}"
                f" {100 * s['dort_yerde']:8.1f} {s['govde_z_std_mm']:5.1f} {s['kayma_hizi']:6.3f}{ek}"
                + (f"  oturamayan robot: {s['oturamayan']}" if s["oturamayan"] else "")
            )
        else:
            print(
                f"{s['ad']:13s} {'':>8s} {'':>7s} {'':>8s} {'':>6s} {'':>7s} {'':>7s} {'':>7s}"
                f" {s['kayma_en_kotu']:16.3f} {s['capraz_uyum']:7.2f} {'':>9s} {s['on_ayak_uyum']:6.2f}"
                f" {100 * s['ucus']:6.1f} {100 * s['dort_yerde']:8.1f} {s['govde_z_std_mm']:5.1f} {s['kayma_hizi']:6.3f}"
            )
    hareket = [s for s in sonuc if "ort" in s]
    ozet = {
        "ort_mutlak_sapma_yuzde": float(np.mean([abs(s["sapma_yuzde"]) for s in hareket])),
        "en_buyuk_sapma_yuzde": float(max(abs(s["sapma_yuzde"]) for s in hareket)),
        "ort_kayma": float(np.mean([s["kayma_en_kotu"] for s in hareket])),
        "ort_oturma": float(np.mean([s["oturma_ort"] for s in hareket])),
        "en_dusuk_capraz": float(min(s["capraz_uyum"] for s in hareket)),
        "ort_on_ayak": float(np.mean([s["on_ayak_uyum"] for s in hareket])),
    }
    print("ÖZET:", {k: round(v, 3) for k, v in ozet.items()})
    return ozet


if __name__ == "__main__":
    girdiler = sys.argv[1:] or ["son_model=" + os.path.join(KOK, "sonuclar", "detay16", "kayit.npz")]
    tum = {}
    for arg in girdiler:
        ad, yol = arg.split("=")
        if not os.path.exists(yol):
            print(f"\n=== {ad}: kayıt yok ({yol})")
            continue
        sonuc, N = analiz(yol)
        tum[ad] = {"ozet": yazdir(ad, sonuc, N), "bolumler": sonuc}
    with open(os.path.join(KOK, "sonuclar", "detay16.json"), "w") as f:
        json.dump(tum, f, indent=1, default=float)
