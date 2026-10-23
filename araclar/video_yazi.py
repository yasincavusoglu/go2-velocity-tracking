import csv
import os
import sys


def etiket(cmd):
    vx, vy, wz = cmd
    if vx == vy == wz == 0:
        return "Dur"
    parcalar = []
    if vx:
        parcalar.append("ileri" if vx > 0 else "geri")
    if vy:
        parcalar.append("sola yan" if vy > 0 else "sağa yan")
    if wz:
        yon = "sola" if wz > 0 else "sağa"
        parcalar.append(f"{yon} dönüş" if parcalar else f"yerinde {yon} dönüş")
    metin = " + ".join(parcalar)
    return {"i": "İ"}.get(metin[0], metin[0].upper()) + metin[1:]


def video_yaz(rows, ham_video, cikti, isim):
    import imageio.v2 as imageio
    import matplotlib
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    yazi_tipi = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    F_baslik = ImageFont.truetype(os.path.join(yazi_tipi, "DejaVuSans-Bold.ttf"), 44)
    F_kalin = ImageFont.truetype(os.path.join(yazi_tipi, "DejaVuSans-Bold.ttf"), 24)
    F_alt = ImageFont.truetype(os.path.join(yazi_tipi, "DejaVuSans.ttf"), 26)
    F_satir = ImageFont.truetype(os.path.join(yazi_tipi, "DejaVuSans.ttf"), 24)

    def sayi(v, birim):
        v = round(v, 2) + 0.0
        metin = f"+{v:.2f}" if v > 0 else f"{v:.2f}"
        return metin.replace(".", ",").replace("-", "−") + f" {birim}"

    okuyucu = imageio.get_reader(ham_video)
    fps = okuyucu.get_meta_data()["fps"]
    yazici = imageio.get_writer(
        cikti, fps=fps, codec="libx264", quality=6,
        pixelformat="yuv420p", macro_block_size=8,
    )

    kart = Image.fromarray((np.asarray(okuyucu.get_data(0)) * 0.25).astype(np.uint8))
    d = ImageDraw.Draw(kart)
    w, h = kart.size
    d.text((w / 2, h / 2 - 80), "Unitree Go2: hız takibi", font=F_baslik, fill="white", anchor="mm")
    d.text((w / 2, h / 2 - 20), "Isaac Lab + PPO, eğitilmiş policy", font=F_alt, fill="white", anchor="mm")
    d.text((w / 2, h / 2 + 25), "Komutlar: ileri, geri, sola, sağa, yerinde dönüş (sola, sağa), ileri + dönüş",
           font=F_alt, fill="white", anchor="mm")
    d.text((w / 2, h / 2 + 100), isim, font=F_alt, fill="white", anchor="mm")
    for _ in range(int(3.5 * fps)):
        yazici.append_data(np.asarray(kart))

    pencere = int(round(0.2 * fps))
    kutu_sag = 510 + int(d.textlength("yaw rate −0,00 rad/s", font=F_satir)) + 24
    for i, kare in enumerate(okuyucu):
        r = rows[min(i, len(rows) - 1)]
        b = max(0, i - pencere + 1)
        olc = [sum(s[k] for s in rows[b : i + 1]) / (i + 1 - b) for k in (4, 5, 6)]
        img = Image.fromarray(kare)
        katman = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(katman).rounded_rectangle((16, 16, kutu_sag, 150), radius=10, fill=(0, 0, 0, 150))
        img = Image.alpha_composite(img.convert("RGBA"), katman).convert("RGB")
        d = ImageDraw.Draw(img)
        d.text((32, 26), etiket(r[1:4]), font=F_kalin, fill="white")
        d.text((32, 66), "Komut", font=F_satir, fill="white")
        d.text((32, 104), "Ölçülen", font=F_satir, fill="white")
        for j, (ad, birim, x) in enumerate((("Vx", "m/s", 150), ("Vy", "m/s", 330), ("yaw rate", "rad/s", 510))):
            d.text((x, 66), f"{ad} {sayi(r[1 + j], birim)}", font=F_satir, fill="white")
            d.text((x, 104), f"{ad} {sayi(olc[j], birim)}", font=F_satir, fill=(255, 220, 120))
        d.text((w - 16, h - 16), f"t = {r[0]:.1f} s".replace(".", ","), font=F_satir, fill="white", anchor="rd")
        yazici.append_data(np.asarray(img))
    yazici.close()
    print(f"video: {cikti}", flush=True)


if __name__ == "__main__":
    ham_video, csv_yolu, cikti = sys.argv[1:4]
    with open(csv_yolu) as f:
        rows = [[float(r[k]) for k in ("t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz")] for r in csv.DictReader(f)]
    video_yaz(rows, ham_video, cikti, "Yasin Çavuşoğlu")
