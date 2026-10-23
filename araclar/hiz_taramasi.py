import importlib.util
import json
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dosya = os.path.join(KOK, "go2_hiz_takibi.py")
model = os.path.abspath(os.environ.get("MODEL", os.path.join(KOK, "logs", "2026-10-14_18-29-11", "model_1499.pt")))
cikti = os.path.abspath(os.environ.get("CIKTI", os.path.join(KOK, "sonuclar", "tarama")))
robot_sayisi = int(os.environ.get("ROBOT", "16"))
os.makedirs(cikti, exist_ok=True)

sys.argv = [dosya, "play", "--headless", "--checkpoint", model, "--num_envs", str(robot_sayisi)]
spec = importlib.util.spec_from_file_location("go2_hiz_takibi", dosya)
g = importlib.util.module_from_spec(spec)
sys.modules["go2_hiz_takibi"] = g
spec.loader.exec_module(g)
torch = g.torch
args = g.args

DUR = (1.0, (0.0, 0.0, 0.0))
PROGRAM = [DUR]
for v in (0.2, 0.4, 0.6, 0.8, 1.0):
    PROGRAM += [(3.0, (v, 0.0, 0.0))]
PROGRAM += [DUR]
for v in (-0.2, -0.4, -0.6, -0.8, -1.0):
    PROGRAM += [(3.0, (v, 0.0, 0.0))]
PROGRAM += [DUR]
for v in (0.2, 0.4, 0.6, -0.2, -0.4, -0.6):
    PROGRAM += [(3.0, (0.0, v, 0.0)), DUR]
for w in (0.3, 0.6, 1.0, -0.3, -0.6, -1.0):
    PROGRAM += [(3.0, (0.0, 0.0, w)), DUR]
for c in ((0.5, 0.3, 0.0), (0.8, 0.0, 0.5), (-0.4, 0.0, -0.5), (0.3, -0.3, 0.4)):
    PROGRAM += [(3.0, c), DUR]

try:
    env_cfg = g.Go2HizPlayEnvCfg()
    env_cfg.scene.num_envs = robot_sayisi
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg = g.Go2HizPPOCfg()
    dt = env_cfg.sim.dt * env_cfg.decimation
    adim = int(round(sum(d for d, _ in PROGRAM) / dt))
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, adim * dt + 1.0)
    env = g.RslRlVecEnvWrapper(g.ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = g.OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, foot_names = contact.find_bodies(g.AYAKLAR)
    gecis = int(round(0.5 / dt))
    obs = env.get_observations()
    sonuc = []
    for sure, cmd in PROGRAM:
        hizlar, temaslar, dusme = [], [], 0
        for step in range(int(round(sure / dt))):
            cmd_term.vel_command_b[:] = torch.tensor(cmd, device=env.unwrapped.device)
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            dusme += int(dones.sum().item())
            if step >= gecis:
                vel_xy = g.yaw_frame_xy(robot.data.root_quat_w, robot.data.root_lin_vel_w)
                hizlar.append(torch.cat((vel_xy, robot.data.root_ang_vel_w[:, 2:3]), dim=1).cpu())
                temaslar.append((contact.data.current_contact_time[:, foot_ids] > 0).float().cpu())
        if not any(cmd):
            continue
        h = torch.stack(hizlar)
        c = torch.stack(temaslar)
        robot_ort = h.mean(dim=0)
        sira = [foot_names.index(n) for n in ("FL_foot", "RR_foot", "FR_foot", "RL_foot")]
        uyum = []
        for r in range(c.shape[1]):
            d = []
            for a, b in ((sira[0], sira[1]), (sira[2], sira[3])):
                if c[:, r, a].std() > 0 and c[:, r, b].std() > 0:
                    d.append(torch.corrcoef(torch.stack((c[:, r, a], c[:, r, b])))[0, 1].item())
            if len(d) == 2:
                uyum.append(sum(d) / 2)
        sonuc.append({
            "komut": list(cmd),
            "olculen": [round(v, 3) for v in robot_ort.mean(dim=0).tolist()],
            "robot_min": [round(v, 3) for v in robot_ort.min(dim=0).values.tolist()],
            "robot_max": [round(v, 3) for v in robot_ort.max(dim=0).values.tolist()],
            "capraz_uyum": round(sum(uyum) / len(uyum), 2) if uyum else None,
            "dort_ayak_yerde": round((c.sum(dim=2) == 4).float().mean().item(), 2),
            "dusme": dusme,
        })
    env.close()
    with open(os.path.join(cikti, "tarama.json"), "w") as f:
        json.dump(sonuc, f, indent=1)
    satirlar = []
    for s in sonuc:
        k, o = s["komut"], s["olculen"]
        fark = ", ".join(f"{100 * (o[i] - k[i]) / abs(k[i]):+.1f}%" for i in range(3) if k[i] != 0)
        satirlar.append(f"komut {k} -> ölçülen {o}  fark {fark}  çapraz uyum {s['capraz_uyum']}  düşme {s['dusme']}")
    with open(os.path.join(cikti, "tarama.txt"), "w") as f:
        f.write("\n".join(satirlar) + "\n")
    for satir in satirlar:
        print("[tarama]", satir, flush=True)
finally:
    g.app.close()
