import importlib.util
import os
import sys

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dosya = os.path.join(KOK, "go2_hiz_takibi.py")
model = os.path.abspath(os.environ.get("MODEL", os.path.join(KOK, "logs", "2026-10-14_18-29-11", "model_1499.pt")))
cikti = os.path.abspath(os.environ.get("CIKTI", os.path.join(KOK, "sonuclar", "bacak")))
n = int(os.environ.get("ROBOT", "16"))
os.makedirs(cikti, exist_ok=True)

sys.argv = [dosya, "play", "--headless", "--checkpoint", model, "--num_envs", str(n)]
spec = importlib.util.spec_from_file_location("go2_hiz_takibi", dosya)
g = importlib.util.module_from_spec(spec)
sys.modules["go2_hiz_takibi"] = g
spec.loader.exec_module(g)
torch = g.torch
args = g.args

try:
    env_cfg = g.Go2HizPlayEnvCfg()
    env_cfg.scene.num_envs = n
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg = g.Go2HizPPOCfg()
    dt = env_cfg.sim.dt * env_cfg.decimation
    adim = int(round(sum(d for d, _ in g.PROGRAM) / dt))
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, adim * dt + 1.0)
    env = g.RslRlVecEnvWrapper(g.ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = g.OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene.sensors["contact_forces"]
    adlar = [f"{b}_{p}" for p in ("thigh", "calf", "foot") for b in ("FL", "FR", "RL", "RR")]
    ids, bulunan = robot.find_bodies(adlar, preserve_order=True)
    c_ids, c_bulunan = contact.find_bodies(adlar, preserve_order=True)
    assert list(bulunan) == adlar and list(c_bulunan) == adlar, (bulunan, c_bulunan)
    kayit = {k: [] for k in ("komut", "konum", "kok_konum", "kok_quat", "kuvvet", "dusme")}
    obs = env.get_observations()
    for sure, cmd in g.PROGRAM:
        for step in range(int(round(sure / dt))):
            cmd_term.vel_command_b[:] = torch.tensor(cmd, device=env.unwrapped.device)
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            d = robot.data
            kayit["komut"].append(np.array(cmd))
            kayit["konum"].append(d.body_pos_w[:, ids].cpu().numpy())
            kayit["kok_konum"].append(d.root_pos_w.cpu().numpy())
            kayit["kok_quat"].append(d.root_quat_w.cpu().numpy())
            kayit["kuvvet"].append(contact.data.net_forces_w[:, c_ids].norm(dim=-1).cpu().numpy())
            kayit["dusme"].append(dones.cpu().numpy())
    np.savez_compressed(
        os.path.join(cikti, "bacak.npz"), dt=dt, adlar=np.array(adlar), **{k: np.array(v) for k, v in kayit.items()}
    )
    env.close()
    print(f"[bacak] tamam -> {cikti}", flush=True)
finally:
    g.app.close()
