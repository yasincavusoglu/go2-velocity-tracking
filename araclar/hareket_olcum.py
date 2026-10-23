import importlib.util
import os
import sys

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dosya = os.path.join(KOK, "go2_hiz_takibi.py")
model = os.path.abspath(os.environ.get("MODEL", os.path.join(KOK, "logs", "2026-10-14_18-29-11", "model_1499.pt")))
cikti = os.path.abspath(os.environ.get("CIKTI", os.path.join(KOK, "sonuclar", "hareket")))
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
    foot_c, names = contact.find_bodies(g.AYAKLAR)
    foot_r, names_r = robot.find_bodies(g.AYAKLAR)
    assert names == names_r, (names, names_r)
    alanlar = ("komut", "ayak_z", "temas", "govde_z", "tork", "eklem_hiz", "eklem_aci", "hiz", "dusme")
    kayit = {k: [] for k in alanlar}
    obs = env.get_observations()
    for sure, cmd in g.PROGRAM:
        for step in range(int(round(sure / dt))):
            cmd_term.vel_command_b[:] = torch.tensor(cmd, device=env.unwrapped.device)
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            d = robot.data
            kayit["komut"].append(np.array(cmd))
            kayit["ayak_z"].append(d.body_pos_w[:, foot_r, 2].cpu().numpy())
            kayit["temas"].append((contact.data.current_contact_time[:, foot_c] > 0).cpu().numpy())
            kayit["govde_z"].append(d.root_pos_w[:, 2].cpu().numpy())
            kayit["tork"].append(d.applied_torque.cpu().numpy())
            kayit["eklem_hiz"].append(d.joint_vel.cpu().numpy())
            kayit["eklem_aci"].append(d.joint_pos.cpu().numpy())
            vel_xy = g.yaw_frame_xy(d.root_quat_w, d.root_lin_vel_w)
            kayit["hiz"].append(torch.cat((vel_xy, d.root_ang_vel_w[:, 2:3]), 1).cpu().numpy())
            kayit["dusme"].append(dones.cpu().numpy())
    kutle = robot.root_physx_view.get_masses().sum(dim=1).cpu().numpy()
    np.savez_compressed(
        os.path.join(cikti, "hareket.npz"),
        dt=dt,
        ayaklar=np.array(names),
        eklemler=np.array(robot.joint_names),
        kutle=kutle,
        varsayilan=robot.data.default_joint_pos[0].cpu().numpy(),
        sinir=robot.data.joint_pos_limits[0].cpu().numpy(),
        yumusak=robot.data.soft_joint_pos_limits[0].cpu().numpy(),
        **{k: np.array(v) for k, v in kayit.items()},
    )
    env.close()
    print(f"[olcum] tamam -> {cikti}", flush=True)
finally:
    g.app.close()
