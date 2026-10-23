import importlib.util
import os
import sys

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dosya = os.path.join(KOK, "go2_hiz_takibi.py")
model = os.path.abspath(os.environ.get("MODEL", os.path.join(KOK, "logs", "2026-10-14_18-29-11", "model_1499.pt")))
cikti = os.path.abspath(os.environ.get("CIKTI", os.path.join(KOK, "sonuclar", "detay16")))
robot_sayisi = int(os.environ.get("ROBOT", "16"))
video = os.environ.get("VIDEO", "") == "1"
os.makedirs(cikti, exist_ok=True)

sys.argv = [dosya, "play", "--headless", "--checkpoint", model, "--num_envs", str(robot_sayisi)]
if video:
    sys.argv.append("--video")
spec = importlib.util.spec_from_file_location("go2_hiz_takibi", dosya)
g = importlib.util.module_from_spec(spec)
sys.modules["go2_hiz_takibi"] = g
spec.loader.exec_module(g)
torch = g.torch
args = g.args

try:
    env_cfg = g.Go2HizPlayEnvCfg()
    env_cfg.scene.num_envs = robot_sayisi
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg = g.Go2HizPPOCfg()
    dt = env_cfg.sim.dt * env_cfg.decimation
    adim = int(round(sum(sure for sure, _ in g.PROGRAM) / dt))
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, adim * dt + 1.0)

    env = g.ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if video else None)
    if video:
        env = g.gym.wrappers.RecordVideo(
            env,
            video_folder=cikti,
            name_prefix="ham",
            step_trigger=lambda s: s == 0,
            video_length=adim,
            disable_logger=True,
        )
    env = g.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = g.OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, foot_names = contact.find_bodies(g.AYAKLAR)
    rfoot_ids, rfoot_names = robot.find_bodies(g.AYAKLAR)
    assert foot_names == rfoot_names

    plan = [cmd for sure, cmd in g.PROGRAM for _ in range(int(round(sure / dt)))]
    kayit = {k: [] for k in ("komut", "hiz", "temas", "govde_z", "egim", "kayma", "tork")}
    obs = env.get_observations()
    for cmd in plan:
        cmd_term.vel_command_b[:] = torch.tensor(cmd, device=env.unwrapped.device)
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        vel_xy = g.yaw_frame_xy(robot.data.root_quat_w, robot.data.root_lin_vel_w)
        kayit["komut"].append(list(cmd))
        kayit["hiz"].append(torch.cat((vel_xy, robot.data.root_ang_vel_w[:, 2:3]), dim=1).cpu())
        temas = contact.data.current_contact_time[:, foot_ids] > 0
        kayit["temas"].append(temas.cpu())
        kayit["govde_z"].append(robot.data.root_pos_w[:, 2].cpu())
        kayit["egim"].append(torch.norm(robot.data.projected_gravity_b[:, :2], dim=1).cpu())
        ayak_hiz = torch.norm(robot.data.body_lin_vel_w[:, rfoot_ids, :2], dim=-1)
        kayit["kayma"].append((ayak_hiz * temas).sum(dim=1).cpu())
        kayit["tork"].append(robot.data.applied_torque.abs().mean(dim=1).cpu())
    env.close()

    veri = {k: (np.array(v) if k == "komut" else torch.stack(v).numpy()) for k, v in kayit.items()}
    np.savez_compressed(os.path.join(cikti, "kayit.npz"), dt=dt, ayaklar=np.array(foot_names), **veri)
    with open(os.path.join(cikti, "robot0.csv"), "w") as f:
        f.write("t,cmd_vx,cmd_vy,cmd_wz,vx,vy,wz\n")
        for i, (c, h) in enumerate(zip(veri["komut"], veri["hiz"])):
            f.write(",".join(f"{v:.4f}" for v in (i * dt, *c, *h[0])) + "\n")
    print(f"[dd] {len(plan)} adım, {robot_sayisi} robot -> {cikti}", flush=True)
finally:
    g.app.close()
