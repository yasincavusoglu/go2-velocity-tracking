import argparse
import glob
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 hız takibi (Isaac Lab + PPO)")
parser.add_argument("komut", choices=["train", "play"])
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--video", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import GaitReward
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

KLASOR = os.path.dirname(os.path.abspath(__file__))
AYAKLAR = ".*_foot"
YERINDE_DONUS = 0.1  # Vx ve Vy'si sıfırlanıp yalnız yaw rate bırakılan (yerinde dönüş) komutların oranı


def sahneyi_ayarla(cfg):
    cfg.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.terrain_levels = None

    cfg.sim.dt = 0.005
    cfg.decimation = 4
    cfg.episode_length_s = 20.0
    cfg.scene.terrain.physics_material.static_friction = 1.0
    cfg.scene.terrain.physics_material.dynamic_friction = 1.0

    cfg.scene.contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=cfg.sim.dt
    )
    cfg.scene.height_scanner = None
    cfg.observations.policy.height_scan = None

    cfg.events.physics_material.params["static_friction_range"] = (0.5, 1.25)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    cfg.events.base_com = None
    cfg.events.base_external_force_torque = None


class YerindeDonusluKomut(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        yerinde = torch.rand(len(ids), device=self.device) < YERINDE_DONUS
        self.vel_command_b[ids[yerinde], :2] = 0.0


def komutlari_ayarla(cfg):
    cmd = cfg.commands.base_velocity
    cmd.class_type = YerindeDonusluKomut
    cmd.heading_command = False
    cmd.ranges.lin_vel_x = (-1.0, 1.0)
    cmd.ranges.lin_vel_y = (-0.6, 0.6)
    cmd.ranges.ang_vel_z = (-1.0, 1.0)

    cfg.actions.joint_pos.scale = 0.25
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Go2HizPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_hiz_takibi"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def yaw_frame_xy(quat_wxyz, vec_w):
    w, x, y, z = quat_wxyz.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx, vy = vec_w[:, 0], vec_w[:, 1]
    return torch.stack((c * vx + s * vy, -s * vx + c * vy), dim=-1)


def track_lin_vel_xy_yaw_exp(env, command_name, std, asset_name="robot"):
    robot = env.scene[asset_name]
    vel_xy = yaw_frame_xy(robot.data.root_quat_w, robot.data.root_lin_vel_w)
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-torch.sum(torch.square(cmd_xy - vel_xy), dim=1) / std**2)


def track_yaw_rate_exp(env, command_name, std, asset_name="robot"):
    robot = env.scene[asset_name]
    cmd_yaw_rate = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(cmd_yaw_rate - robot.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def feet_air_time(env, command_name, sensor_cfg, threshold):
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward * moving


def feet_slip(env, sensor_cfg, asset_cfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0
    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.square(foot_vel_xy), dim=-1)
    return torch.sum(slip * in_contact, dim=1)


def feet_too_close(env, asset_cfg, min_width):
    robot = env.scene[asset_cfg.name]
    rel = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w.unsqueeze(1)
    y = torch.stack([yaw_frame_xy(robot.data.root_quat_w, rel[:, i])[:, 1] for i in range(4)], dim=1)
    width = torch.stack((y[:, 0] - y[:, 1], y[:, 2] - y[:, 3]), dim=1)
    return torch.sum((min_width - width).clamp(min=0.0), dim=1)


@configclass
class Go2OdulCfg:
    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy_yaw_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_yaw_rate = RewTerm(func=track_yaw_rate_exp, weight=0.75, params={"command_name": "base_velocity", "std": 0.5})

    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    feet_air_time = RewTerm(
        func=feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=AYAKLAR),
            "threshold": 0.5,
        },
    )

    feet_slip = RewTerm(
        func=feet_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=AYAKLAR),
            "asset_cfg": SceneEntityCfg("robot", body_names=AYAKLAR),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )
    ayak_araligi = RewTerm(
        func=feet_too_close,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "min_width": 0.1,
        },
    )

    tiris = RewTerm(
        func=GaitReward,
        weight=1.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)


@configclass
class Go2HizEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        sahneyi_ayarla(self)
        komutlari_ayarla(self)
        self.rewards = Go2OdulCfg()


def egit():
    env_cfg = Go2HizEnvCfg()
    agent_cfg = Go2HizPPOCfg()
    env_cfg.scene.num_envs = args.num_envs or 2048
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(KLASOR, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


@configclass
class Go2HizPlayEnvCfg(Go2HizEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = False
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (1.2, 1.2, 0.6)


PROGRAM = [
    (2.0, (0.0, 0.0, 0.0)),
    (4.0, (0.8, 0.0, 0.0)),
    (3.0, (-0.5, 0.0, 0.0)),
    (3.0, (0.0, 0.4, 0.0)),
    (3.0, (0.0, -0.4, 0.0)),
    (3.0, (0.0, 0.0, 0.8)),
    (3.0, (0.0, 0.0, -0.8)),
    (3.0, (0.5, 0.0, 0.8)),
    (2.0, (0.0, 0.0, 0.0)),
]


def oynat():
    adaylar = glob.glob(os.path.join(KLASOR, "logs", "*", "model_*.pt"))
    if not args.checkpoint and not adaylar:
        raise SystemExit("model bulunamadı: önce train çalıştırın ya da --checkpoint ile model yolu verin")
    ckpt = os.path.abspath(args.checkpoint or max(adaylar, key=os.path.getmtime))

    env_cfg = Go2HizPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs or 4
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg = Go2HizPPOCfg()
    dt = env_cfg.sim.dt * env_cfg.decimation
    toplam_adim = int(round(sum(sure for sure, _ in PROGRAM) / dt))
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, toplam_adim * dt + 1.0)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(ckpt), "video"),
            name_prefix="go2_hiz_takibi",
            step_trigger=lambda step: step == 0,
            video_length=toplam_adim,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    gecis = int(round(0.5 / dt))
    obs = env.get_observations()
    for sure, cmd in PROGRAM:
        olcumler = []
        for step in range(int(round(sure / dt))):
            cmd_term.vel_command_b[:] = torch.tensor(cmd, device=env.unwrapped.device)
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            if step >= gecis:
                vel_xy = yaw_frame_xy(robot.data.root_quat_w, robot.data.root_lin_vel_w)
                yaw_rate = robot.data.root_ang_vel_w[:, 2:3]
                olcumler.append(torch.cat((vel_xy, yaw_rate), dim=1).mean(dim=0).cpu())
        olculen = torch.stack(olcumler).mean(dim=0).tolist()
        print(f"[play] komut {cmd} -> ölçülen {[round(v, 2) for v in olculen]}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        if args.komut == "train":
            egit()
        else:
            oynat()
    finally:
        app.close()
