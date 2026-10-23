import ast
import math
import os
from types import SimpleNamespace

import torch

KOD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "go2_hiz_takibi.py")
ADLAR = ("yaw_frame_xy", "track_lin_vel_xy_yaw_exp", "track_yaw_rate_exp", "feet_air_time", "feet_slip", "feet_too_close")


def fonksiyonlari_yukle():
    agac = ast.parse(open(KOD, encoding="utf-8").read())
    govde = [n for n in agac.body if isinstance(n, ast.FunctionDef) and n.name in ADLAR]
    alan = {"torch": torch}
    exec(compile(ast.Module(body=govde, type_ignores=[]), KOD, "exec"), alan)
    return SimpleNamespace(**{ad: alan[ad] for ad in ADLAR})


odul = fonksiyonlari_yukle()


class FakeScene(dict):
    def __init__(self, robot=None, sensors=None):
        super().__init__(robot=robot)
        self.sensors = sensors or {}


def quat_from_yaw(yaw):
    return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def make_env(cmd, lin_vel_w=(0.0, 0.0, 0.0), ang_vel_z=0.0, yaw=0.0, sensors=None, body_vel=None, body_pos=None):
    data = SimpleNamespace(
        root_quat_w=quat_from_yaw(yaw),
        root_pos_w=torch.zeros(1, 3),
        root_lin_vel_w=torch.tensor([lin_vel_w]),
        root_ang_vel_w=torch.tensor([[0.0, 0.0, ang_vel_z]]),
        body_lin_vel_w=body_vel,
        body_pos_w=body_pos,
    )
    return SimpleNamespace(
        scene=FakeScene(robot=SimpleNamespace(data=data), sensors=sensors),
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([cmd])),
        step_dt=0.02,
    )


def test_tracking_is_one_when_velocity_matches_command():
    env = make_env(cmd=(0.5, 0.2, 0.0), lin_vel_w=(0.5, 0.2, 0.0))
    assert torch.allclose(odul.track_lin_vel_xy_yaw_exp(env, "base_velocity", 0.5), torch.tensor([1.0]))


def test_tracking_drops_with_error():
    good = odul.track_lin_vel_xy_yaw_exp(make_env((0.5, 0, 0), lin_vel_w=(0.45, 0, 0)), "c", 0.5)
    bad = odul.track_lin_vel_xy_yaw_exp(make_env((0.5, 0, 0), lin_vel_w=(0.0, 0, 0)), "c", 0.5)
    assert good > bad
    assert torch.allclose(bad, torch.tensor([math.exp(-1.0)]))


def test_velocity_is_measured_in_yaw_frame():
    env = make_env(cmd=(0.5, 0.0, 0.0), lin_vel_w=(0.0, 0.5, 0.0), yaw=math.pi / 2)
    assert torch.allclose(odul.track_lin_vel_xy_yaw_exp(env, "c", 0.5), torch.tensor([1.0]), atol=1e-6)


def test_yaw_rate_tracking():
    env = make_env(cmd=(0.0, 0.0, 0.8), ang_vel_z=0.8)
    assert torch.allclose(odul.track_yaw_rate_exp(env, "c", 0.5), torch.tensor([1.0]))


class FakeContactSensor:
    def __init__(self, first_contact, last_air_time, forces_history):
        self._first = first_contact
        self.data = SimpleNamespace(last_air_time=last_air_time, net_forces_w_history=forces_history)

    def compute_first_contact(self, dt):
        return self._first


FEET = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
ROBOT_FEET = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])


def air_time_env(first, air, cmd=(0.5, 0.0, 0.0)):
    sensor = FakeContactSensor(torch.tensor([first]), torch.tensor([air]), None)
    return make_env(cmd=cmd, sensors={"contact_forces": sensor})


def test_air_time_rewards_only_at_touchdown():
    env = air_time_env([1.0, 0.0, 0.0, 0.0], [0.6, 0.9, 0.9, 0.9])
    assert torch.allclose(odul.feet_air_time(env, "c", FEET, 0.5), torch.tensor([0.1]))


def test_short_steps_are_penalized():
    env = air_time_env([1.0, 1.0, 0.0, 0.0], [0.2, 0.3, 0.0, 0.0])
    assert odul.feet_air_time(env, "c", FEET, 0.5).item() < 0


def test_no_air_time_reward_when_standing_still():
    env = air_time_env([1.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0], cmd=(0.0, 0.0, 0.5))
    assert odul.feet_air_time(env, "c", FEET, 0.5).item() == 0.0


def slip_env(force_z, foot_vel_xy):
    history = torch.zeros(1, 3, 4, 3)
    history[0, :, :, 2] = torch.tensor(force_z)
    sensor = FakeContactSensor(None, None, history)
    vel = torch.zeros(1, 4, 3)
    vel[0, :, :2] = torch.tensor(foot_vel_xy)
    return make_env(cmd=(0, 0, 0), sensors={"contact_forces": sensor}, body_vel=vel)


def test_slip_counts_only_feet_on_ground():
    env = slip_env([30.0, 0.0, 30.0, 30.0], [[0.2, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(odul.feet_slip(env, FEET, ROBOT_FEET), torch.tensor([0.04]))


def test_no_slip_when_feet_planted():
    env = slip_env([30.0] * 4, [[0.0, 0.0]] * 4)
    assert odul.feet_slip(env, FEET, ROBOT_FEET).item() == 0.0


def stance_env(feet_xy, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    pos = torch.tensor([[[c * x - s * y, s * x + c * y, 0.0] for x, y in feet_xy]])
    return make_env(cmd=(0, 0, 0), yaw=yaw, body_pos=pos)


def test_wide_stance_has_no_penalty():
    env = stance_env([(0.2, 0.15), (0.2, -0.15), (-0.2, 0.15), (-0.2, -0.15)])
    assert odul.feet_too_close(env, ROBOT_FEET, 0.1).item() == 0.0


def test_crossed_rear_feet_are_penalized():
    env = stance_env([(0.2, 0.15), (0.2, -0.15), (-0.2, -0.02), (-0.2, 0.02)])
    assert torch.allclose(odul.feet_too_close(env, ROBOT_FEET, 0.1), torch.tensor([0.14]))


def test_stance_width_is_measured_in_yaw_frame():
    feet = [(0.2, 0.15), (0.2, -0.15), (-0.2, -0.02), (-0.2, 0.02)]
    env = stance_env(feet, yaw=math.pi / 2)
    assert torch.allclose(odul.feet_too_close(env, ROBOT_FEET, 0.1), torch.tensor([0.14]), atol=1e-6)
