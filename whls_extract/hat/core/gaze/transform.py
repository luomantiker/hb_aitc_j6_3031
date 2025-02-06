import torch


def pitchyaw_to_unit_vector(pitch, yaw, angle_rad=True):
    if not angle_rad:
        pitch = pitch * torch.pi / 180
        yaw = yaw * torch.pi / 180

    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = -(torch.cos(pitch) * torch.cos(yaw))
    return x, y, z


def unit_vector_to_pitchyaw(vector, angle_rad=True):
    vector = torch.clip(vector, -1 + 5e-6, 1 - 5e-6)
    pitch = torch.asin(vector[:, 1])
    yaw = -torch.atan(vector[:, 0] / vector[:, 2])
    gaze_angle = torch.stack([pitch, yaw], dim=1)
    if not angle_rad:
        gaze_angle = gaze_angle * 180 / torch.pi
    return gaze_angle
