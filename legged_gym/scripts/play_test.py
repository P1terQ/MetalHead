

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import time
import math
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi
import numpy as np
import torch


def trans_matrix_ba(m, t):
    r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                   np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                   np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                  [np.sin(t[2]) * np.cos(t[1]),
                   np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                   np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                  [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
    trans = np.hstack([r, np.array(m)[:, np.newaxis]])
    trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
    return trans


def quaternion2rpy(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.jump_up_command = False  # TODO
    env_cfg.env.episode_length_s = 200.
    env_cfg.env.reference_state_initialization = False
    env_cfg.env.recovery_init_prob = 0

    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)


    camera_pos_bias = np.array([-2.8, -1.8, 1.8])

    for i in range(1000 * int(env.max_episode_length)):
        time.sleep(0.02)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
 
                
        if MOVE_CAMERA:
            robot_pos = env.root_states[0, :3].cuda().cpu().numpy()
            robot_quat = env.root_states[0, 3:7].cuda().cpu().numpy()
            robot_ang = np.array(quaternion2rpy(robot_quat))
            camera_position = (trans_matrix_ba(robot_pos, [0, 0, robot_ang[-1]]) @ np.append(camera_pos_bias, 1))[:-1]
            box_pos_bias = np.array([np.random.randint(-30, 30)*0.1, np.random.randint(-20, 20)*0.1, np.random.randint(0, 20)*0.1])
            box_position = (trans_matrix_ba(robot_pos, [0, 0, robot_ang[-1]]) @ np.append(box_pos_bias, 1))[:-1]
            env.set_camera(camera_position, robot_pos)
        

        env.gym.sync_frame_time(env.sim)


if __name__ == '__main__':
    MOVE_CAMERA = True
    args = get_args()
    args.task = "a1_amp_jump_cmd_play"
    play(args)
