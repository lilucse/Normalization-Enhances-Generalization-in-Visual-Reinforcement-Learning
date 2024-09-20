from agent.drqv2_agent import DrQV2Agent
from agent.augmentations import random_overlay,random_conv
import torch
import agent.drqv2_utils
import torch.nn.functional as F

def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)

class sveaAgent(DrQV2Agent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau, num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_tb):
        super().__init__(obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau, num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_tb)
        self.svea_alpha = 0.5
        self.svea_beta = 0.5

    def update_critic(self, obs, action, reward, not_done, next_obs, step,obs_state,L):
        metrics = dict()
        with torch.no_grad():
            stddev = agent.drqv2_utils.schedule(self.stddev_schedule, step)
            next_obs = self.encoder(next_obs)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.discount * target_V)

        if self.svea_alpha == self.svea_beta:
            #obs_aug =random_conv(obs.clone())
            obs_aug =random_overlay(obs.clone())
            obs_aug_state = self.encoder(obs_aug)
            action = cat(action, action)
            target_Q = cat(target_Q, target_Q)
            obs = cat(obs_state, obs_aug_state)
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
            L.log('train_critic/loss', critic_loss, step)
        ### leave for revise
        else:
            print('alpha!=beta')
            assert self.svea_alpha == self.svea_beta

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics


    def update(self, replay_buffer, L, step):
        metrics = dict()
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        L.log('train/batch_reward', reward.mean(), step)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs_state = self.encoder(obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, not_done, next_obs, step,obs_state,L))

        # update actor
        metrics.update(self.update_actor(obs_state.detach(), step,L))

        # update critic target
        agent.drqv2_utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

from agent.drqv2_agent import Encoder_cnsn
class svea_cnsnAgent(sveaAgent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
        hidden_dim, critic_target_tau, num_expl_steps,
        update_every_steps, stddev_schedule, stddev_clip, use_tb,act_cn):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.discount = 0.99

        # models
        self.encoder = Encoder_cnsn(obs_shape,act_cn=act_cn).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        self.svea_alpha = 0.5
        self.svea_beta = 0.5