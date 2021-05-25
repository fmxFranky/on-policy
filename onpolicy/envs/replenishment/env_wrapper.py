import gym
import numpy as np

from .config.inventory_config import env_config
from .env.inventory_env import InventoryManageEnv
from .env.inventory_utils import Utils
from .scheduler.inventory_eoq_policy import \
    ConsumerEOQPolicy as ConsumerBaselinePolicy
from .scheduler.inventory_random_policy import (BaselinePolicy,
                                                ProducerBaselinePolicy)


def is_trainable(agent_id):
  return Utils.is_consumer_agent(agent_id) and (
      agent_id.startswith('SKUStoreUnit') or
      agent_id.startswith('OuterSKUStoreUnit'))


class MyInventoryManageEnv(gym.Env):

  def __init__(self):
    self.env = InventoryManageEnv(env_config)
    self.producer_policy = ProducerBaselinePolicy(
        self.env.observation_space, self.env.action_space_producer,
        BaselinePolicy.get_config_from_env(self.env))
    self.consumer_policy = ConsumerBaselinePolicy(
        self.env.observation_space, self.env.action_space_consumer,
        BaselinePolicy.get_config_from_env(self.env))
    self.all_agents_ids = self.env.reset().keys()
    self.agents_ids = []
    for agent_id in self.all_agents_ids:
      if is_trainable(agent_id):
        self.agents_ids.append(agent_id)
    self.observation_space = [self.env.observation_space] * len(self.agents_ids)
    self.action_space = [self.env.action_space_consumer] * len(self.agents_ids)
    self.share_observation_space = [
        gym.spaces.Box(
            -300.,
            300.,
            shape=(self.env.observation_space.shape[0] * len(self.agents_ids),))
    ] * len(self.agents_ids)

  def reset(self):
    self.observation_dict = self.env.reset()
    self.extra_info_dcit = self.env.state_calculator.world_to_state(
        self.env.world)[1]
    return np.stack(
        [v for k, v in self.observation_dict.items() if k in self.agents_ids])

  def step(self, actions):
    action_dict = {}
    for agent_id in self.all_agents_ids:
      if agent_id in self.agents_ids:
        action_dict[agent_id] = actions[self.agents_ids.index(agent_id)]
      else:
        if Utils.is_producer_agent(agent_id):
          action_dict[agent_id] = self.producer_policy.compute_single_action(
              self.observation_dict[agent_id],
              info=self.extra_info_dcit[agent_id],
              explore=True)[0]
        else:
          action_dict[agent_id] = self.consumer_policy.compute_single_action(
              self.observation_dict[agent_id],
              info=self.extra_info_dcit[agent_id],
              explore=True)[0]
    self.observation_dict, self.reward_dict, self.done_dict, self.extra_info_dcit = self.env.step(
        action_dict)
    next_observations = np.stack(
        [v for k, v in self.observation_dict.items() if k in self.agents_ids])
    rewards = np.stack(
        [v for k, v in self.reward_dict.items() if k in self.agents_ids])
    dones = np.stack(
        [v for k, v in self.done_dict.items() if k in self.agents_ids])
    return next_observations, rewards[..., None], dones, self.extra_info_dcit
