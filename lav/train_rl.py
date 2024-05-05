import tqdm
import torch
from lav.lav_final_v2 import LAV
from lav.utils.datasets import get_data_loader
from lav.utils.logger import Logger
import argparse
import pathlib
import ruamel.yaml as yaml
import lav.tools as tools
import sys
import lav.wrappers as wrappers
import functools
import numpy as np
from team_code_v2.lav_agent import LavAgent
from agents.navigation.carla_env_dream import CarlaEnv

sys.path.append(str(pathlib.Path(__file__).parent))

def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))

def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  length = len(episode['reward']) - 1
  if length >= 50:
    filename = tools.save_episodes(directory, [episode])[0]
    if mode == 'eval':
      cache.clear()
    if mode == 'train' and config.dataset_size:
      total = 0
      for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if total <= config.dataset_size - length:
          total += len(ep['reward']) - 1
        else:
          del cache[key]
      logger.scalar('dataset_size', total + length)
    cache[str(filename)] = episode
    logger.scalar(f'{mode}_episodes', len(cache))
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()

def make_env(config, logger, mode, train_eps, eval_eps):
  suite, task = config.task.split('_', 1)
  if suite == 'carla':
    env = CarlaEnv(
            render_display=False,  # for local debugging only
            display_text=False,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=config.image_size,
            max_episode_steps=1000,
            frame_skip=config.action_repeat,
            is_other_cars=True,
            port=2000
        )
    env_eval = env
  else:
    raise NotImplementedError(suite)

  if suite == 'carla':
    env = wrappers.TimeLimit(env, config.time_limit)
    env_eval = wrappers.TimeLimit(env_eval, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    env_eval = wrappers.SelectAction(env_eval, key='action')
    mode = 'train'
    callbacks = [functools.partial(
          process_episode, config, logger, mode, train_eps, eval_eps)]
    env = wrappers.CollectDataset(env, callbacks)
    mode = 'eval'
    callbacks_eval = [functools.partial(
          process_episode, config, logger, mode, train_eps, eval_eps)]
    env_eval = wrappers.CollectDataset(env_eval, callbacks_eval)
    env = wrappers.RewardObs(env)
    env_eval = wrappers.RewardObs(env_eval)
    return env, env_eval
  else:
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    if (mode == 'train') or (mode == 'eval'):
      callbacks = [functools.partial(
          process_episode, config, logger, mode, train_eps, eval_eps)]
      env = wrappers.CollectDataset(env, callbacks)
    env = wrappers.RewardObs(env)
    return env

def main(config):
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / 'train_eps'
    config.evaldir = config.evaldir or logdir / 'eval_eps'
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)

    print('Logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print('Create envs.')
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    suite, task = config.task.split('_', 1)
    if suite == 'carla':
        envs_list = [make('train') for _ in range(config.envs)]
        train_envs = [envs_list[0][0]]
        eval_envs = [envs_list[0][1]]
    else:
       print('No {}. return'.format(suite))
       return
    
    # load agent run
    print('load agent from config {}'.format(config.agent_config))
    agent = LavAgent(config.agent_config)

    # todo start to evaluate
    train_envs[0].simulate(agent, config.eval_every)
    # state = None
    # while agent._step < config.steps:
    #     logger.write()
    #     print('Start evaluation.')
    #     agent.eval()
    #     eval_policy = functools.partial(agent, training=False)
    #     tools.simulate(eval_policy, eval_envs, episodes=1)
    #     print('Start training.')
    #     agent.train()
    #     state = tools.simulate(agent, train_envs, config.eval_every, state=state)
    #     torch.save(agent.state_dict(), logdir / f'latest_model.pt')
    # for env in train_envs + eval_envs:
    #     try:
    #         env.close()
    #     except Exception:
    #         pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / 'rl_configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
