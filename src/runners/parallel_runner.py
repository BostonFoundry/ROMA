from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        # self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
        #                     for worker_conn in self.worker_conns]

        # TODO get this from config
        # task_maps = ['MMMRG1', 'MMMRG2', 'MMMRG3', 'MMMRG4', 'MMMRG5']
        #task_maps = ['S0Z7','S1Z6','S2Z5','S3Z4','S4Z3','S5Z2', 'S6Z1', 'S7Z0']
        # maps with hold back in center
        task_maps = ['S0Z7','S1Z6','S2Z5','S5Z2', 'S6Z1', 'S7Z0']
        #task_maps = ['S3Z4']
        #task_maps = ['S4Z3']
        #task_maps = ['new_map_H1', 'new_map_H2', 'new_map_H3', 'new_map_H4', 'new_map_H5']
        print(f'Running the following maps: {task_maps}')

        self.n_tasks = len(task_maps)
        self.ps = []

        for i, worker_conn in enumerate(self.worker_conns):
            env_args = self.args.env_args.copy()
            env_args["map_name"] = task_maps[i % self.n_tasks]
            self.ps.append(Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_args)))
            ))

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        # mshiffer - only saving from first environment
        self.parent_conns[0].send(("save_replay", None))
        self.parent_conns[0].recv()

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "agent_mask": []
        }

        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            # used to build mask
            n_agents = len(data["obs"])
            # used to pad with zeros
            max_agents = self.args.n_agents

            # # to be used to pad (maybe do zeros like)
            obs_dim = len(data["obs"][0])
            action_dim = len(data["avail_actions"][0])

            # # Pad obs
            obs = np.zeros((max_agents, obs_dim), dtype=np.float32)
            obs[:n_agents] = np.array(data["obs"])

            # # Pad avail_actions
            avail_actions = np.zeros((max_agents, action_dim), dtype=np.float32)
            avail_actions[:n_agents] = np.array(data["avail_actions"])

            # Pad agent_mask
            agent_mask = np.zeros((max_agents), dtype=np.float32)
            agent_mask[:n_agents] = 1.0

            # TODO Need to update the appends here
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(avail_actions)
            pre_transition_data["obs"].append(obs)
            pre_transition_data["agent_mask"].append(agent_mask)

            # print(f'DEBUG shape of obs {data["obs"]}')
            # print(f'DEBUG shape of avail_actions {data["avail_actions"]}')
            # print(f'DEBUG shape of state {data["state"]}')
            # print(f'DEBUG shape of agent_mask {agent_mask}')


        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.mac == "separate_mac":
            self.mac.init_latent(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        # Calculate num_agents per episode here
        agent_mask = self.batch['agent_mask'][:, 0, :]
        n_agents_per_env = agent_mask.sum(dim=1).squeeze(-1).to(th.int)

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        # mshiffer Updated to only send actions for n_agents in this episode
                        parent_conn.send(("step", cpu_actions[action_idx][:(n_agents_per_env[idx].item())]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [], 
                "agent_mask": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    n_agents = len(data["obs"])
                    # used to pad with zeros
                    max_agents = self.args.n_agents

                    # # to be used to pad (maybe do zeros like)
                    obs_dim = len(data["obs"][0])
                    action_dim = len(data["avail_actions"][0])

                    # # Pad obs
                    obs = np.zeros((max_agents, obs_dim), dtype=np.float32)
                    obs[:n_agents] = np.array(data["obs"])

                    # # Pad avail_actions
                    avail_actions = np.zeros((max_agents, action_dim), dtype=np.float32)
                    avail_actions[:n_agents] = np.array(data["avail_actions"])

                    # Pad agent_mask
                    agent_mask = np.zeros((max_agents), dtype=np.float32)
                    agent_mask[:n_agents] = 1.0

                    #TODO need to update appends to use the above padded versions
                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(avail_actions)
                    pre_transition_data["obs"].append(obs)
                    pre_transition_data["agent_mask"].append(agent_mask)

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            #mshiffer 11/17
            self.logger.print_recent_stats()
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_replay":
            remote.send(env.save_replay())
        else:
            raise NotImplementedError(f'{cmd}')


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

