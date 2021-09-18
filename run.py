import os
import gym
import numpy as np
import torch
import tqdm
import wandb
from torch.nn import MSELoss
from nn import ActorNN, CriticNN

WANDB = False
# WANDB = True
RUN_ID = np.random.randint(1000000)
checkpoint_every_n_epochs = 10

n_epochs = 100000
n_episodes = 20
max_steps = 200
discount = 0.99

grad_clip = 1
entropy_weight = 0.

actor = ActorNN(
    in_size=8,
    h_size=200,
    out_size=2,
    n_h_layers=2,
)

critic = CriticNN(
    in_size=8,
    h_size=200,
    out_size=1,
    n_h_layers=2,
)
# actor
actor_lr = 0.00005
momentum = 0.9
weight_decay = 0.0001
actor_optimizer = torch.optim.SGD(lr=actor_lr, momentum=momentum, weight_decay=weight_decay,
                                  params=actor.parameters())
# critic
critic_lr = 0.00001
critic_momentum = 0.9
critic_weight_decay = 0.001
critic_optimizer = torch.optim.SGD(lr=critic_lr, momentum=critic_momentum, weight_decay=critic_weight_decay,
                                   params=critic.parameters())

if __name__ == '__main__':
    torch.set_num_threads(1)

    env = gym.make('Swimmer-v2')

    if WANDB:
        wandb_dir = 'wandb'
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_params = dict(
            dir=wandb_dir,
            project='rl_policy_gradient',
            group='swimmer_a2c_400T',
            name=f'swimmer_{RUN_ID}_0.0001lr_20*200_0.00001lr2_gain0.5_entr0_gam0.99_LNet',
            notes='',
            entity='kaglowka',
        )
        wandb.init(**wandb_params)

    actor.train()
    for e in range(n_epochs):
        print(f'--EPOCH {e}--')
        if e % checkpoint_every_n_epochs == 0:
            result_dir = os.path.join('results', f'run_{RUN_ID}')
            os.makedirs(result_dir, exist_ok=True)
            torch.save(actor.state_dict(), os.path.join(result_dir, f'actor_{e}.ch'))
            torch.save(critic.state_dict(), os.path.join(result_dir, f'critic_{e}.ch'))

        transitions = []
        stats_critic_losses = []
        stats_actor_losses = []
        stats_rewards = []
        stats_action_stds = []
        for _ in tqdm.tqdm(range(n_episodes)):
            state = env.reset()
            done = False
            t = 0
            epi_transitions = []
            disc_rewards = []
            while not done and t < max_steps:
                inp_state = torch.from_numpy(state.astype(np.float32))
                action, action_log_p, action_std = actor.predict(inp_state[None, :])
                state, reward, done, _ = env.step(np.clip(action[0].numpy(), -1, 1))
                disc_rewards.append(reward)
                inp_new_state = torch.from_numpy(state.astype(np.float32))
                transition = (inp_state, action, action_log_p[0], action_std[0], reward.astype(np.float32), inp_new_state)
                epi_transitions.append(transition)

                stats_action_stds.append(action_std.detach().numpy())
                stats_rewards.append(reward)
                t += 1

            steps = t
            last_t = t - 1
            if done:
                disc_reward = 0
            else:
                with torch.no_grad():
                    inp_state = torch.from_numpy(state.astype(np.float32))
                    # bootstrap the terminal state value from critic prediction
                    disc_reward = critic.predict(inp_state[None, :])[0]
                epi_transitions[last_t] = (*epi_transitions[last_t], disc_reward)
            for t in range(last_t - 1, -1, -1):
                disc_reward = disc_rewards[t] + discount * disc_reward
                epi_transitions[t] = (*epi_transitions[t], disc_reward)
            transitions.extend(epi_transitions)

        ### Update actor ###
        critic.eval()
        actor_optimizer.zero_grad()

        states = []
        action_stds = []
        action_log_ps = []
        disc_rewards = []
        for transition in transitions:
            # print(sars)
            (inp_state, action, action_log_p, action_std, reward, inp_new_state, disc_reward) = transition
            states.append(inp_state)
            action_stds.append(action_std)
            action_log_ps.append(action_log_p)
            disc_rewards.append(disc_reward)

        # make a batch input
        inp_state = torch.vstack(states)
        action_std = torch.vstack(action_stds)
        action_log_p = torch.vstack(action_log_ps)
        disc_reward = torch.tensor(disc_rewards, dtype=torch.float32)
        pred_disc_reward = critic.predict(inp_state)
        adv = disc_reward - pred_disc_reward[:, 0]
        entropy_term = entropy_weight * (-(torch.log(2 * np.pi * action_std) + 1) / 2).mean()
        model_loss = ((-1) * adv[:, None] * action_log_p + entropy_term) / adv.shape[0]
        model_loss.sum().backward()
        torch.nn.utils.clip_grad_value_(actor.parameters(), grad_clip)
        stats_actor_losses.append(model_loss.detach().numpy())
        actor_optimizer.step()

        ### Update critic ###
        mse_loss = MSELoss()
        critic.train()
        critic_optimizer.zero_grad()
        states = []
        new_states = []
        disc_rewards = []

        for transition in transitions:
            (state, action, action_log_p, action_std, reward, new_state, disc_reward) = transition
            states.append(state)
            new_states.append(new_state)
            disc_rewards.append(disc_reward)

        inp_state = torch.vstack(states)
        disc_reward = torch.tensor(disc_rewards, dtype=torch.float32)

        pred_disc_reward = critic.predict(inp_state)[:, 0]
        critic_loss = mse_loss(pred_disc_reward, disc_reward).sum() / disc_reward.shape[0]
        critic_loss.backward()

        torch.nn.utils.clip_grad_value_(actor.parameters(), grad_clip)
        critic_optimizer.step()

        stats_critic_losses.append(critic_loss.detach())

        ### Report epoch results
        mean_reward_sum = np.mean(stats_rewards)
        mean_critic_loss = np.mean(np.abs(stats_critic_losses))
        mean_actor_loss = np.mean(np.abs(stats_actor_losses))
        var_mean = np.mean(stats_action_stds)
        if WANDB:
            wandb.log({
                'reward_sum mean': mean_reward_sum,
                'reward_sum max': np.max(stats_rewards),
                'reward_sum min': np.min(stats_rewards),
                'critic_loss mean': mean_critic_loss,
                'actor_loss mean': mean_actor_loss,
                'var_mean': var_mean,

            })

        if (e + 1) % 1 == 0:
            if WANDB:
                print(f'{wandb_params["name"]}:')
            print(f"\rEpoch {e + 1} mean reward: {mean_reward_sum}")
            print(f"\rEpoch {e + 1} mean critic loss: {mean_critic_loss}")
            print(f"\rEpoch {e + 1} mean actor loss: {mean_actor_loss}")
