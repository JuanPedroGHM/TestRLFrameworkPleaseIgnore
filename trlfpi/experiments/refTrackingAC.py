import argparse
import gym
import numpy as np
import torch
from torch import nn

from trlfpi.nn.actor import NNActor
from trlfpi.nn.critic import NNCritic
from trlfpi.nn.activation import InvertedRELU
from trlfpi.report import Report
from trlfpi.timer import Timer
from trlfpi.memory import GymMemory

torch.set_default_dtype(torch.double)

# Report
report = Report('refTrackingAC')
timer = Timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--cpus", type=int, default=1)

    # Env params
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.7)

    # NN params
    parser.add_argument("--c_lr", type=float, default=1e-3)
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--a_update_freq", type=int, default=10)
    parser.add_argument("--c_update_freq", type=int, default=10)

    # Exploration params
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--exploration_std", type=float, default=0.5)

    # Plot args
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # SETUP ARGUMENTS
    args = parser.parse_args()

    cpus = args.cpus

    episodes = args.episodes
    max_episode_len = args.max_episode_len
    nRefs = args.nRefs
    discount = args.discount

    a_lr = args.a_lr
    c_lr = args.c_lr
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    a_update_freq = args.a_update_freq
    c_update_freq = args.c_update_freq

    epsilonDecay = args.epsilonDecay
    exploration_std = args.exploration_std

    systemPlots = args.plots
    plot_freq = args.plot_freq

    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

    # Init policy networks
    actor = NNActor(2 + nRefs, 1, [64], outputActivation=nn.Tanh).to(device)
    actorTarget = NNActor(2 + nRefs, 1, [64], outputActivation=nn.Tanh).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)

    # Init critit [State + Ref + Action]

    critic = NNCritic(2 + nRefs + 1, [256, 32], outputActivation=InvertedRELU)
    criticTarget = NNCritic(2 + nRefs + 1, [256, 32], outputActivation=InvertedRELU)
    criticTarget.load_state_dict(critic.state_dict())
    criticTarget.eval()

    criticOptim = torch.optim.Adam(critic.parameters(), lr=c_lr)
    criticLossF = torch.nn.MSELoss()

    # Replay buffer
    replayBuff = GymMemory(env.observation_space,
                           env.action_space,
                           reference_space=env.reference_space,
                           maxSize=buffer_size)

    def update():

        total_critic_loss = 0
        total_actor_loss = 0
        if replayBuff.size >= batch_size:

            # Update critic
            for _ in range(20):
                criticOptim.zero_grad()
                states, actions, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)

                cInput = torch.cat((actions, states, refs[:, :nRefs]), axis=1)
                q = critic(cInput)

                with torch.no_grad():
                    next_actions, _ = actorTarget(torch.cat([states, refs[:, :nRefs]], axis=1))
                    next_q = (1 - dones) * criticTarget(torch.cat([next_actions, next_states, refs[:, 1:nRefs + 1]], axis=1))

                loss = criticLossF(q, rewards + discount * next_q)
                loss.backward()
                criticOptim.step()

                total_critic_loss += loss

            # Get q values
            states, actions, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)
            qs = critic(cInput)

            # Optimize actor
            actorOptim.zero_grad()

            actorInput = torch.cat([states, refs[:, :nRefs]], axis=1)
            actions, log_probs = actor.act(actorInput, sample=False, prevActions=actions)

            actor_loss = (-log_probs * qs).mean()
            actor_loss.backward()
            actorOptim.step()

            total_actor_loss += actor_loss

        return total_actor_loss, total_critic_loss

    for episode in range(1, episodes + 1):
        state, ref = env.reset()
        total_reward = 0
        tC_updates = 0

        epsilon = np.exp(-episode * epsilonDecay / episodes)
        if epsilon <= 0.1:
            epsilon = 0

        states = [state[0, 0]]

        for step in range(max_episode_len):
            sample = (np.random.random() < epsilon)
            # Take action
            actorInput = torch.tensor(np.hstack([state, ref[:, :nRefs]]), device=device)
            action, _ = actorTarget.act(actorInput, numpy=True, sample=sample)
            next_state, reward, done, ref = env.step(action)
            total_reward += reward

            # Update actor
            replayBuff.add(*map(lambda x: torch.tensor(x), [state, action, reward, next_state, done, ref]))
            actor_loss, critic_loss = update()
            if actor_loss != 0:
                report.log('actorLoss', actor_loss)
                report.log('criticLoss', critic_loss)

            states.append(next_state[0, 0])

            if done:
                break

            state = next_state

        if episode % a_update_freq == 0:
            actorTarget.load_state_dict(actor.state_dict())

        if episode % c_update_freq == 0:
            criticTarget.load_state_dict(critic.state_dict())

            # Plot to see how it looks
        if systemPlots and episode % plot_freq == 0:
            plotData = np.stack((states, env.reference.r[:len(states)]), axis=-1)
            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref'],
                            plotData)

        print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon:.2f}")
        report.log('rewards', total_reward, episode)
        report.log('epsilon', epsilon, episode)

    report.generateReport()
    report.pickle('actor', actor.state_dict())
