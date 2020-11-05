import argparse
import gym
import numpy as np
import torch
from torch import nn

from trlfpi.nn.actor import NNActor
from trlfpi.report import Report
from trlfpi.memory import GymMemory

torch.set_default_dtype(torch.double)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--aCost", type=float, default=0.0)
    parser.add_argument("--a_lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.3)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--weightDecay", type=float, default=0.0)
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # SETUP ARGUMENTS
    args = parser.parse_args()
    discount = args.discount
    a_lr = args.a_lr
    episodes = args.episodes
    max_episode_len = args.max_episode_len
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    update_freq = args.update_freq
    weightDecay = args.weightDecay
    nRefs = args.nRefs
    aCost = args.aCost
    systemPlots = args.plots
    plot_freq = args.plot_freq

    # Report
    report = Report('REINFORCE')
    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)
    env.alpha = aCost

    # Smaller net
    # Init policy network
    actor = NNActor(2 + nRefs, 1, [64, 8],
                    activation=nn.Tanh,
                    outputActivation=nn.Identity).to(device)
    actorTarget = NNActor(2 + nRefs, 1, [64, 8],
                          activation=nn.Tanh,
                          outputActivation=nn.Identity).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr, weight_decay=weightDecay)

    replayBuff = GymMemory(env.observation_space, env.action_space, reference_space=env.reference_space, maxSize=buffer_size, device=device)

    def updateActor():

        actor_loss = 0
        if replayBuff.size >= batch_size:

            states, actions, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)

            # Optimize actor
            actor.train()
            actorOptim.zero_grad()

            actorInput = torch.cat((states, refs[:, 1:nRefs + 1]), axis=1)
            pActions, log_probs = actor.act(actorInput, sample=False, prevActions=actions)

            actor_loss = (-log_probs * (rewards - rewards.mean())).mean()
            actor_loss.backward()
            actorOptim.step()

        return actor_loss

    bestScore = -1e6

    for episode in range(1, episodes + 1):
        state, ref = env.reset()
        total_reward = 0

        states = []
        actions = []
        refs = []

        for step in range(max_episode_len):
            # sample = (np.random.random() < epsilon)
            sample = True

            # Take action
            with torch.no_grad():
                actorInput = torch.tensor(np.hstack([state, ref[:, 1:nRefs + 1]]), device=device)
                action, _ = actorTarget.act(actorInput, numpy=True, sample=sample)

            next_state, reward, done, next_ref = env.step(action)
            total_reward += reward

            # Save for plotting
            states.append(state[0, 0])
            actions.append(action[0, 0])
            refs.append(ref[0, 0])

            # Update actor
            replayBuff.add(*map(lambda x: torch.tensor(x), [state,
                                                            action,
                                                            reward,
                                                            next_state,
                                                            done,
                                                            ref]))
            actor_loss = updateActor()
            if actor_loss != 0:
                report.log('actorLoss', actor_loss)

            if done:
                break

            state = next_state
            ref = next_ref

        # Plot to see how it looks
        if total_reward > bestScore:
            bestScore = total_reward
            report.pickle('actor_best', actor.state_dict())

        if systemPlots and episode % plot_freq == 0:
            plotData = np.stack((states, refs, actions), axis=-1)
            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref', 'Actions'],
                            plotData)

        if episode % update_freq == 0:
            actorTarget.load_state_dict(actor.state_dict())

        print(f"Episode {episode}: Reward = {total_reward}")
        report.log('rewards', total_reward, episode)

    report.generateReport()
    report.pickle('actor_cp', actor.state_dict())
