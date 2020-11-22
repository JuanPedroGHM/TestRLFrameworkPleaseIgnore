import argparse
import gym
import numpy as np
import torch
from torch import nn

from trlfpi.nn.actor import NNActor
from trlfpi.report import Report
from trlfpi.timer import Timer
from trlfpi.memory import GymMemory

torch.set_default_dtype(torch.double)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=str)
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--cpus", type=int, default=1)

    # Env params
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.3)

    # NN Actor params
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--update_freq", type=int, default=250)
    parser.add_argument("--aCost", type=float, default=0.0)
    parser.add_argument("--weightDecay", type=float, default=0.0)

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
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    update_freq = args.update_freq
    aCost = args.aCost
    weightDecay = args.weightDecay

    systemPlots = args.plots
    plot_freq = args.plot_freq

    # Report
    report = Report(args.report)
    timer = Timer()
    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)
    env.alpha = aCost

    # Init policy networks
    actor = NNActor(nRefs + 1, 1, [64, 8],
                    activation=nn.Tanh,
                    outputActivation=nn.Identity).to(device)
    actorTarget = NNActor(nRefs + 1, 1, [64, 8],
                          activation=nn.Tanh,
                          outputActivation=nn.Identity).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr, weight_decay=weightDecay)

    # Replay buffer
    replayBuff = GymMemory(buffer_size)

    def update(step):

        states, actions, log_probs, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)

        # Get logProbs
        actorOptim.zero_grad()
        actorInput = torch.cat([refs[:, 1:nRefs + 1] - states[:, [0]],
                                states[:, [1]]], axis=1)
        actorAction, c_log_probs = actor.act(actorInput, sample=False, prevActions=actions)

        # Importance Weight
        iw = torch.exp(c_log_probs - log_probs).detach() + 1e-9

        # Optimize actor
        adv = rewards - rewards.mean()
        a_loss = (-c_log_probs.T * adv * iw).mean()
        a_loss.backward()
        actorOptim.step()

        # Update target networks
        if step % update_freq == 0:
            actorTarget.load_state_dict(actor.state_dict())

        return a_loss.detach().cpu().numpy()

    bestScore = -1e9
    for episode in range(1, episodes + 1):
        state, ref = env.reset()
        total_reward = 0
        total_c_loss = 0
        total_a_loss = 0

        states = []
        refs = []
        actions = []

        for step in range(max_episode_len):
            # Take action
            with torch.no_grad():
                actorInput = torch.tensor(np.hstack([ref[:, 1:nRefs + 1] - state[:, [0]],
                                                     state[:, [1]]]), device=device)
                action, log_prob = actorTarget.act(actorInput, numpy=True, sample=True)
            next_state, reward, done, next_ref = env.step(action)
            total_reward += reward

            states.append(state[0, 0])
            refs.append(ref[0, 0])
            actions.append(action[0, 0])

            # Update actor
            replayBuff.add(list(map(lambda x: torch.tensor(x, device=device),
                           [state, action, log_prob, reward, next_state, int(done), ref])))

            if replayBuff.size >= batch_size:
                actor_loss = update(step)
                total_a_loss += actor_loss

            if done:
                break

            state = next_state
            ref = next_ref

            # Plot to see how it looks
        if systemPlots and episode % plot_freq == 0:
            plotData = np.stack((states, refs, actions), axis=-1)
            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref', 'Action'],
                            plotData)

        print(f"Episode {episode}: Reward = {total_reward}, Actor_Loss = {total_a_loss}")
        report.log('rewards', total_reward, episode)
        report.log('actor_loss', total_a_loss, episode)

        if total_reward > bestScore:
            bestScore = total_reward
            report.pickle('actor_best', actor.state_dict())
            report.pickle('actorT_best', actorTarget.state_dict())

    report.generateReport()
    report.pickle('actor_cp', actor.state_dict())
    report.pickle('actorT_cp', actorTarget.state_dict())
