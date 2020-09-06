import argparse
import gym
import numpy as np
import torch
from torch import nn

from trlfpi.approx import NNActor
from trlfpi.report import Report
from trlfpi.memory import GymMemory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--nRefs", type=int, default=3)
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--c_lr", type=float, default=1e-5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_freq", type=int, default=25)
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--plots", action='store_true')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # SETUP ARGUMENTS
    args = parser.parse_args()
    discount = args.discount
    a_lr = args.a_lr
    c_lr = args.c_lr
    episodes = args.episodes
    max_episode_len = args.max_episode_len
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    update_freq = args.update_freq
    epsilonDecay = args.epsilonDecay
    nRefs = args.nRefs
    systemPlots = args.plots

    # Report
    report = Report('pg_ref_no_critic')
    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

    # Smaller net
    # Init policy network
    actor = NNActor(2 + nRefs, 1, [32], outputActivation=nn.Tanh).to(device)
    actorTarget = NNActor(2 + nRefs, 1, [32], outputActivation=nn.Tanh).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)

    replayBuff = GymMemory(env.observation_space, env.action_space, maxSize=buffer_size)

    def actWithRef(actorNet, x: np.ndarray, sample=False):
        # x : B x X
        currentState = x[:, :2]
        refs = x[:, 2:]

        # B x T x D
        state_p = np.zeros((nRefs, 2))
        actions = np.zeros((nRefs, 1))
        for i in range(nRefs):
            actorInput = torch.as_tensor(
                np.hstack((currentState, refs[:, i:nRefs + i])),
                dtype=torch.float32
            ).to(device)
            actions[i, :], _ = actorNet.act(actorInput, numpy=True, sample=sample)
            currentState = env.predict(currentState.T, actions[[i], :]).T
            state_p[i, :] = currentState

        deltas = state_p[:, [0]].T - refs[:, :nRefs]
        return actions[[0], :], deltas

    def updateActor():

        actor_loss = 0
        if replayBuff.size >= batch_size:

            states, actions, rewards, next_states, dones = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), replayBuff.get(batchSize=batch_size)))

            # Optimize actor
            actorOptim.zero_grad()

            # B x D
            currentStates = states[:, :2]
            refs = states[:, 2:]

            # B x T x D
            state_p = torch.zeros((batch_size, nRefs, 2)).to(device)
            next_actions = torch.zeros((batch_size, nRefs)).to(device)
            log_probs = torch.zeros((batch_size, nRefs)).to(device)
            discountVector = torch.tensor([[discount**i for i in range(1, nRefs + 1)]]).to(device).T
            for i in range(nRefs):
                actorInput = torch.cat((currentStates, refs[:, i:nRefs + i]), 1)

                tmpAct, tmpLP = actor.act(actorInput, sample=False, prevActions=actions)
                next_actions[:, [i]] = tmpAct
                log_probs[:, [i]] = tmpLP

                currentState = torch.as_tensor(env.predict(currentStates.T.cpu().data.numpy(),
                                                           actions[:, [0]].T.cpu().data.numpy()))
                currentState = currentState.T
                state_p[:, i, :] = currentState

            deltas = state_p[:, :, 0] - refs[:, :nRefs]
            G = rewards - torch.pow(deltas, 2) @ discountVector

            actor_loss = (-log_probs[:, [0]] * (G - rewards.mean())).mean()
            actor_loss.backward()
            actorOptim.step()

        return actor_loss

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        epsilon = np.exp(-episode * epsilonDecay / episodes)

        states = [state[0, 0]]

        for step in range(max_episode_len):
            sample = (np.random.random() < epsilon)
            action, deltas = actWithRef(actorTarget, state, sample=sample)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update actor
            actor_loss = updateActor()
            if actor_loss != 0:
                report.log('actorLoss', actor_loss)

            replayBuff.add(state, action, reward, next_state, done)
            states.append(next_state[0, 0])

            if done:
                break

            state = next_state

        if episode % update_freq == 0:
            actorTarget.load_state_dict(actor.state_dict())

            # Plot to see how it looks
            if systemPlots:
                plotData = np.stack((states, env.reference.r[:len(states)]), axis=-1)
                report.savePlot(f"episode_{episode}_plot",
                                ['State', 'Ref'],
                                plotData)

        print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon:.2f}")
        report.log('rewards', total_reward, episode)
        report.log('epsilon', epsilon, episode)

    report.generatePlots()
    report.pickle('actor', actor.state_dict())
