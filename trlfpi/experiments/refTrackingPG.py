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
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--exploration_noise", type=float, default=0.25)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--update_freq", type=int, default=25)
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--plots", action='store_true')

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
    actor = NNActor(2 + nRefs, 1, [256, 64], outputActivation=nn.Tanh).to(device)
    actorTarget = NNActor(2 + nRefs, 1, [256, 64], outputActivation=nn.Tanh).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)

    replayBuff = GymMemory(env.observation_space, env.action_space, reference_space=env.reference_space, maxSize=buffer_size, device=device)

    def updateActor():

        actor_loss = 0
        if replayBuff.size >= batch_size:

            states, actions, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)

            # Optimize actor
            actorOptim.zero_grad()

            actorInput = torch.cat((states, refs[:, :nRefs]), axis=1)
            actions, log_probs = actor.act(actorInput, sample=False, prevActions=actions)

            actor_loss = (-log_probs * (rewards - rewards.mean())).sum()
            actor_loss.backward()
            actorOptim.step()

        return actor_loss

    for episode in range(1, episodes + 1):
        state, ref = env.reset()
        total_reward = 0
        epsilon = np.exp(-episode * epsilonDecay / episodes)
        if epsilon <= 0.1:
            epsilon = 0

        states = [state[0, 0]]

        for step in range(max_episode_len):
            sample = (np.random.random() < epsilon)

            # Take action
            with torch.no_grad():
                actorInput = torch.tensor(np.hstack([state, ref[:, :nRefs]]), device=device)
                action, _ = actorTarget.act(actorInput, numpy=True, sample=sample)

            next_state, reward, done, next_ref = env.step(action)
            total_reward += reward

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
                report.log('netSigma', actor.sigma.item())

            states.append(next_state[0, 0])

            if done:
                break

            state = next_state
            ref = next_ref

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

    report.generateReport()
    report.pickle('actor', actor.state_dict())
