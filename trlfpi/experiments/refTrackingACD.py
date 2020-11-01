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
report = Report('ACD')
timer = Timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--cpus", type=int, default=1)

    # Env params
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.3)

    # NN Actor params
    parser.add_argument("--c_lr", type=float, default=1e-3)
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--update_freq", type=int, default=2)
    parser.add_argument("--aCost", type=float, default=0.0)
    parser.add_argument("--weightDecay", type=float, default=0.0)

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
    epsilonDecay = args.epsilonDecay
    exploration_std = args.exploration_std
    tau = args.tau
    update_freq = args.update_freq
    aCost = args.aCost
    weightDecay = args.weightDecay

    systemPlots = args.plots
    plot_freq = args.plot_freq

    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

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

    # Init critit [State + Ref + Action]
    critic = NNCritic((1 + nRefs) * 2 + 1, [256, 32],
                      activation=nn.Tanh,
                      outputActivation=InvertedRELU).to(device)
    criticTarget = NNCritic((1 + nRefs) * 2 + 1, [256, 32],
                            activation=nn.Tanh,
                            outputActivation=InvertedRELU).to(device)
    criticTarget.load_state_dict(critic.state_dict())
    criticTarget.eval()

    criticOptim = torch.optim.Adam(critic.parameters(), lr=c_lr, weight_decay=weightDecay)
    criticLossF = torch.nn.MSELoss()

    # Replay buffer
    replayBuff = GymMemory(env.observation_space,
                           env.action_space,
                           reference_space=env.reference_space,
                           device=device,
                           maxSize=buffer_size)

    def update(step):

        actor_loss = 0
        critic_loss = 0
        if replayBuff.size >= batch_size:

            states, actions, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)
            # 1) Update critic
            # 1.5) Get deltas

            with torch.no_grad():
                deltas = torch.cat([refs[:, [0]] - states[:, [0]],
                                    refs[:, [1]] - next_states[:, [0]],
                                    torch.zeros((states.shape[0], nRefs), device=device)], axis=1)
                d2 = torch.cat([states[:, [1]],
                                next_states[:, [1]],
                                torch.zeros((states.shape[0], nRefs), device=device)], axis=1)
                cStates = next_states
                predictedActions = []
                for i in range(2, nRefs + 2):
                    cActionsInput = torch.cat([refs[:, i:nRefs + i] - cStates[:, [0]],
                                              cStates[:, [1]]], axis=1)
                    cActions, _ = actorTarget(cActionsInput)
                    predictedActions.append(cActions)

                    pStates = env.predict(cStates, cActions, gpu=True).T
                    deltas[:, [i]] += refs[:, [i]] - pStates[:, [0]]
                    d2[:, [i]] += pStates[:, [1]]
                    cStates = pStates

                cInput = torch.cat([actions, deltas[:, :nRefs + 1], d2[:, :nRefs + 1]], axis=1)
                cNextInput = torch.cat([predictedActions[0], deltas[:, 1:nRefs + 2], d2[:, 1:nRefs + 2]], axis=1)

            # 2) Update actor
            criticOptim.zero_grad()
            qs = critic(cInput)
            next_q = (1 - dones) * criticTarget(cNextInput)
            loss = criticLossF(qs, rewards + discount * next_q)
            loss.backward()
            criticOptim.step()
            critic_loss += loss

            # Get logProbs
            actorOptim.zero_grad()
            qs = critic(cInput)
            actorInput = torch.cat([refs[:, 1:nRefs + 1] - states[:, [0]],
                                    states[:, [1]]], axis=1)
            actorAction, log_probs = actor.act(actorInput, sample=False, prevActions=actions)

            # Optimize actor
            nll = -log_probs.T @ qs
            actionLoss = aCost * actorAction.pow(2).sum()
            actor_loss = nll + actionLoss
            actor_loss.backward()
            actorOptim.step()

            # Update target networks
            if step % update_freq == 0:
                for targetP, oP in zip(criticTarget.parameters(), critic.parameters()):
                    targetP = (1 - tau) * targetP + tau * oP

                for targetP, oP in zip(actorTarget.parameters(), actor.parameters()):
                    targetP = (1 - tau) * targetP + tau * oP

        return actor_loss, critic_loss

    bestScore = -1e9
    for episode in range(1, episodes + 1):
        state, ref = env.reset()
        total_reward = 0

        states = []
        refs = []
        actions = []

        for step in range(max_episode_len):
            # Take action
            with torch.no_grad():
                actorInput = torch.tensor(np.hstack([ref[:, 1:nRefs + 1] - state[:, [0]],
                                                     state[:, [1]]]), device=device)
                action, _ = actor.act(actorInput, numpy=True, sample=True)
            next_state, reward, done, next_ref = env.step(action)
            total_reward += reward

            states.append(state[0, 0])
            refs.append(ref[0, 0])
            actions.append(action[0, 0])

            # Update actor
            replayBuff.add(*map(lambda x: torch.tensor(x), [state, action, reward, next_state, done, ref]))
            actor_loss, critic_loss = update(step)
            if actor_loss != 0:
                report.log('actorLoss', actor_loss)
                report.log('criticLoss', critic_loss)

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

        print(f"Episode {episode}: Reward = {total_reward}")
        report.log('rewards', total_reward, episode)

        if total_reward > bestScore:
            bestScore = total_reward
            report.pickle('actor_best', actor.state_dict())
            report.pickle('critic_best', critic.state_dict())
            report.pickle('actorT_best', actorTarget.state_dict())
            report.pickle('criticT_best', criticTarget.state_dict())

    report.generateReport()
    report.pickle('actor_cp', actor.state_dict())
    report.pickle('critic_cp', critic.state_dict())
    report.pickle('actorT_cp', actorTarget.state_dict())
    report.pickle('criticT_cp', criticTarget.state_dict())
