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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=str)
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--cpus", type=int, default=1)

    # Env params
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.7)

    # NN params
    parser.add_argument("--c_lr", type=float, default=1e-5)
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--update_freq", type=int, default=2)
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
    aCost = args.aCost

    a_lr = args.a_lr
    c_lr = args.c_lr
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    tau = args.tau
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
    actor = NNActor(2 + nRefs, 1,
                    [64, 8],
                    activation=nn.Tanh,
                    outputActivation=nn.Identity).to(device)
    actorTarget = NNActor(2 + nRefs, 1,
                          [64, 8],
                          activation=nn.Tanh,
                          outputActivation=nn.Identity).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr, weight_decay=weightDecay)

    # Init critit [Action + State + nRefs]

    critic = NNCritic(1 + 2 + nRefs,
                      [128, 32],
                      activation=nn.Tanh,
                      outputActivation=InvertedRELU).to(device)
    criticTarget = NNCritic(1 + 2 + nRefs,
                            [128, 32],
                            activation=nn.Tanh,
                            outputActivation=InvertedRELU).to(device)
    criticTarget.load_state_dict(critic.state_dict())
    criticTarget.eval()

    criticOptim = torch.optim.Adam(critic.parameters(), lr=c_lr, weight_decay=weightDecay)
    criticLossF = torch.nn.MSELoss()

    # Replay buffer
    replayBuff = GymMemory(buffer_size)

    def update(step: int):

        # Update critic
        states, actions, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)

        criticOptim.zero_grad()
        cInput = torch.cat((actions, states, refs[:, 1:nRefs + 1]), axis=1)
        q = critic(cInput)

        with torch.no_grad():
            next_actions, _ = actorTarget(torch.cat([next_states, refs[:, 2:nRefs + 2]], axis=1))
            next_q = (1 - dones) * criticTarget(torch.cat([next_actions,
                                                           next_states,
                                                           refs[:, 2:nRefs + 2]], axis=1))

        c_loss = criticLossF(q, rewards + discount * next_q)
        c_loss.backward()
        criticOptim.step()

        # Optimize actor
        actorOptim.zero_grad()

        actorInput = torch.cat([states, refs[:, 1:nRefs + 1]], axis=1)
        predActions, _ = actor.act(actorInput, sample=True)
        qs = -critic(torch.cat([predActions, actorInput], axis=1))

        actor_loss = qs.mean()
        actor_loss.backward()
        actorOptim.step()

        # Update target networks
        if step % update_freq == 0:
            for targetP, oP in zip(criticTarget.parameters(), critic.parameters()):
                targetP = (1 - tau) * targetP + tau * oP

            for targetP, oP in zip(actorTarget.parameters(), actor.parameters()):
                targetP = (1 - tau) * targetP + tau * oP

        return c_loss, actor_loss

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
                actorInput = torch.tensor(np.hstack([state, ref[:, 1:nRefs + 1]]), device=device)
                action, _ = actor.act(actorInput, numpy=True, sample=True)
            next_state, reward, done, next_ref = env.step(action)
            total_reward += reward

            states.append(next_state[0, 0])
            refs.append(ref[0, 0])
            actions.append(action[0, 0])

            # Update actor
            replayBuff.add(list(map(lambda x: torch.tensor(x, device=device),
                           [state, action, reward, next_state, int(done), ref])))

            if replayBuff.size >= batch_size:
                critic_loss, actor_loss = update(step)
                total_c_loss += critic_loss
                total_a_loss += actor_loss

            if done:
                break

            state = next_state
            ref = next_ref

            # Plot to see how it looks
        if systemPlots and episode % plot_freq == 0:
            plotData = np.stack((states, refs, actions), axis=-1)
            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref', 'Action'], plotData)

        print(f"Episode {episode}: Reward = {total_reward}, Critic_Loss = {total_c_loss}, Actor_Loss = {total_a_loss}")
        report.log('rewards', total_reward, episode)
        report.log('critic_loss', total_c_loss, episode)
        report.log('actor_loss', total_a_loss, episode)

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
