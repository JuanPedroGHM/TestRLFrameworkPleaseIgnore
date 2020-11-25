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
    parser.add_argument("report", type=str)

    # Env params
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--aCost", type=float, default=1e-3)

    # Agent params
    parser.add_argument("--a_lr", type=float, default=1e-5)
    parser.add_argument("--discount", type=float, default=0.7)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--weightDecay", type=float, default=1e-3)
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # SETUP ARGUMENTS
    args = parser.parse_args()
    discount = args.discount
    a_lr = args.a_lr
    episodes = args.episodes
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    update_freq = args.update_freq
    weightDecay = args.weightDecay
    nRefs = args.nRefs
    aCost = args.aCost
    systemPlots = args.plots
    plot_freq = args.plot_freq

    # Report
    report = Report(args.report)
    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env, horizon=nRefs, deltaActionCost=aCost)

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

    replayBuff = GymMemory(buffer_size)

    def updateActor():

        states, actions, log_probs, rewards, next_states, dones, refs = replayBuff.get(batchSize=batch_size)

        # Optimize actor
        actorOptim.zero_grad()

        actorInput = torch.cat((states, refs[:, 1:nRefs + 1]), axis=1)
        pActions, c_log_probs = actor.act(actorInput, sample=False, prevActions=actions)

        # Importance Weight
        iw = torch.exp(c_log_probs - log_probs).detach() + 1e-9
        adv = rewards - rewards.mean()

        actor_loss = (-c_log_probs * adv * iw).mean()
        actor_loss.backward()
        actorOptim.step()

        return actor_loss.detach().cpu().numpy()

    bestScore = -1e6

    for episode in range(1, episodes + 1):
        state, ref = env.reset()
        done = False
        total_reward = 0
        total_actor_loss = 0
        step = 0

        states = []
        actions = []
        refs = []

        while not done:
            # Take action
            with torch.no_grad():
                # Pad ref at the end of the episode
                actorInput = torch.tensor(np.hstack([state, ref[:, 1:nRefs + 1]]), device=device)
                action, log_prob = actorTarget.act(actorInput, numpy=True, sample=True)

            next_state, reward, done, next_ref = env.step(action)
            total_reward += reward

            # Save for plotting
            states.append(state[0, 0])
            actions.append(action[0, 0])
            refs.append(ref[0, 0])

            # Update actor
            replayBuff.add(list(map(lambda x: torch.tensor(x, device=device),
                           [state, action, log_prob, reward, next_state, int(done), ref])))

            if replayBuff.size >= batch_size:
                actor_loss = updateActor()
                total_actor_loss += actor_loss

            if done:
                break

            state = next_state
            ref = next_ref
            step += 1

        # Plot to see how it looks
        if systemPlots and episode % plot_freq == 0:
            plotData = np.stack((states, refs, actions), axis=-1)
            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref', 'Actions'],
                            plotData)

        print(f"Episode {episode}: Reward = {total_reward}, Actor Loss = {total_actor_loss}")
        report.log('rewards', total_reward, episode)
        report.log('actorLoss', total_actor_loss, episode)

        if total_reward > bestScore:
            bestScore = total_reward
            report.pickle('actor_best', actorTarget.state_dict())

        if episode % update_freq == 0:
            actorTarget.load_state_dict(actor.state_dict())

    report.generateReport()
    report.pickle('actor_cp', actor.state_dict())
