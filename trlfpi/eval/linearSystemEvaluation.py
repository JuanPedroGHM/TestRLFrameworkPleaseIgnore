import argparse
import torch
from torch import nn
import numpy as np

from trlfpi.report import Report
from trlfpi.nn.actor import NNActor

import gym

torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentName")
    parser.add_argument("reportId")
    parser.add_argument("--nTests", type=int, default=10)
    parser.add_argument("--episodeLength", type=int, default=1000)
    parser.add_argument("--plots", action='store_true')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    args = parser.parse_args()

    report = Report(args.experimentName, args.reportId)
    trainingArgs = report.getArgs()
    nRefs = trainingArgs['nRefs']
    modelParams = report.unpickle("actor_best")

    actor = NNActor(2 + nRefs, 1, [64, 8],
                    activation=nn.Tanh,
                    outputActivation=nn.Identity).to(device)
    actor.load_state_dict(modelParams)
    actor.eval()

    results = []

    env = gym.make("linear-with-ref-v0")

    with torch.no_grad():
        for i in range(args.nTests):
            state, ref = env.reset()
            total_reward = 0

            states = []
            actions = []
            refs = []

            for step in range(args.episodeLength):

                actorInput = torch.tensor(np.hstack([state, ref[:, 1:nRefs + 1]]), device=device)
                action, _ = actor.act(actorInput, numpy=True, sample=True)

                next_state, reward, done, next_ref = env.step(action)
                total_reward += reward

                # Save for plotting
                states.append(state[0, 0])
                actions.append(action[0, 0])
                refs.append(ref[0, 0])

                if done:
                    break

                state = next_state
                ref = next_ref

            # Plot to see how it looks
            if args.plots:
                plotData = np.stack((states, refs, actions), axis=-1)
                report.savePlot(f"eval_{i}_plot",
                                ['State', 'Ref', 'Actions'],
                                plotData)

            print(f"Episode {i}: Reward = {total_reward}")
            results.append(total_reward)

    report.pickle('evalResults', {"rewards": results, "meanReward": np.array(results).mean()})
