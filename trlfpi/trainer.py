import gym
import numpy as np
from .report import Report
from .agents import Agent
from functools import reduce


class Trainer():

    default_config = {
        'episodes': 500,
        'maxEpisodeLenght': 1000,
        'evalFreq': 10,
        'plot': False,
        'plotAction': False
    }

    def __init__(self, config: dict):
        self.config = {key: config.get(key, self.default_config[key]) for key in self.default_config}

    def train(self, report: Report, agent: Agent, env: gym.Env):
        self.report = report
        self.agent = agent
        self.env = env

        bestScore = -1e9
        for episode in range(1, self.config['episodes'] + 1):
            agent.train()
            epsReward, epsAgentLoss, states, actions, refs = self.episode()
            self.recordEpisodeScore(episode, epsReward, epsAgentLoss)

            if episode % self.config['evalFreq'] == 0:
                agent.eval()
                epsReward, epsAgentLoss, states, actions, refs = self.episode(eval=True)
                self.recordEpisodeScore(episode, epsReward, epsAgentLoss, eval=True)

                if epsReward > bestScore:
                    bestScore = epsReward
                    report.pickle('agent_best', agent.toDict())

                if self.config['plot']:
                    if self.config['plotAction']:
                        plotData = np.stack((states, refs, actions), axis=-1)
                        self.report.savePlot(f"episode_{episode}_plot",
                                             ['State', 'Ref', 'Actions'],
                                             plotData)
                    else:
                        plotData = np.stack((states, refs), axis=-1)
                        self.report.savePlot(f"episode_{episode}_plot",
                                             ['State', 'Ref'],
                                             plotData)

        self.report.generateReport()
        self.report.pickle('agent_cp', agent.toDict())

    def recordEpisodeScore(self, episode, epsReward, epsAgentLoss, eval=False):
        agentLossStr = reduce(lambda final, current: final + f", {current[0]} = {current[1]}", epsAgentLoss.items(), '')
        if eval:
            print(f"Eval {episode}: Reward = {epsReward}" + agentLossStr)
            self.report.log('eval_reward', epsReward, episode)
            for key in epsAgentLoss:
                self.report.log(f"eval_{key}", epsAgentLoss[key], episode)
        else:
            print(f"Episode {episode}: Reward = {epsReward}" + agentLossStr)
            self.report.log('reward', epsReward, episode)
            for key in epsAgentLoss:
                self.report.log(key, epsAgentLoss[key], episode)

    def episode(self, eval=False):
        state, ref = self.env.reset()
        done = False
        epsReward = 0
        epsAgentLoss: dict = {}
        step = 0

        states = []
        actions = []
        refs = []

        while not done:
            # Take action
            action, log_prob = self.agent.act(state, ref)
            next_state, reward, done, next_ref = self.env.step(action)
            epsReward += reward

            # Save for plotting
            states.append(state[0, 0])
            actions.append(action[0, 0])
            refs.append(ref[0, 0])

            # Update actor and save loss
            if not eval:
                agent_loss = self.agent.update([state,
                                                ref,
                                                action,
                                                log_prob,
                                                reward,
                                                int(done),
                                                next_state])
                for key in agent_loss:
                    epsAgentLoss[key] = agent_loss[key] + epsAgentLoss[key] if key in epsAgentLoss else agent_loss[key]

            if done or step >= self.config['maxEpisodeLenght']:
                break

            state = next_state
            ref = next_ref
            step += 1

        return epsReward, epsAgentLoss, states, actions, refs

    def test(self, report: Report, agent: Agent, env: gym.Env, tests: int = 10):
        self.report = report
        self.agent = agent
        self.env = env

        for i in range(tests):
            agent.eval()
            epsReward, epsAgentLoss, states, actions, refs = self.episode(eval=True)
            print(f"Test {i}: Reward = {epsReward}")
            self.report.log('test_reward', epsReward)

            if self.config['plotAction']:
                plotData = np.stack((states, refs, actions), axis=-1)
                self.report.savePlot(f"test_{i}_plot",
                                     ['State', 'Ref', 'Actions'],
                                     plotData)
            else:
                plotData = np.stack((states, refs), axis=-1)
                self.report.savePlot(f"test_{i}_plot",
                                     ['State', 'Ref'],
                                     plotData)
