import click
import torch
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import gym
from .agents import Agent
from .report import Report
from .trainer import Trainer


@click.group()
@click.option('-c', '--config', required=True, type=click.File('r'))
@click.option('--gpu', is_flag=True)
@click.pass_context
def trlfpi(ctx: click.Context, config: click.File, gpu: bool):

    print("Started TestRLFrameworkPleaseIgnore")
    ctx.ensure_object(dict)

    # Load config and create env, agent and report objs
    configDict = load(config, Loader=Loader)
    agent = Agent.create(configDict['agent']['name'], configDict['agent'])
    env = gym.make(configDict['env']['name'], **configDict['env']['args'])
    report = Report(configDict['report']['path'])
    trainer = Trainer(configDict['trainer'])

    # Pass env and agent to subcommand
    ctx.obj['agent'] = agent
    ctx.obj['env'] = env
    ctx.obj['report'] = report
    ctx.obj['trainer'] = trainer

    # Configure torch and setup gpu
    torch.set_default_dtype(torch.double)

    if gpu:
        ctx.obj['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using on {ctx.obj['device']}")
    else:
        ctx.obj['device'] = 'cpu'


@trlfpi.command()
@click.option('-n', type=int, default=1)
@click.pass_context
def train(ctx, n: int):
    agent: Agent = ctx.obj['agent']
    env: gym.Env = ctx.obj['env']
    report: Report = ctx.obj['report']
    trainer: Trainer = ctx.obj['trainer']
    device = ctx.obj['device']

    for i in range(n):
        id = report.new()
        print(f"Training Iteration {id}")
        agent.setup(device=device)
        trainer.train(report, agent, env)
        print(f"Finished Iteration {id} \n\n")

    print("Finish training.")


@trlfpi.command()
@click.option('-n', type=int, default=10)
@click.pass_context
def test(ctx, n):
    agent: Agent = ctx.obj['agent']
    env: gym.Env = ctx.obj['env']
    report: Report = ctx.obj['report']
    trainer: Trainer = ctx.obj['trainer']
    device = ctx.obj['device']

    for id in report.listExisting():
        report.id(id)
        print(f"Testing report {id}")
        agent.setup(checkpoint=report.unpickle('agent_best'), device=device)
        trainer.test(report, agent, env, n)

    print("Finish testing.")
