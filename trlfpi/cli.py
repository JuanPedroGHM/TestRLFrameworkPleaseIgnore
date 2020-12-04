import click
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
    ctx.obj['gpu'] = gpu


@trlfpi.command()
@click.option('-n', type=int, default=1)
@click.pass_context
def train(ctx, n: int):
    print('TRLFPI started training.')
    agent: Agent = ctx.obj['agent']
    env: gym.Env = ctx.obj['env']
    report: Report = ctx.obj['report']
    trainer: Trainer = ctx.obj['trainer']
    gpu = ctx.obj['gpu']

    print("Started TRFLPI train")

    for i in range(n):
        print(f"Training Iteration {i}")
        report.new()
        agent.setup(gpu=gpu)
        trainer.train(report, agent, env)
        print(f"Finished Iteration {i} \n\n")


@trlfpi.command()
@click.pass_context
def eval():
    print('Eval!')
