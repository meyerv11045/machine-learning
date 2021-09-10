**Reinforcement Learning**- the science of learning to make decisions from interaction

- Differs from other machine learning paradigms in that:
  - No supervision, only a reward signal
  - Feedback can be delayed, not instantaneous
  - Time matters
  - Earlier decisions affect later interactions
- Sequential decision problems can be solved w/ RL
- RL is a framework for how to think about these problems as well as a set of algorithms that can be used to solved these problems 

## Core Concepts:

1. **Environment**
2. **Reward** Signal
   - External to the learning algorithm even if its internal to the learning system as a whole
3. **Agent** containing:
   - Agent state
   - Policy
   - Value function (probably)
   - Model (optionally)

## Rewards

- A reward $R_t$ is a scalar feedback signal
- Indicates how well agent is doing at step *t*
- The agent's job is to *maximize cumulative reward*
  - $G_t = R_{t+1} + R_{t+2} + ...$
- RL is based on the **reward hypothesis**
  - *Any goal can be formalized as the outcome of maximizing a cumulative reward*
- Dense- reward on every step
- Sparse- reward only when the desired event happens
- Can be negative or positive rewads (Carrot vs. Stick in helping train the algorithm)



## Values

- **Value**- the expected cumulative reward from state $s$
  - States use the v function to find their value
- $v(s) = E[G_t | S_t = s]$ where $G_t = R_{t+1} + R_{t+2} + ...$
- Goal is to **maximize value** by picking suitable actions
- Rewards and values define **desirability** of state or acton
- Returns and values can be defined recursively 
  - $G_t = R_{t+1} + G_{t+1}$

## Actions

- Goal: Select actions to maximize value
- Actions may have long term consequences meaning rewards may be delayed
- Might be better to sacrific immediate reward to gain more long-term reward
- Ex:
  - Financial investments might take months to mature
  - Blocking opponent moves might help winning chances many moves in the future
- **Policy**- a mapping from states to actions

## Action Values

- It is possible to condition the value on *actions*
  - $q(s,a) = E[G_t | S_t = s, A_t = a]$
- State action pairs use the Q-function to find their value



## Agent Components

- Agent State
- Policy
- Value Function
- Model

## State

- Actions depend on the state of the agent
- Both agent and environment may have an internal state
  - In the simplest case, there is only one shared state by the environemnt and agent 
- Often there are multiple (sometimes infinite) different states
- The agent's state is usually different than the environment's state
- The agent might not know the full state of the environment

### Environment State

- The environment's internal state
- Usually not visible to the agent
- May contain lots of irrelevant information if visible

### Agent State

- A **history** is a sequence of observations, action, rewards
  - $H_t = O_0,A_0,R_1,O_1,...,O_{t-1},A_{t-1},R_t,O_t$
- The agent state $S_t$ can be constructed from the history and is considerd to be a function of the history
- $S_{t+1} = f(S_t,A_t,R_{t+1},O_{t+1})$ where $f$ is a state update function
  - Means the next agent state is dependent on the current agent state, the action taken, the reward, and the observation
  - A simple update function can be concatinating the previous states together to create the current state (Done w/ the Atari game agents by cocatenating frames together since frames are the state)
- Actions depend on this state
- Agent state is typically much smaller than the environment state and the full history

### Fully Observable Environments

- When the agent sees the full environment state
- observation = environment state
- Agent state could just be the observation
  - $S_t = O_t = environment state$
- Then the agent is in a *Markov Decision Process*
- Ex: 
  - Single player board games in which the player can see the entire board 
  - Smart Pong is an example of fully observable environment (VERIFY)

## Markov Decision Processes

- MDPs provide a useful mathematical framework for talking about many RL topics 
  - Easier to reason about than the full problem (long markovian)
  - Limited b/c of markov assumption
- Def- A decision process is Markov if the future is independent of the past given the present 
  - $H_t => S_t => H_{t+1}$
  - Once the current state ($S_t$) is known , the history ($H_t$) may be thrown away (useful for space consideration)
  - The current state gives enough information for the next state to be predicted
- The environment state is typically Markov
- The history is Markov 

### Partially Observable Environments

- The agent gets parial information
  - A robot w/ camera vision doesn't have its absolute location
  - A poker playing agent only observes public cards
- Now the observation is not Markov 
- Formally this is a *partially observable Markov decision process* (POMDP)
- The environment state can still be markov, but the agent does not know it
- The state should contain enough information for good policies/value predictions

## Policy

- Defines the agent's behavior
- A map from agent state to action
- *Deterministic Policy*- function that outputs an action based on an input agent state
- *Stochastic Policy*- a probability of selecting each action in each state

## Value Function:

- The actual value function is the expected return

  $V_\pi(s)=E[G_t|S_t=s,\pi]$

  ​		$ = E[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+... | S_t = s,\pi]$

- The return has a recursive form since $G_t = R_{t+1}+\gamma G_{t+1}$

- *Bellman Equation*: $V_\pi (s) = E[R_{t+1} + \gamma G_{t+1} | S_t = s,A_t \sim \pi (s)]$ where $a \sim \pi (s)$ means action $a$ is chosen by policy $\pi$ in state $s$ 

- *Discount Factor* ($\gamma \in [0,1]$) trades off importance of immediate vs. long-term rewards

  - Closer to 1 means higher importane on immediate rewards (AKA discounting future rewards in favor of immediate rewards)

- The value depends on the policy

- Can be used to evaluate desirability of states

- Can be used to select btw actions

- 



