# Interpreting the 8×8 delayed-reward results

The 8×8 run does **not** show that delay 50 is better than delay 0. Both passed in **10/10 trials**, and both averaged **1.0 return over their final 100 training episodes**. Delay 50 crossed the chart’s pass threshold earlier on average—about episode 3,782 versus 4,101 for delay 0—but that 319-episode difference is small beside the trial-to-trial spread (about 2,356 and 1,659 episodes, respectively). With only ten trials, that is not evidence that delay 50 helps.

## What the chart measures

“Passed” means a trial’s rolling average return reached 0.5 over 100 episodes at least once. Since episode returns are 0 or 1, that means at least 50 successes in some 100-episode window. The chart then keeps counting that trial as passed, even if performance later falls. It is a *first-crossing* chart, not a plot of current performance.

For this run, the final 100-episode returns tell a simple story: delays 0 and 50 both finished at 1.0. The difference is when their noisy training curves first crossed the threshold.

## What delay changes in the MDP

In an MDP, the state needs to contain whatever determines the next state and reward. For the immediate-reward task, the pose `(x, y, direction)` is a reasonable state: the grid is fixed and empty, so that pose tells the learner what actions do.

With a delayed reward, the environment also tracks whether the goal reward is pending and how many steps remain on its timer. The learner receives only the pose from [`PoseEncoder`](../src/delayed_reward/encoding/pose.py), while [`DelayedRewardWrapper`](../src/delayed_reward/environments/delayed_reward.py) keeps that timer internally. So the same pose can mean “no reward pending” or “reward due in 20 steps.” Those situations have different future rewards, but tabular Q-learning assigns them the same Q-values. The learner’s encoded state is therefore no longer Markov. Sutton and Barto explain that value predictions depend on having a state representation that captures information relevant to future rewards and transitions ([*Reinforcement Learning: An Introduction*](https://incompleteideas.net/book/bookdraft2016sep.pdf)).

The wrapper also lets the agent keep acting after reaching the goal while the timer runs. That changes the transition sequence and adds steps to successful episodes. In the final 100 episodes, average episode length was about **15 steps at delay 0** and **66 at delay 50**. The chart compares progress by episode count, even though conditions can involve different numbers of environment steps. If an episode reaches its 256-step limit while a reward is pending, the wrapper delivers it at truncation, so some late rewards arrive sooner than their nominal delay.

## What Q-learning can and cannot tell us here

Tabular Q-learning estimates action values: roughly, the expected discounted future reward from taking an action in a state. Its update uses the immediate reward plus a discounted estimate of the best next action’s value. With γ = 0.99, a reward 50 steps away is discounted by about 0.61; one 100 steps away by about 0.37. A one-step update passes that signal backward through visited transitions over time, so delayed reward can make learning slower and noisier. Watkins and Dayan’s convergence result is for Markovian domains with sufficient repeated sampling, among other conditions ([paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html)). Here we have finite training, a constant learning rate, and—once delay is introduced—a state encoding that omits the timer.

Tabular learning itself is not inherently too large for this grid. The pose representation has at most roughly 144 interior pose combinations. But adding the pending-reward phase and countdown would create many more distinct states. Tabular Q-learning also learns values separately for each state-action pair; it does not generalize from one pose or timer value to another.

There is another source of noise in the comparison: all trials use environment seed 0, while each delay gets different exploration RNG seeds. That means trials vary in their action choices, but the delays are not matched on the same exploration seeds. A lucky sequence can make one delay cross the threshold sooner.

The current result is best read as: **this learner eventually solved both delay 0 and delay 50 in these trials. The experiment does not establish that delay 50 is easier.**

## How to get cleaner comparisons

- Make the timer part of the state, or use a fixed, non-controllable waiting state after goal contact.
- Evaluate a frozen greedy policy separately from training.
- Compare delays across matched trial seeds with more repetitions.
- Report learning against both episodes and environment steps, since delayed successful episodes are longer.
