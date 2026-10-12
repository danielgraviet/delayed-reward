# delayed-reward

This project measures how much delay between reaching the goal and receiving
reward tabular Q-learning can tolerate. Each delay is run across independent
trials to see whether the learner still discovers a successful policy.

Every run's `delay_survival.png` shows, by training episode, the percentage of
trials at each delay that have passed. A trial passes when its rolling mean
return reaches 0.5 over 100 episodes. The chart makes it easy to compare both
how quickly each delay is learned and how many trials learn it by the end.
