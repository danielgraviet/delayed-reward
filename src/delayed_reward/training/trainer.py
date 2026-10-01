"""Training loop orchestration."""

from __future__ import annotations

from delayed_reward.protocols import Agent, Environment, EpisodeObserver


class Trainer:
    """Run episodic interaction between an Environment and an Agent."""

    def __init__(
        self,
        env: Environment,
        agent: Agent,
        observers: list[EpisodeObserver] | None = None,
    ) -> None:
        self._env = env
        self._agent = agent
        self._observers = list(observers) if observers is not None else []

    def train(self, n_episodes: int) -> None:
        for episode in range(n_episodes):
            state = self._env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action = self._agent.act(state)
                transition = self._env.step(action)
                self._agent.observe(
                    state,
                    action,
                    transition.reward,
                    transition.state,
                    transition.terminated,
                    transition.truncated,
                )
                total_reward += transition.reward
                steps += 1
                state = transition.state

                if transition.terminated or transition.truncated:
                    break

            for observer in self._observers:
                observer.on_episode_end(episode, total_reward, steps)
