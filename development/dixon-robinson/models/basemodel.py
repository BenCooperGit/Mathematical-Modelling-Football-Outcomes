import numpy as np


class BaseModel:
    def __init__(
        self,
        teams_home,
        teams_away,
        start_minutes: list,
        end_minutes: list,
        goals: list,
        home_scores: list,
        away_scores: list,
        homes: list,
    ):
        self.teams_home = teams_home
        self.teams_away = teams_away
        self.start_minutes = start_minutes
        self.end_minutes = end_minutes
        self.goals = goals
        self.home_scores = home_scores
        self.away_scores = away_scores
        self.homes = homes
        self.setup_teams()

    def setup_teams(self):
        self.teams = np.sort(
            np.unique(np.concatenate([self.teams_home, self.teams_away]))
        )
        self.n_teams = len(self.teams)
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        self.home_indices = np.array(
            [self.team_to_idx[t] for t in self.teams_home], dtype=np.int64, order="C"
        )
        self.away_indices = np.array(
            [self.team_to_idx[t] for t in self.teams_away], dtype=np.int64, order="C"
        )
