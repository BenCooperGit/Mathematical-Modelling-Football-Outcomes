import numpy as np
import pandas as pd
import scipy

from utils import _integrate
from models.basemodel import BaseModel


class Model1(BaseModel):
    def __init__(
        self,
        teams_home: list,
        teams_away: list,
        start_minutes: list,
        end_minutes: list,
        goals: list,
        home_scores: list,
        away_scores: list,
        homes: list,
    ):
        super().__init__(
            teams_home,
            teams_away,
            start_minutes,
            end_minutes,
            goals,
            home_scores,
            away_scores,
            homes,
        )

    def lambda_k(
        self,
        t: int,
        alpha_i: float,
        beta_j: float,
        constant: float,
        home_advantage: float,
    ) -> float:
        """
        Scoring intensity for the home team in match k.
        """

        return np.exp(constant + alpha_i + beta_j + home_advantage)

    def mu_k(self, t: int, alpha_j: float, beta_i: float, constant: float) -> float:
        """
        Scoring intensity for the away team in match k.
        """

        return np.exp(constant + alpha_j + beta_i)

    def calculate_log_likelihood(
        self,
        # params
        alphas: list[np.float64],
        betas: list[np.float64],
        constant: float,
        home_advantage: float,
        # model functions,
        lambda_k,
        mu_k,
        # codes
        home_indices: list[int],
        away_indices: list[int],
        # obs data
        start_minutes: list[float],
        end_minutes: list[float],
        goals: list[bool],
        home_scores: list[int],
        away_scores: list[int],
        homes: list[int],
    ):
        num_rows = start_minutes.shape[0]

        log_likelihood = 0.0
        for row_idx in range(num_rows):
            home_idx = home_indices[row_idx]
            away_idx = away_indices[row_idx]

            alpha_i = alphas[home_idx]
            alpha_j = alphas[away_idx]

            beta_i = betas[home_idx]
            beta_j = betas[away_idx]

            log_likelihood += self.calculate_row_log_likelihood(
                # params
                alpha_i,
                beta_i,
                alpha_j,
                beta_j,
                constant,
                home_advantage,
                # model functions,
                lambda_k,
                mu_k,
                # obs data
                start_minutes[row_idx],
                end_minutes[row_idx],
                goals[row_idx],
                home_scores[row_idx],
                away_scores[row_idx],
                homes[row_idx],
            )

        return log_likelihood

    def calculate_match_log_likelihood(
        self,
        # params
        alpha_i: float,
        beta_i: float,
        alpha_j: float,
        beta_j: float,
        constant: float,
        home_advantage: float,
        # model functions,
        lambda_k,
        mu_k,
        # obs data
        start_minutes: list[float],
        end_minutes: list[float],
        goals: list[bool],
        home_scores: list[int],
        away_scores: list[int],
        homes: list[int],
    ) -> float:
        """
        Log likelihood for a match between team i and team j.
        """
        log_likelihood = 0.0
        for match_row_idx in range(len(start_minutes)):
            log_likelihood += self.calculate_row_log_likelihood(  # params
                alpha_i,
                beta_i,
                alpha_j,
                beta_j,
                constant,
                home_advantage,
                # model functions,
                lambda_k,
                mu_k,
                # obs data
                start_minutes[match_row_idx],
                end_minutes[match_row_idx],
                goals[match_row_idx],
                home_scores[match_row_idx],
                away_scores[match_row_idx],
                homes[match_row_idx],
            )
        return log_likelihood

    def calculate_row_log_likelihood(
        self,
        # params
        alpha_i: float,
        beta_i: float,
        alpha_j: float,
        beta_j: float,
        constant: float,
        home_advantage: float,
        # model functions,
        lambda_k,
        mu_k,
        # obs data
        start_minute: float,
        end_minute: float,
        goal: bool,
        home_score: int,
        away_score: int,
        home: int,
    ):
        if goal:
            t = start_minute / 90.0
            J_k_l = home
            return self.calculate_goal_log_likelihood(  # params
                alpha_i,
                beta_i,
                alpha_j,
                beta_j,
                constant,
                home_advantage,
                # model functions,
                lambda_k,
                mu_k,
                # obs data
                t,
                J_k_l,
            )
        else:
            return self.calculate_wait_log_likelihood(
                lambda t: lambda_k(t, alpha_i, beta_j, constant, home_advantage),
                lambda t: mu_k(t, alpha_j, beta_i, constant),
                start_minute / 90.0,
                end_minute / 90.0,
            )

    def calculate_goal_log_likelihood(
        self,
        # params
        alpha_i: float,
        beta_i: float,
        alpha_j: float,
        beta_j: float,
        constant,
        home_advantage: float,
        # model functions,
        lambda_k,
        mu_k,
        # obs data
        t: float,
        J_k_l: float,
    ) -> float:
        return np.log(
            (lambda_k(t, alpha_i, beta_j, constant, home_advantage) ** J_k_l)
            * (mu_k(t, alpha_j, beta_i, constant) ** (1 - J_k_l))
        )

    def calculate_wait_log_likelihood(
        self, lamdba_k_of_t, mu_k_of_t, start_minute: float, end_minute: float
    ) -> float:
        return -_integrate(lamdba_k_of_t, start_minute, end_minute) - _integrate(
            mu_k_of_t, start_minute, end_minute
        )

    def loss_function(self, params) -> float:
        # get params
        alphas = params[: self.n_teams]
        betas = params[self.n_teams : 2 * self.n_teams]
        constant = params[-2]
        home_advantage = params[-1]

        return -self.calculate_log_likelihood(
            # params
            alphas,
            betas,
            constant,
            home_advantage,
            # model functions,
            self.lambda_k,
            self.mu_k,
            # codes
            self.home_indices,
            self.away_indices,
            # obs data
            self.start_minutes,
            self.end_minutes,
            self.goals,
            self.home_scores,
            self.away_scores,
            self.homes,
        )

    def fit(self, init_params: list, callback_function=None):
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams])},
            {"type": "eq", "fun": lambda x: sum(x[self.n_teams : 2 * self.n_teams])},
        ]
        bounds = [(-2, 2)] * (2 * self.n_teams) + [(-2, 1)] + [(-0.5, 1.0)]
        options = {
            "maxiter": 50,
            "disp": True,
        }

        self.res = scipy.optimize.minimize(
            self.loss_function,
            init_params,
            constraints=constraints,
            bounds=bounds,
            options=options,
            callback=callback_function,
        )
        self.save_params()

    def save_params(self):
        self.params = dict(
            zip(
                ["constant", "home_advantage"],
                self.res["x"][-2:],
            )
        )
        self.team_params = pd.DataFrame(
            {
                "team": self.teams,
                "alpha": self.res["x"][: self.n_teams],
                "beta": self.res["x"][self.n_teams : 2 * self.n_teams],
            }
        )
