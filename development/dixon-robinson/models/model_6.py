import numpy as np
import pandas as pd
import scipy

from utils import _integrate
from models.basemodel import BaseModel


class Model6(BaseModel):
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
        epsilon_1: float,
        injury_time_1: float,
        injury_time_2: float,
        lambda_10: float,
        lambda_01: float,
        lambda_11: float,
        lambda_22: float,
        lambda_21: float,
        lambda_12: float,
        home_score: int,
        away_score: int,
    ) -> float:
        """
        Scoring intensity for the home team in match k.
        """

        return (
            np.exp(
                constant
                + alpha_i
                + beta_j
                + home_advantage
                + injury_time_1 * (44 / 90 < t <= 45 / 90)
                + injury_time_2 * (89 / 90 < t <= 90 / 90)
                + lambda_10 * ((home_score == 1) and (away_score == 0))
                + lambda_01 * ((home_score == 0) and (away_score == 1))
                + lambda_11 * ((home_score == 1) and (away_score == 1))
                + lambda_22
                * (
                    (home_score >= 2)
                    and (away_score >= 2)
                    and (home_score - away_score == 0)
                )
                + lambda_21 * ((home_score >= 2) and (home_score - away_score >= 1))
                + lambda_12 * ((away_score >= 2) and (home_score - away_score <= -1))
            )
            + np.exp(epsilon_1) * t
        )

    def mu_k(
        self,
        t: int,
        alpha_j: float,
        beta_i: float,
        constant: float,
        epsilon_2: float,
        injury_time_1: float,
        injury_time_2: float,
        mu_10: float,
        mu_01: float,
        mu_11: float,
        mu_22: float,
        mu_21: float,
        mu_12: float,
        home_score: int,
        away_score: int,
    ) -> float:
        """
        Scoring intensity for the away team in match k.
        """

        return (
            np.exp(
                constant
                + alpha_j
                + beta_i
                + injury_time_1 * (44 / 90 < t <= 45 / 90)
                + injury_time_2 * (89 / 90 < t <= 90 / 90)
                + mu_10 * ((home_score == 1) and (away_score == 0))
                + mu_01 * ((home_score == 0) and (away_score == 1))
                + mu_11 * ((home_score == 1) and (away_score == 1))
                + mu_22
                * (
                    (home_score >= 2)
                    and (away_score >= 2)
                    and (home_score - away_score == 0)
                )
                + mu_21 * ((home_score >= 2) and (home_score - away_score >= 1))
                + mu_12 * ((away_score >= 2) and (home_score - away_score <= -1))
            )
            + np.exp(epsilon_2) * t
        )

    def calculate_wait_log_likelihood(
        self, lamdba_k_of_t, mu_k_of_t, t_start_minute: float, t_end_minute: float
    ) -> float:
        return -_integrate(lamdba_k_of_t, t_start_minute, t_end_minute) - _integrate(
            mu_k_of_t, t_start_minute, t_end_minute
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
        epsilon_1: float,
        epsilon_2: float,
        injury_time_1: float,
        injury_time_2: float,
        lambda_10: float,
        lambda_01: float,
        lambda_11: float,
        lambda_22: float,
        lambda_21: float,
        lambda_12: float,
        mu_10: float,
        mu_01: float,
        mu_11: float,
        mu_22: float,
        mu_21: float,
        mu_12: float,
        # model functions,
        lambda_k,
        mu_k,
        # obs data
        t: float,
        J_k_l: float,
        home_score: int,
        away_score: int,
    ) -> float:
        return np.log(
            (
                lambda_k(
                    t,
                    alpha_i=alpha_i,
                    beta_j=beta_j,
                    constant=constant,
                    home_advantage=home_advantage,
                    epsilon_1=epsilon_1,
                    injury_time_1=injury_time_1,
                    injury_time_2=injury_time_2,
                    lambda_10=lambda_10,
                    lambda_01=lambda_01,
                    lambda_11=lambda_11,
                    lambda_22=lambda_22,
                    lambda_21=lambda_21,
                    lambda_12=lambda_12,
                    home_score=home_score,
                    away_score=away_score,
                )
                ** J_k_l
            )
            * (
                mu_k(
                    t=t,
                    alpha_j=alpha_j,
                    beta_i=beta_i,
                    constant=constant,
                    epsilon_2=epsilon_2,
                    injury_time_1=injury_time_1,
                    injury_time_2=injury_time_2,
                    mu_10=mu_10,
                    mu_01=mu_01,
                    mu_11=mu_11,
                    mu_22=mu_22,
                    mu_21=mu_21,
                    mu_12=mu_12,
                    home_score=home_score,
                    away_score=away_score,
                )
                ** (1 - J_k_l)
            )
        )

    def calculate_row_log_likelihood(
        self,
        # params
        alpha_i: float,
        beta_i: float,
        alpha_j: float,
        beta_j: float,
        constant,
        home_advantage: float,
        epsilon_1: float,
        epsilon_2: float,
        injury_time_1: float,
        injury_time_2: float,
        lambda_10: float,
        lambda_01: float,
        lambda_11: float,
        lambda_22: float,
        lambda_21: float,
        lambda_12: float,
        mu_10: float,
        mu_01: float,
        mu_11: float,
        mu_22: float,
        mu_21: float,
        mu_12: float,
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
            return self.calculate_goal_log_likelihood(
                # params
                alpha_i=alpha_i,
                beta_i=beta_i,
                alpha_j=alpha_j,
                beta_j=beta_j,
                constant=constant,
                home_advantage=home_advantage,
                epsilon_1=epsilon_1,
                epsilon_2=epsilon_2,
                injury_time_1=injury_time_1,
                injury_time_2=injury_time_2,
                lambda_10=lambda_10,
                lambda_01=lambda_01,
                lambda_11=lambda_11,
                lambda_22=lambda_22,
                lambda_21=lambda_21,
                lambda_12=lambda_12,
                mu_10=mu_10,
                mu_01=mu_01,
                mu_11=mu_11,
                mu_22=mu_22,
                mu_21=mu_21,
                mu_12=mu_12,
                # model functions,
                lambda_k=lambda_k,
                mu_k=mu_k,
                # obs data
                t=t,
                J_k_l=J_k_l,
                home_score=home_score,
                away_score=away_score,
            )
        else:
            return self.calculate_wait_log_likelihood(
                lambda t: lambda_k(
                    t=t,
                    alpha_i=alpha_i,
                    beta_j=beta_j,
                    constant=constant,
                    home_advantage=home_advantage,
                    epsilon_1=epsilon_1,
                    injury_time_1=injury_time_1,
                    injury_time_2=injury_time_2,
                    lambda_10=lambda_10,
                    lambda_01=lambda_01,
                    lambda_11=lambda_11,
                    lambda_22=lambda_22,
                    lambda_21=lambda_21,
                    lambda_12=lambda_12,
                    home_score=home_score,
                    away_score=away_score,
                ),
                lambda t: mu_k(
                    t=t,
                    alpha_j=alpha_j,
                    beta_i=beta_i,
                    constant=constant,
                    epsilon_2=epsilon_2,
                    injury_time_1=injury_time_1,
                    injury_time_2=injury_time_2,
                    mu_10=mu_10,
                    mu_01=mu_01,
                    mu_11=mu_11,
                    mu_22=mu_22,
                    mu_21=mu_21,
                    mu_12=mu_12,
                    home_score=home_score,
                    away_score=away_score,
                ),
                start_minute / 90.0,
                end_minute / 90.0,
            )

    def calculate_log_likelihood(
        self,
        # params
        alphas: list[np.float64],
        betas: list[np.float64],
        constant: float,
        home_advantage: float,
        epsilon_1: float,
        epsilon_2: float,
        injury_time_1: float,
        injury_time_2: float,
        lambda_10: float,
        lambda_01: float,
        lambda_11: float,
        lambda_22: float,
        lambda_21: float,
        lambda_12: float,
        mu_10: float,
        mu_01: float,
        mu_11: float,
        mu_22: float,
        mu_21: float,
        mu_12: float,
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
                alpha_i=alpha_i,
                beta_i=beta_i,
                alpha_j=alpha_j,
                beta_j=beta_j,
                constant=constant,
                home_advantage=home_advantage,
                epsilon_1=epsilon_1,
                epsilon_2=epsilon_2,
                injury_time_1=injury_time_1,
                injury_time_2=injury_time_2,
                lambda_10=lambda_10,
                lambda_01=lambda_01,
                lambda_11=lambda_11,
                lambda_22=lambda_22,
                lambda_21=lambda_21,
                lambda_12=lambda_12,
                mu_10=mu_10,
                mu_01=mu_01,
                mu_11=mu_11,
                mu_22=mu_22,
                mu_21=mu_21,
                mu_12=mu_12,
                # model functions,
                lambda_k=lambda_k,
                mu_k=mu_k,
                # obs data
                start_minute=start_minutes[row_idx],
                end_minute=end_minutes[row_idx],
                goal=goals[row_idx],
                home_score=home_scores[row_idx],
                away_score=away_scores[row_idx],
                home=homes[row_idx],
            )

        return log_likelihood

    def loss_function(self, params) -> float:
        # get params
        alphas = params[: self.n_teams]
        betas = params[self.n_teams : 2 * self.n_teams]
        constant = params[2 * self.n_teams]
        home_advantage = params[2 * self.n_teams + 1]
        injury_time_1 = params[2 * self.n_teams + 2]
        injury_time_2 = params[2 * self.n_teams + 3]
        lambda_10 = params[2 * self.n_teams + 4]
        lambda_01 = params[2 * self.n_teams + 5]
        lambda_11 = params[2 * self.n_teams + 6]
        lambda_22 = params[2 * self.n_teams + 7]
        lambda_21 = params[2 * self.n_teams + 8]
        lambda_12 = params[2 * self.n_teams + 9]
        mu_10 = params[2 * self.n_teams + 10]
        mu_01 = params[2 * self.n_teams + 11]
        mu_11 = params[2 * self.n_teams + 12]
        mu_22 = params[2 * self.n_teams + 13]
        mu_21 = params[2 * self.n_teams + 14]
        mu_12 = params[2 * self.n_teams + 15]
        epsilon_1 = params[2 * self.n_teams + 16]
        epsilon_2 = params[2 * self.n_teams + 17]

        return -self.calculate_log_likelihood(
            # params
            alphas=alphas,
            betas=betas,
            constant=constant,
            home_advantage=home_advantage,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            injury_time_1=injury_time_1,
            injury_time_2=injury_time_2,
            lambda_10=lambda_10,
            lambda_01=lambda_01,
            lambda_11=lambda_11,
            lambda_22=lambda_22,
            lambda_21=lambda_21,
            lambda_12=lambda_12,
            mu_10=mu_10,
            mu_01=mu_01,
            mu_11=mu_11,
            mu_22=mu_22,
            mu_21=mu_21,
            mu_12=mu_12,
            # model functions,
            lambda_k=self.lambda_k,
            mu_k=self.mu_k,
            # codes
            home_indices=self.home_indices,
            away_indices=self.away_indices,
            # obs data
            start_minutes=self.start_minutes,
            end_minutes=self.end_minutes,
            goals=self.goals,
            home_scores=self.home_scores,
            away_scores=self.away_scores,
            homes=self.homes,
        )

    def fit(self, init_params, callback_function=None):
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams])},
            {"type": "eq", "fun": lambda x: sum(x[self.n_teams : 2 * self.n_teams])},
        ]
        bounds = [(-2, 2)] * (2 * self.n_teams) + [
            (-2, 1),  # constant
            (-0.5, 1.0),  # home advantage
            (-0.5, 2.0),  # injury_time_1
            (-0.5, 2.0),  # injury_time_1
            (-0.5, 0.5),  # lambda_10
            (-0.5, 0.5),  # lambda_01
            (-0.5, 0.5),  # lambda_11
            (-0.5, 0.5),  # lambda_22
            (-0.5, 0.5),  # lambda_21
            (-0.5, 0.5),  # lambda_12
            (-0.5, 0.5),  # mu_10
            (-0.5, 0.5),  # mu_01
            (-0.5, 0.5),  # mu_11
            (-0.5, 0.5),  # mu_22
            (-0.5, 0.5),  # mu_21
            (-0.5, 0.5),  # mu_12
            (-20, 1),  # epsilon_1
            (-20, 1),  # epsilon_2
            # (-2, 2),  # epsilon_1
            # (-2, 2),  # epsilon_2
        ]
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
                [
                    "constant",
                    "home_advantage",
                    "injury_time_1",
                    "injury_time_2",
                    "lambda_10",
                    "lambda_01",
                    "lambda_11",
                    "lambda_22",
                    "lambda_21",
                    "lambda_12",
                    "mu_10",
                    "mu_01",
                    "mu_11",
                    "mu_22",
                    "mu_21",
                    "mu_12",
                    "epsilon_1",
                    "epsilon_2",
                ],
                self.res["x"][2 * self.n_teams :],
            )
        )
        self.team_params = pd.DataFrame(
            {
                "team": self.teams,
                "alpha": self.res["x"][: self.n_teams],
                "beta": self.res["x"][self.n_teams : 2 * self.n_teams],
            }
        )
