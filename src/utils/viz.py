from typing import List, Tuple, Optional, Dict

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import datetime

from src.utils.logger import logger
import scienceplots  # noqa

plt.style.use(["science", "no-latex"])

def save_fig(
    fig: matplotlib.figure.Figure,
    fig_name: str,
    fig_dir: str,
    fig_fmt: str,
    fig_size: Tuple[float, float] = [6.4, 4],
    save: bool = True,
    dpi: int = 300,
    transparent_png=True,
):
    """
    Code adapted from https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/
    This procedure stores the generated matplotlib figure to the specified
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4]
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return

    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(fig_dir, "{}.{}".format(fig_name, fig_fmt.lower()))
    if fig_fmt == "pdf":
        metadata = {"Creator": "", "Producer": "", "CreationDate": None}
        fig.savefig(pth, bbox_inches="tight", metadata=metadata)
    elif fig_fmt == "png":
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth,
            bbox_inches="tight",
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches="tight")
        except Exception as e:
            print("Cannot save figure: {}".format(e))


class Plotter:
    def __init__(self, exp_name: str, viz: bool = True, res_path: str = None, fig_fmt="pdf") -> None:
        self.viz = viz

        # Plotting options
        self.figure_size = (6.4, 4.0)
        # Create a directory to store the results if viz is False
        script_directory = os.path.abspath(os.curdir)
        self.res_path = res_path
        if self.res_path is None:
            self.res_path = os.path.join(script_directory, "assets", datetime.datetime.now().isoformat(), exp_name)
        logger.info(f"Saving results to {self.res_path}")
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        # Savefig partial
        self.savefig = partial(
            save_fig, fig_dir=self.res_path, fig_fmt=fig_fmt, fig_size=self.figure_size, transparent_png=True
        )

    def plot_observations(
        self,
        sampling_times: List[float],
        true_state: List[float],
        t: List[float],
        obs: List[int],
        location: str = "lower left",
        figure_size: Optional[Tuple[int, int]] = None,
        f_name: str = "observations",
    ) -> None:
        """
        Plot binary observations against the true state over time.

        Args:
            sampling_times (List[float]): Time points for the true state samples.
            true_state (List[float]): True state values at sampling times.
            t (List[float]): Time points for the observations.
            obs (List[int]): Binary observations at times `t`.
            location (str, optional): Legend location in the plot. Default is "lower left".
            figure_size (Optional[Tuple[int, int]], optional): Figure size in inches. Defaults to `self.figure_size`.
            f_name (str, optional): File name for saving the plot. Default is "observations".

        Returns:
            None
        """

        # Set figure size
        figure_size = figure_size or self.figure_size

        # Define color palette
        colors = {
            "true_state": "#9e9e9e",
            "observation": "#FF2C00",
        }

        # Create figure
        fig, ax = plt.subplots(figsize=figure_size)

        # Plot true state
        ax.plot(
            sampling_times, true_state,
            color=colors["true_state"], ls="--", linewidth=1,
            label="True state"
        )

        # Plot observations
        ax.plot(
            t, obs,
            color=colors["observation"], linestyle="None",
            marker=".", markersize=6, label="Observation"
        )

        # Labels and legend
        ax.set_xlabel("Time (t)")
        ax.set_title("Sampled Observations")
        ax.legend(loc=location, numpoints=1, frameon=False)

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True)

        plt.close()

    def plot_change_points(
        self,
        obs_times: np.ndarray,
        obs: np.ndarray,
        change_points: np.ndarray,
        gt_change_points: np.ndarray,
        location: str = "lower left",
        figure_size: Optional[Tuple[int, int]] = None,
        f_name: str = "change_points",
    ) -> None:

        if figure_size is None:
            figure_size = self.figure_size

        fig = plt.figure(figsize=figure_size)
        plt.plot(
            obs_times,
            obs,
            "#FF2C00",
            linestyle="None",
            # alpha=0.6,
            marker=".",
            markersize=6,
            label="Observation",
        )

        # Add change points with a single label
        label_added = False  # Flag to track if the label has been added
        for cp in change_points[:-1]:
            if not label_added:
                plt.axvline(x=obs_times[cp], color="#0C5DA5", linestyle=":", label="Change Point")
                label_added = True
            else:
                plt.axvline(x=obs_times[cp], color="#0C5DA5", linestyle=":")  # No label

        # Add change points with a single label
        label_added = False
        for cp in gt_change_points:
            if not label_added:
                plt.axvline(x=cp, color="#9e9e9e", linestyle="--", label="GT Change Point")
                label_added = True
            else:
                plt.axvline(x=cp, color="#9e9e9e", linestyle="--")  # No label

        plt.xlabel("Time (t)")
        plt.ylabel("Observation")
        plt.legend(loc=location, numpoints=1, frameon=False)
        plt.title("Change Point Detection Results")

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True)
        plt.close()

    def plot_loss_curve(
        self,
        iterations: List[int],
        loglikelihoods: List[float],
        title: str = "Loss Curve",
        f_name: str = "loss_curve.png",
    ) -> None:
        """
        Plot the log-likelihood values against the number of iterations.

        Args:
            iterations (List[int]): The number of iterations.
            loglikelihoods (List[float]): The log-likelihood values.
            title (str, optional): The title of the plot. Default is "Loss Curve".
            f_name (str, optional): File name for saving the plot. Default is "loss_curve.png".

        Returns:
            None
        """

        # Define color palette
        colors = {
            "loss_curve": "#9467bd"
        }

        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot log-likelihood values
        ax.plot(iterations, loglikelihoods, lw=2, color=colors["loss_curve"])

        # Labels and title
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title(title)

        # Adjust layout and save/display the plot
        plt.tight_layout()

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True)

        plt.close()

    def plot_mixture_filter(
        self,
        sampling_times: np.ndarray,
        true_state: np.ndarray,
        t: np.ndarray,
        obs: np.ndarray,
        query_times: np.ndarray,
        belief: np.ndarray,
        location: str = "lower left",
        figure_size: Tuple[int, int] = (6.4, 4),
        f_name: str = "mixture_death_filters",
    ) -> None:
        """
        Plot the results of a mixture filter, showing the true state, observations, and filter beliefs over time.

        Args:
            sampling_times (np.ndarray): The times at which the true state was sampled.
            true_state (np.ndarray): The true state values corresponding to the sampling times.
            t (np.ndarray): The times at which observations were made.
            obs (np.ndarray): The binary observations made at times `t`.
            query_times (np.ndarray): The times at which beliefs were queried.
            belief (List[np.ndarray]): The belief values from the mixture filter for each component and the marginal posterior.
            location (str): The location of the legend in the plot.
            figure_size (Tuple[int, int]): The size of the figure in inches. Defaults to (6.4, 4).
            f_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Set figure size
        figure_size = figure_size or self.figure_size

        # Number of belief components
        n_components = len(belief)

        # Create figure and axes
        fig, axes = plt.subplots(n_components + 1, 1, figsize=figure_size, sharex=True)

        # Define color palette
        colors = {
            "true_state": "#9e9e9e",
            "observation": "#FF2C00",
            "belief_component": "#9467bd",
            "best_belief": "#0C5DA5",
        }

        # Plot true state and observations
        axes[0].plot(sampling_times, true_state, color=colors["true_state"], ls="--", linewidth=1, label="True state")
        axes[0].plot(
            t, obs, color=colors["observation"], linestyle="None", marker=".", markersize=6, label="Observation"
        )

        axes[0].legend(loc=location, numpoints=1, frameon=False)
        axes[0].set_ylabel("State")

        # Plot conditional posterior beliefs
        for i in range(n_components - 1):
            axes[i + 1].plot(
                query_times,
                belief[i],
                color=colors["belief_component"],
                ls="--",
                lw=2,
                label=rf"$p(X_t = 1 \mid Y_{{1:N}}, C = {i})$",
            )
            axes[i + 1].set_ylabel("Belief")
            axes[i + 1].legend(loc=location, numpoints=1, frameon=False)

        # Plot the marginal posterior belief
        axes[-1].plot(
            query_times,
            belief[-1],
            color=colors["best_belief"],
            lw=2,
            label=rf"$p(X_t = 1 \vert C_{{K^*}}, Y_{{1:N}})$",
        )

        axes[-1].set_ylabel("Belief")
        axes[-1].set_xlabel("Time (t)")
        axes[-1].legend(loc=location, numpoints=1, frameon=False)

        # Adjust layout and save or display the figure
        plt.tight_layout()

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True)

        plt.close()

    def plot_perpetua(
        self,
        sampling_times: np.ndarray,
        true_state: np.ndarray,
        t: np.ndarray,
        obs: np.ndarray,
        query_times: np.ndarray,
        belief: np.ndarray,
        weights: np.ndarray,
        location: str = "lower left",
        interval: Tuple[int, int] = (0, np.inf),
        figure_size: Tuple[int, int] = (6.4, 4),
        f_name: str = "perpetua_results",
    ) -> None:
        """
        Create a figure with multiple vertically arranged subplots to visualize the results of the Perpetua model.

        Args:
            sampling_times (np.ndarray): The times at which the true state was sampled.
            true_state (np.ndarray): The true state values corresponding to the sampling times.
            t (np.ndarray): The times at which observations were made.
            obs (np.ndarray): The binary observations made at times `t`.
            query_times (np.ndarray): The times at which beliefs were queried.
            belief (List[np.ndarray]): The belief values from the Perpetua model for each component and the heaviest-weighted component.
            weights (np.ndarray): The weights of each component over time.
            location (str): The location of the legend in the plot.
            interval (Tuple[int, int]): The time interval to plot.
            marker_size (int): The size of the markers for observations.
            figure_size (Optional[Tuple[int, int]]): The size of the figure in inches. If None, defaults to the instance's figure size.
            f_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Set figure size
        figure_size = figure_size or self.figure_size

        # Number of belief components + final combined plot
        n_components = len(belief)
        n_axes = n_components + 1

        # Create figure and axes
        fig, axes = plt.subplots(n_axes, 1, figsize=figure_size, sharex=True)

        # Define color palette
        colors = {
            "true_state": "#9e9e9e",
            "observation": "#FF2C00",
            "belief_component": "#00B945",
            "weights": "k",
            "heaviest_weighted": "#0C5DA5"
        }

        # Select indices within the given time interval
        indices_gt = np.logical_and(sampling_times > interval[0], sampling_times <= interval[1])
        indices_obs = np.logical_and(t > interval[0], t <= interval[1])
        indices_query = np.logical_and(query_times > interval[0], query_times <= interval[1])

        # Plot true state and observations on the first subplot
        axes[0].plot(sampling_times[indices_gt], true_state[indices_gt], color=colors["true_state"], ls="--", linewidth=1, label="True state")
        axes[0].plot(t[indices_obs], obs[indices_obs], color=colors["observation"], linestyle="None", marker=".", markersize=6, label="Observation")
        axes[0].legend(loc=location, numpoints=1, frameon=False)
        axes[0].set_ylim([0, 1.05])
        axes[0].set_ylabel("State")

        # Plot conditional posterior beliefs and weights
        for i in range(n_components - 1):
            ax_idx = i + 1
            axes[ax_idx].plot(query_times[indices_query], belief[i][indices_query], color=colors["belief_component"], ls="-", lw=1, label=f"Combination {i}")
            axes[ax_idx].plot(query_times[indices_query], weights[i][indices_query], color=colors["weights"], ls="--", label=rf"$w_{i}$")
            axes[ax_idx].legend(loc=location, numpoints=1, frameon=False)
            axes[ax_idx].set_ylim([0, 1.05])
            axes[ax_idx].set_ylabel("Belief")

        # Plot the heaviest-weighted component
        axes[-1].plot(sampling_times[indices_gt], true_state[indices_gt], color=colors["true_state"], ls="--", linewidth=1, label="True state")
        axes[-1].plot(query_times[indices_query], belief[-1][indices_query], color=colors["heaviest_weighted"], lw=1, label="Heaviest-weighted component")
        axes[-1].legend(loc=location, numpoints=1, frameon=False)
        axes[-1].set_ylim([0, 1.05])
        axes[-1].set_xlabel("Time (t)")
        axes[-1].set_ylabel("Belief")

        # Adjust layout and save/display the figure
        plt.tight_layout()

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False)
            plt.show()
        else:
            self.savefig(fig, fig_name=f_name, save=True)

        plt.close()

    def plot_room_environment(
        self,
        observations_dict: Dict[int, Dict[str, np.ndarray]],
        figure_size: Tuple[int, int] = None,
        interval: Tuple[int, int] = (0, -1),
        x_label: str = "Time (s)",
        f_name: str = "all_mixture_death_filters",
    ) -> None:
        if figure_size is None:
            figure_size = self.figure_size
        # Create a figure with three vertically arranged subplots
        n_landmarks = len(observations_dict)
        fig, axes = plt.subplots(n_landmarks, 1, figsize=figure_size, sharex=True)
        plt.subplots_adjust(hspace=1.0)

        for l in range(n_landmarks):
            indices_gt = np.logical_and(
                observations_dict[l]["gt_t"] > interval[0], observations_dict[l]["gt_t"] <= interval[1]
            )
            indices_obs = np.logical_and(
                observations_dict[l]["obs_t"] > interval[0], observations_dict[l]["obs_t"] <= interval[1]
            )
            axes[l].plot(
                observations_dict[l]["gt_t"][indices_gt],
                observations_dict[l]["gt_obs"][indices_gt],
                "#9e9e9e",
                ls="--",
                linewidth=1,
                label="True state",
            )
            axes[l].plot(
                observations_dict[l]["obs_t"][indices_obs],
                observations_dict[l]["obs"][indices_obs],
                "#FF2C00",
                linestyle="None",
                marker=".",
                markersize=2,
                label="Observation",
            )
            axes[l].set_ylabel("Obs")
            axes[l].set_title(f"Landmark {l}")

        axes[-1].set_xlabel(x_label)
        # Add a single legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))

        if self.viz:
            self.savefig(fig, fig_name=f_name, save=False)
            plt.show()
        else:
            # Store the figure with a large DPI
            self.savefig(fig, fig_name=f_name, save=True)
        plt.close()