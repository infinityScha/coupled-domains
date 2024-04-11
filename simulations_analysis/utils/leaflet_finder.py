import numpy as np
import MDAnalysis as mda
from lipyphilic import AssignCurvedLeaflets
from MDAnalysis.lib.log import ProgressBar

class AssignCurvedLeafletsWrapper():
    """
      Wrapper for AssignCurvedLeaflets class from lipyphilic package. Intended to change the run method to be more efficient over short runs
    """
    def __init__(self, universe, lipid_sel, midplane_sel, midplane_cutoff, lf_cutoff):
        self.assigner = AssignCurvedLeaflets(universe=universe, lipid_sel=lipid_sel, midplane_sel=midplane_sel, midplane_cutoff=midplane_cutoff, lf_cutoff=lf_cutoff)

    def run(self, start=None, stop=None, step=None, frames=None,
            verbose=None, *, progressbar_kwargs={}):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        frames : array_like, optional
            array of integers or booleans to slice trajectory; `frames` can
            only be used *instead* of `start`, `stop`, and `step`. Setting
            *both* `frames` and at least one of `start`, `stop`, `step` to a
            non-default value will raise a :exc:`ValueError`.

            .. versionadded:: 2.2.0

        verbose : bool, optional
            Turn on verbosity
        
        progressbar_kwargs : dict, optional
            ProgressBar keywords with custom parameters regarding progress bar position, etc; 
            see :class:`MDAnalysis.lib.log.ProgressBar` for full list.


        .. versionchanged:: 2.2.0
            Added ability to analyze arbitrary frames by passing a list of
            frame indices in the `frames` keyword argument.

        .. versionchanged:: 2.5.0
            Add `progressbar_kwargs` parameter, 
            allowing to modify description, position etc of tqdm progressbars
        """
        # if verbose unchanged, use class default
        verbose = getattr(self.assigner, '_verbose',
                          False) if verbose is None else verbose

        self.assigner._setup_frames(self.assigner._trajectory, start=start, stop=stop,
                           step=step, frames=frames)
        self.assigner._prepare()
        
        for i, ts in enumerate(ProgressBar(
                self.assigner._sliced_trajectory,
                verbose=verbose,
                **progressbar_kwargs)):
            self.assigner._frame_index = i
            self.assigner._ts = ts
            self.assigner.frames[i] = ts.frame
            self.assigner.times[i] = ts.time
            self.assigner._single_frame()
        self.assigner._conclude()
        return self

    @property
    def leaflets(self):
        return self.assigner.results.leaflets