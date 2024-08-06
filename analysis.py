import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm

class AnalysisBase(ABC):
    """
    Base class for analysis.

    Methods:
        run(profile): Abstract method to run the analysis over a profile.
        plot(): Abstract method to plot the analysis results.
        results(): Abstract method to return the analysis results.
    """
    def __init__(self):
        pass

    @abstractmethod
    def run(self, profile):
        """
        Abstract method to run the analysis on the current profile.

        Args:
            profile (pandas.DataFrame): The data for analysis.        
        """
        pass

    @abstractmethod
    def results(self):
        """
        Abstract method to return the analysis results.
        """
        pass


class PropertyAnalysis(AnalysisBase):
    """
    Class for one property-per-profile analysis.

    Methods:
        run(): Calculates and adds the property to an array.
        save(): Save the array to a file.
    """
    def __init__(self):
        super().__init__()
        self.property_arr = np.array([])

    def _calculate_property(self, profile):
        """
        Calculate the property for the given profile, return the result.

        Args:
            profile (pandas.DataFrame): The data for analysis.
        
        Returns:
            float: The property value.
        """
        pass

    def run(self, profile):
        """
        Calculates the property.

        Args:
            profile (pandas.DataFrame): The data for analysis.
        """
        self.property_arr = np.append(self.property_arr, self._calculate_property(profile))


    def results(self):
        return self.property_arr


def run_analyses(filenames, analyses):
    """
    Run multiple analyses on the given data.

    Args:
        filenames (list): List of filenames.
        analyses (list): List of analysis objects.

    Returns:
        None
    """
    # run the analyses on the profiles
    for filename in tqdm(filenames):
        profile = pd.read_csv(filename, dtype=float)
        for analysis in analyses:
            analysis.run(profile)

class Registration(PropertyAnalysis):
    """
    Class for registration analysis.

    Methods:
        _calculate_property(): Calculate the registration property.
    """
    def __init__(self):
        super().__init__()

    def _calculate_property(self, profile):
        """
        Calculate the registration of the upper and lower leaflets.

        Args:
            profile (pandas.DataFrame): The data for analysis.
        
        Returns:
            float: The registration value.
        """
        cu = profile['cᵤ'].values
        cl = profile['cₗ'].values
        return 1.0-0.5*np.mean(np.abs(cu - cl))
    
class InterfaceWidth(PropertyAnalysis):
    """
    Class for interface width analysis.

    Methods:
        _calculate_property(): Calculate the interface width property.
    """
    def __init__(self, is_upper = True):
        super().__init__()
        self.is_upper = is_upper

    def periodic_distance(self, x, x0, L):
        """
        Calculate the distance between 2 points periodically, with direction.

        Args:
            x (numpy.array): The x values.
            x0 (float): The interface position.

        Returns:
            numpy.array: The distance values.
        """
        dist_vec = x - x0
        dist_vec = np.where(dist_vec > L/2, dist_vec - L, dist_vec)
        dist_vec = np.where(dist_vec < -L/2, dist_vec + L, dist_vec)
        return dist_vec

    def _calculate_property(self, profile):
        """
        Calculate the interface width.

        Args:
            profile (pandas.DataFrame): The data for analysis.
        
        Returns:
            float: The interface width value.
        """
        x = profile['x'].values
        c = profile['cᵤ'].values if self.is_upper else profile['cₗ'].values
        # find the peak from first fourier mode
        w = np.exp(-2j*np.pi*np.arange(len(c)) / len(c))
        f1 = c.dot(w)
        # get the angle
        angle = np.angle(f1.conj())
        if angle < 0:
            angle += 2*np.pi
        peak = int(len(x) * angle / (2*np.pi))
        # shift the data so that the peak would be at the center
        c_shifted = np.roll(c, -peak)
        # fit both sides to tanh
        from scipy.optimize import curve_fit
        def tanh(x, b, c, d, e):
            return b*np.tanh((x-d)/(np.sqrt(2)*c)) - e
        # 1st half
        x1 = x[:len(x)//2]
        c1 = c_shifted[:len(c_shifted)//2]
        fitted_width = -1
        try:
            popt1, _ = curve_fit(tanh, x1, c1, p0=[-1.0, 2.0, 0.25*x[-1], 0.0])
            # 2nd half
            x2 = x[len(x)//2:]
            c2 = c_shifted[len(c_shifted)//2:]
            popt2, _ = curve_fit(tanh, x2, c2, p0=[1.0, 2.0, 0.75*x[-1], 0.0])
            # calculate the interface width
            fitted_width = 0.5*(np.abs(popt1[1]) + np.abs(popt2[1]))
            import matplotlib.pyplot as plt
            plt.plot(x1, c1)
            plt.plot(x2, c2)
            plt.plot(x1, tanh(x1, *popt1))
            plt.plot(x2, tanh(x2, *popt2))
            plt.savefig('test.png')
            plt.close()
        except:
            pass
        return fitted_width