import datetime
import unittest

from satplan.visualizer import Visualizer
import os
import numpy as np
from tqdm import tqdm

class TestVisualizer(unittest.TestCase): 
    
    def setUp(self) -> None:
        super().setUp()

        # clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')

        # set parameters
        self.data_dir = './tests/test_data/'
        self.output_dir = './tests/test_results/'
        self.initial_datetime = datetime.datetime(2020,1,1,0,0,0)
        self.timestep = 20
        self.duration = 0.5
        self.clear = False

        self.visualizer = Visualizer(
                                        self.data_dir,
                                        self.output_dir,
                                        self.initial_datetime,
                                        self.timestep,
                                        self.duration
                                    )

    def test_init(self) -> None:
        """ Tests constructor """
        self.assertEqual(self.data_dir, self.visualizer.data_dir)
        self.assertEqual(self.output_dir, self.visualizer.output_dir)
        self.assertEqual(self.initial_datetime, self.visualizer.initial_datetime)
        self.assertEqual(self.timestep, self.visualizer.timestep)
        self.assertEqual(self.duration, self.visualizer.duration)
        
    def test_processing(self) -> None:
        """ Processes test data and checks if outputs """

        # clear processed data folder (if it already exists)
        if self.clear:
            if os.path.exists(self.output_dir):
                # for f in os.listdir(self.output_dir):
                for f in tqdm(
                            os.listdir(self.output_dir), 
                            desc="Clearing Data Folder files", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir, f)):
                        for h in tqdm(
                            os.listdir(self.output_dir + f), 
                            desc=f"Deleting files in `{self.output_dir}/{f}`", 
                            unit="files",
                            leave=False
                            ):
                            os.remove(os.path.join(self.output_dir, f, h))
                        os.rmdir(self.output_dir + f)

                    else:
                        os.remove(os.path.join(self.output_dir, f)) 
            else:
                os.mkdir(self.output_dir)

        # process data
        self.visualizer.process_mission_data()

        # check data correctness
        dir_names = ['sat_positions', 
                     'sat_visibilities',
                     'sat_observations',
                     'constellation_past_observations',
                     'ground_swaths',
                     'crosslinks']

        steps = np.arange(0,self.duration*24*3600,self.timestep,dtype=int)

        for dir_name in dir_names:
            # Check output exists
            dir_path = self.output_dir+f"/{dir_name}"
            self.assertTrue(os.path.exists(dir_path))

            # Check contents
            self.assertEqual(len(os.listdir(dir_path)), len(steps))

        # attempt to process data again (should not run again)
        self.visualizer.process_mission_data()

        # check data correctness
        for dir_name in dir_names:
            # Check output exists
            dir_path = self.output_dir+f"/{dir_name}"
            self.assertTrue(os.path.exists(dir_path))

            # Check contents
            self.assertEqual(len(os.listdir(dir_path)), len(steps))

    def test_plot(self) -> None:
        """ Processes test data and checks if outputs """

        self.visualizer.animate()

if __name__ == '__main__':
    unittest.main()