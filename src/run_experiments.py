from run_experiment import run_experiment

ffor_levels = [120,80,60,40,20]
ffov_levels = [20,10,5,2,1]
constellation_size_levels = [10,8,6,4,2]
agility_levels = [10,5,1,0.1,0.01]
event_duration_levels = [24*3600,12*3600,6*3600,3*3600,1.5*3600]
event_frequency_levels = [1/3600,0.1/3600,0.01/3600,0.001/3600,1e-4/3600]
event_density_levels = [1,5,10,50,100]
event_clustering_levels = [1,2,4,8,16]

default_settings = {
    "name": "experiment_test",
    "ffor": 60,
    "ffov": 5,
    "constellation_size": 6,
    "agility": 1,
    "event_duration": 6*3600,
    "event_frequency": 0.01/3600,
    "event_density": 10,
    "event_clustering": 4
}

run_experiment(default_settings)