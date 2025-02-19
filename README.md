
# CausalRivers  
#### Scaling up benchmarking of causal discovery for real-world time-series



[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This is the official Repo of [CausalRivers](https://openreview.net/forum?id=wmV4cIbgl6), the largest real-world Causal Discovery benchmark for time series to this date.


<img src="graphics/teaser.webp" width="300"><img src="graphics/teaser2.webp" width="300">




### Install
For the core benchmarking package simply run (uses conda, unzip and wget):
```
./install.sh
python 0_generate_datsets.py
```
Alternatively, you can execute the following commands: 
```
conda create -f benchmark_env.yml
wget https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip
unzip product
rm product.zip
python 0_generate_datsets.py
```



## Functionality:

This is the core benchmarking package, which only holds the core functionality and some tutorials on usage: 

- Tutorial on how to build your graph subset:  [Custom graph sampling](1_custom_graph_sampling.ipynb)
- Tutorial on how to use the benchmark most efficiently:  [Usage](2_tutorial_benchmarking.ipynb)
- Tutorial on how to subselect specific temporal windows with certain weather conditions: [Temporal selections](3_tutorial_subselect_weather_condition.ipynb)
- Some general display of dataset properties that might be interesting for users:  [Data distribution](4_data_distribution.ipynb)
- Graphical documentation as in our [Publication](https://openreview.net/pdf?id=wmV4cIbgl6):  [Graphics generation](graphics)


## Usage
```
DESCRIBE THE HYDRA STUFF FOR GENERATING
Usecase if ready.
```

If you want to reproduce the experimental results further or compare your method under equal conditions, please clone: 
```
git clone https://github.com/CausalRivers/experiments
```

The experiments were conducted on a Slurm Cluster and via Hydra configurations. However, the script can also be used on a single machine.
We forward to the [Experimental Documentation](https://github.com/CausalRivers/benchmark/blob/main/experiments/README.md) for further information.


## High prio todos: 

- Script for data downloading if called (and generally fix install)
- Slim down the tools.
- Properly update the tutorials
- Tutorial on the standard usage (to equalize)


### Maintainers
[@GideonStein](https://github.com/Gideon-Stein).
[@Timozen](https://github.com/Timozen).


## Contributors
This project exists thanks to the generous provision of data by the following German institutions: 

 [THÜRINGER LANDESAMT FÜR UMWELT, BERGBAU UND NATURSCHUTZ](https://tlubn.thueringen.de/)
  Landesbetrieb für Hochwasserschutz und Wasserwirtschaft Sachsen-Anhalt https://gld.lhw-sachsen-anhalt.de/
Bayerisches Landesamt für Umwelt  https://www.hnd.bayern.de/

Sächsisches Landesamt für Umwelt, Landwirtschaft und Geologie https://www.umwelt.sachsen.de/umwelt/infosysteme/lhwz/index.html
 Landesamt für Umwelt, Naturschutz und Geologie Mecklenburg-Vorpommern https://www.lung.mv-regierung.de/

Senatsverwaltung für Mobilität, Verkehr, Klimaschutz und Umwelt https://wasserportal.berlin.de/start.php>

