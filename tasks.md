# Remaining tasks for stand-alone plug-in probabilistic evaluation

### Framework tasks
- [ ] Define license (e.g. MIT, Apache 2.0, GPLv3) !!
- [ ] Define core classes and data structures based on OpenTPS
    - [ ] Scenario
    - [ ] UncertaintyModel
    - [ ] ProbabilisticEvaluator
    - [ ] ClinicalGoal
- [ ] Implement data I/O functions
    - [ ] DICOM-RT import/export via OpenTPS
    - [ ] JSON configuration files for user-friendly API
- [ ] Implement visualization functions
- [ ] Write documentation and usage examples
- [ ] Write pytest
- [ ] Package for PyPI distribution
- [ ] Set up pipeline for continuous integration (e.g. GitHub Actions)
- [ ] Explore integration with treatment planning systems (e.g. Eclipse, RayStation)
- [ ] Explore web-based GUI options (e.g. Dash, Streamlit)

### Clinical evaluation tasks
#### Passing rate tables
- [ ] Boolean evaluation of DVH metrics for a given scenario
    - [ ] Dose metrics
    - [ ] Volume metrics
- [ ] Distribution of DVH metrics over all scenarios
    - [ ] Dose metrics
    - [ ] Volume metrics
- [ ] Probability of achieving clinical goals
    - [ ] Dose metrics
    - [ ] Volume metrics
#### DVH bands and DEVH
- [ ] Implement DVH bands based on the OpenTPS framework
- [ ] Implement DEVH based on Buti et al. 2019
#### Dose maps
- [ ] Plot dose maps for given scenarios
- [ ] Plot dose maps with vw_min and vw_max

### Scenario generation tasks
#### Rigid deformations
- [ ] Implement translations based on a given vector 
- [ ] Implement efficient translation (e.g. modify the origin of the ROIstructures)
- [ ] Implement scenario reading from files (e.g. user provides a lists of CTs and structs with corresponding probabilities)
#### Non-rigid deformations
- [ ] Implement scenario generation based on deformation vector fields (DVFs)
- [ ] Implement scenario reading from files (e.g. user provides a lists of CTs and structs with corresponding probabilities)
#### Fractionation
- [ ] Implement dose accumulation over multiple fractions


### Uncertainty models
- [ ] Implement MC simulation of translation vectors
- [ ] Implement efficient scenario sampling (e.g. Voronoi sampling)
- [ ] Allow for per-axis uncertainty specification
- [ ] Implement uncertainty models for:
    - [ ] Random errors
    - [ ] Systematic errors
    - [ ] Combined errors
    - [ ] Other models (e.g. breathing, ...)
- [ ] Provide standard uncertainty model parameters based on literature
    - [ ] Head and Neck
    - [ ] Prostate
    - [ ] Lung
    - [ ] Other treatment sites
