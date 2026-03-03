# PSnet: Failure to Rescue
https://psnet.ahrq.gov/primer/failure-rescue  

- Defines failure to rescue (FTR) patients as being caused by a failure to Recognize, Relay and React to deteriorating patients

# archiv: Benchmarking deteriorating patients
https://arxiv.org/html/2602.20168  

- Used only the information available in the first hour (like triage data) to predict clinical deterioration
- Only used point in time predictive models to determine if patients would need ICU, weren't looking at all the decisions leading up to it

# archiv: Measurement Scheduling for ICU patients
https://arxiv.org/pdf/2402.07344  

- Potential framework for offline RL with the MIMIC-IV data
- Only used conditions within the ICU

# JMIR Publications
https://www.jmir.org/2020/7/e18477  

- comprehensive review of RL studies in healthcare settings
- Most used MIMIC-II or MIMIC-III (room for newer data)
- Stressed need for meaningful reward design
- RL limited by inability to explore since it's offline, meaning a simulated env for validation could be a good addition
**Future Work**
- Implementing RL in a partial MDP
- Extending the action space to be continuous
- Improving interpretability

# Nature: Optimal antibiotic selection
https://www.nature.com/articles/s41746-024-01350-y#Sec11  

- Main pull here is they specify that quantifying uncertainty in the models predictions is a piece of future work given that RL implementations tend not to do that

# Arxiv: RL Method for heparin treatment for sepsis
https://arxiv.org/pdf/2512.10973

- Used the MIMIC-IV dataset
- Treat their reward score (SOFA) as a continuous target instead of discrete
- Successfully reduced mortality and length of stay
- Utilized distributional RL

# Arxiv: This one is mainly for ideas about LLM implementation
https://arxiv.org/pdf/2509.00974