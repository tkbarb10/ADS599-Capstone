# Table: RL Agent Design Iterations

| Iteration | Algorithm | Action Space | Reward Structure | Observed Failure | Change Applied |
|---|---|---|---|---|---|
| 1 | DQN | 9 actions | Per-step penalty + correct terminal reward + mortality penalty | Discharge predicted at every timestep; mortality penalty suppressed ICU Q-values since ICU patients more likely to die in hospital | Remove mortality penalty; switch to CQL |
| 2 | CQL | 9 actions | Per-step penalty + correct terminal reward | Action distribution improved; still heavily biased toward discharge at terminal step | Add class weight to reward |
| 3 | CQL | 9 actions | + ICU class weight in reward | ~87% of observed actions were "observe," dominating training signal; concurrent actions not representable; escalation purpose diluted | Trim action space to 3 (wait, discharge, transfer) |
| 4 | CQL | 3 actions | Class-weighted terminal reward | ~50% terminal commitment rate; F1 = 0.49, Recall = 0.33; ICU class still underweighted in Q-values | Add explicit ICU transfer reward weight |
| 5 | CQL | 3 actions | + Explicit ICU transfer weight | Recall = 1.0, F1 = 0.58; agent overpredicted ICU transfer | Use LSTM p_icu step-by-step predictions to weight reward signal |
| 6 (Final) | CQL | 3 actions | LSTM p_icu-weighted terminal reward | 46.3% commit rate; ICU commit rate (1.4%) below true ICU rate (7.9%); conservative behavior at terminal rows | Accepted as inherent tradeoff of offline RL conservatism |