# Exploring Information Asymmetry in MARL Football
This project implements a Multi-Agent Proximal Policy Optimization (MAPPO) framework to train agents in a football environment using TorchRL and VMAS.

This project also explores the role of how information asymmetry and policy composition affects behavioural diversity and the emergent roles that occur i.e. how agents adapt their strategies (e.g. learning to defend by proxy) when critical sensory data, such as ball position or opponent visibility, is masked based on spatial or proximity constraints.


## Information Asymmetry Reference Table

| Flag | Category | Logic Description |
| :--- | :--- | :--- |
| **`mask_pitch_lhs`** | **Spatial** | Agent is "blind" to external objects when its own $X$-coordinate is e.g. $< 0.0$ (Left Half) of the pitch. |
| **`mask_pitch_rhs`** | **Spatial** | Agent is "blind" to external objects when its own $X$-coordinate is e.g. $> 0.0$ (Right Half) of the pitch. |
| **`mask_pitch_bhs`** | **Spatial** | Agent is "blind" to external objects when its own $Y$-coordinate is e.g. $< 0.0$ (Bottom Half) of the pitch. |
| **`mask_pitch_ths`** | **Spatial** | Agent is "blind" to external objects when its own $Y$-coordinate is e.g. $> 0.0$ (Right Half) of the pitch. |
| **`mask_ball`** | **Sensory** | Masks ball information e.g. position and velocity from agent during training. |
| **`mask_opponent`** | **Sensory** | Masks opponent information e.g. position and velocity from agent during training. |
| **`mask_ball_by_distance`** | **Proximity** | Ball information is masked dependant on `DISTANCE_THRESHOLD` and the distance between the agent and the ball. |
| **`mask_opponent_by_distance`** | **Proximity** | Opponent information is masked dependant on `DISTANCE_THRESHOLD` and the distance between the agent and the ball. |
| **`mask_if_far`** | **Logic** | **True**: Masking occurs when distance is **LARGE** (Standard). <br>**False**: Masking occurs when distance is **SMALL** (Inverted). <br>This flag has to be set to the appropriate value if masking by distance. |


### Example configuration

```python
asymmetries = {
    "mask_pitch_lhs": False,            # Spatial awareness of left hand side of pitch
    "mask_pitch_rhs": False,            # Spatial awareness of right hand side of pitch
    "mask_pitch_bhs": False,            # Spatial awareness of bottom side of pitch
    "mask_pitch_ths": False,            # Spatial awareness of top side of pitch
    "mask_ball": False,                 # Ball information is available
    "mask_opponent": False,             # opponent information is available
    "mask_ball_by_distance": True,      # ball information will be masked depending on the agent's distance to ball
    "mask_opponent_by_distance": False, # opponent information is always available
    "mask_if_far": True                 # ball information will be masked if the agent is far away from the ball
}
```