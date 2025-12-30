# Exploring Information Asymmetry in MARL Football
This project implements a Multi-Agent Proximal Policy Optimization (MAPPO) framework to train agents in a football environment using TorchRL and VMAS.

This project also explores the role of how information asymmetry and policy composition affects behavioural diversity and the emergent roles that occur i.e. how agents adapt their strategies (e.g. learning to defend by proxy) when critical sensory data, such as ball position or opponent visibility, is masked based on spatial or proximity constraints.


## Information Asymmetry Reference Table

| Flag | Category | Logic Description |
| :--- | :--- | :--- |
| `mask_pitch_lhs` | **Spatial** | Agent is "blind" to external objects when its own $X$-coordinate is e.g. $< 0.0$ (Left Half) of the pitch. |
| `mask_pitch_rhs` | **Spatial** | Agent is "blind" to external objects when its own $X$-coordinate is e.g. $> 0.0$ (Right Half) of the pitch. |
| `mask_pitch_bhs` | **Spatial** | Agent is "blind" to external objects when its own $Y$-coordinate is e.g. $< 0.0$ (Bottom Half) of the pitch. |
| `mask_pitch_ths` | **Spatial** | Agent is "blind" to external objects when its own $Y$-coordinate is e.g. $> 0.0$ (Right Half) of the pitch. |
| `mask_ball` | **Sensory** | Masks ball information e.g. position and velocity from agent during training. |
| `mask_opponent` | **Sensory** | Masks opponent information e.g. position and velocity from agent during training. |
| `mask_ball_by_distance` | **Proximity** | Ball information is masked dependant on `DISTANCE_THRESHOLD` and the distance between the agent and the ball. |
| `mask_opponent_by_distance` | **Proximity** | Opponent information is masked dependant on `DISTANCE_THRESHOLD` and the distance between the agent and the ball. |
| `mask_if_far` | **Logic** | **True**: Masking occurs when distance is **LARGE** (Standard). <br>**False**: Masking occurs when distance is **SMALL** (Inverted). <br>This flag has to be set to the appropriate value if masking by distance. |


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

## Resulting Behaviours

| Flags | Experiment | Agent behaviour | Visual Demonstration |
| :--- | :--- | :--- | :--- |
| `None` | Agent has full observability, used as a baseline. | Learns how to score towards the end of training. | ![Demo](./videos/baseline.gif) |
| `mask_pitch_rhs` | The offensive right hand side of the pitch is masked i.e. agent will not receive any information regarding the ball and opposing players when in that zone. | Learns a 'goalkeeper' role, i.e. stays close to its own goal and defends the ball. | ![Demo](./videos/mask_rhs.gif) |
| `mask_pitch_lhs` | The defensive area (left 1/6th) of the pitch is masked i.e. the agent will not receive any information regarding the ball and opposing players when in that zone. | The agent learns defensive behaviour in the middle of the pitch. It doesn't score but prevents the other agent from scoring by intercepting the ball in the midfield, similar to a midfielder role and a hybrid between the goalkeeper and baseline behaviour. | ![Demo](./videos/mask_lhs.gif) |
| `mask_ball` | Agent does not receive ball information in its observation. | Learns how to predict the trajectory of the ball via 'perpendicular bumping' near its own goal. It travels perpendicular to the opponent to repeatedly intercept the ball when it is being dribbled by the opponent. | ![Demo](./videos/mask_ball.gif) |
| `mask_opponent` | Agent does not receive information regarding the opponent. | Learns aggressive offensive behaviour and disregards the opponent by going straight towards the ball and attempting to score in the opposing goal, possibly being tackled by the opponent in the process. | ![Demo](./videos/mask_oppo.gif) |
| `mask_ball_by_distance, mask_if_far` | Agent does not receive any ball information in its observation when it is close to the ball. | Compensates by learning to repeatedly tackle the opponent to prevent it from getting the ball. | ![Demo](./videos/) |
| `mask_ball_by_distance, mask_if_far` | Agent does not receive any ball information in its observation when it is far away from the ball. | Learns normal offensive behaviour to score and quickly heads towards the ball. | ![Demo](./videos/) |
| `mask_opponenet_by_distance, mask_if_far` | Agent does not receive any opponent information in its obervation when it is far away from the ball. | Learns offensive behaviour to score. | ![Demo](./videos/) |
| `mask_pitch_bhs` | The top third of the pitch is masked i.e. agents will not receive any information regarding the ball and opposing players when in that zone. | tbc. |
| `mask_pitch_ths` | The bottom third of the pitch is masked i.e. agents will not receive any information regarding the ball and opposing players when in that zone. | tbc. |