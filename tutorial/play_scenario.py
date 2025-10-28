from vmas import make_env, render_interactively

# Use the interactive render for debugging (runs a single env)
print("Starting Interactive Render (Press R to reset, keys to control)")
render_interactively(
    scenario='football',
    n_blue_agents=2, # can change number of controlled agents
    ai_red_agents=2 # passed but not used by the scenario
)