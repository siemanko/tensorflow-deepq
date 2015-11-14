import time

from IPython.display import clear_output, display, HTML
from os.path import join, exists
from os import makedirs

from tf_rl.utils.event_queue import EventQueue

def simulate(game,
             controller,
             fps=60,
             actions_per_game_second=60,
             simulation_resultion=0.001,
             speed=1.0,
             store_every_nth=5,
             train_every_nth=5,
             save_path=None):
    """Start the simulation. Performs three tasks

        - visualizes simulation in iPython notebook
        - advances game simulator state
        - reports state to controller and chooses actions
          to be performed.
    """
    eq = EventQueue()

    time_between_frames  = 1.0 / fps
    game_time_between_actions = 1.0 / actions_per_game_second

    simulation_resultion /= speed

    vis_s = {
        'last_image': 0
    }

    if save_path is not None:
        if not exists(save_path):
            makedirs(save_path)

    ###### VISUALIZATION
    def visualize():
        recent_reward = game.collected_rewards[-100:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in game.objects_eaten.items()])
        clear_output(wait=True)
        svg_html = game.to_html([
            "DTW        = %.1f" % (game.distance_to_walls(),),
            "experience = %d" % (len(controller.experience),),
            "reward = %.1f" % (sum(recent_reward)/len(recent_reward),),
            "objects eaten => %s" % (objects_eaten_str,),
        ])
        display(svg_html)
        if save_path is not None:
            img_path = join(save_path, "%d.svg" % (vis_s['last_image'],))
            with open(img_path, "w") as f:
                svg_html.write_svg(f)
            vis_s['last_image'] += 1

    eq.schedule_recurring(visualize, time_between_frames)


    ###### CONTROL
    ctrl_s = {
        'last_observation': None,
        'last_action':      None,
        'actions_so_far':   0,
    }

    def control():
        # sense
        new_observation = game.observe()
        reward          = game.collect_reward()
        # store last transition
        ctrl_s['actions_so_far'] += 1
        if ctrl_s['last_observation'] is not None and ctrl_s['actions_so_far'] % store_every_nth == 0:
            controller.store(ctrl_s['last_observation'], ctrl_s['last_action'], reward, new_observation)
        # act
        new_action = controller.action(new_observation)
        game.perform_action(new_action)
        ctrl_s['last_action'] = new_action
        ctrl_s['last_observation'] = new_observation

        #train
        if  ctrl_s['last_observation'] is not None and ctrl_s['actions_so_far'] % train_every_nth == 0:
            controller.training_step()

    ##### SIMULATION
    sim_s = {
        'simulated_up_to':             time.time(),
        'game_time_since_last_action': 0,
    }
    def simulate_game():
        while sim_s['simulated_up_to'] < time.time():
            game.step(simulation_resultion)
            sim_s['simulated_up_to'] += simulation_resultion / speed
            sim_s['game_time_since_last_action'] += simulation_resultion
            if sim_s['game_time_since_last_action'] > game_time_between_actions:
                control()
                sim_s['game_time_since_last_action'] = 0

    eq.schedule_recurring(simulate_game, time_between_frames)

    eq.run()
