import numpy as np
import random

from .constants import CLEANUP_MAP
from .map import Map, ACTIONS
from .agent import CleanupAgent  # CLEANUP_VIEW_SIZE

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing beam
ACTIONS['CLEAN'] = 5  # length of cleanup beam

# Custom colour dictionary
CLEANUP_COLORS = {'C': [100, 255, 255],  # Cyan cleaning beam
                  'S': [113, 75, 24],  # Light grey-blue stream cell
                  'H': [99, 156, 194],  # brown waste cells
                  'R': [113, 75, 24]}  # Light grey-blue river cell

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


class CleanupEnv(Map):

    def __init__(self, args, ascii_map=CLEANUP_MAP):
        super().__init__(args, ascii_map)

        # compute potential waste area
        #print("cleanup render:")
        # print(render)
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(
            'H', 0) + counts_dict.get('R', 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == 'B':
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == 'S':
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == 'H':
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] == 'H' or self.base_map[row, col] == 'R':
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == 'R':
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)
        self.waste_cleared = 0

    @property
    def action_space(self):
        action_space = []
        for agent in self.agents.values():
            action_space.append(agent.action_space)
        return action_space

    @property
    def observation_space(self):
        observation_space = []
        for agent in self.agents.values():
            observation_space.append(agent.observation_space)
        return observation_space

    @property
    def share_observation_space(self):
        share_observation_space = []
        for agent in self.agents.values():
            channel_dim = agent.observation_space[0]
            space = [channel_dim * self.num_agents,
                     agent.observation_space[1], agent.observation_space[2]]
            share_observation_space.append(space)
        return share_observation_space

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.world_map[waste_start_point[0], waste_start_point[1]] = 'H'
        for river_point in self.river_points:
            self.world_map[river_point[0], river_point[1]] = 'R'
        for stream_point in self.stream_points:
            self.world_map[stream_point[0], stream_point[1]] = 'S'
        self.compute_probabilities()
        self.waste_cleared = 0

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == 'FIRE':
            agent.fire_beam('F')
            agent.fire += 1
            updates = self.update_map_fire(agent.get_pos().tolist(),
                                           agent.get_orientation(
            ), ACTIONS['FIRE'],
                fire_char='F')
        elif action == 'CLEAN':
            agent.fire_beam('C')
            updates = self.update_map_fire(agent.get_pos().tolist(),
                                           agent.get_orientation(),
                                           ACTIONS['CLEAN'],
                                           fire_char='C',
                                           cell_types=['H'],
                                           update_char=['R'],
                                           blocking_cells=['H'])
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(agent_id, spawn_point,
                                 rotation, map_with_agents)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, 'A'))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != 'H':
                    rand_num = np.random.rand(1)[0]
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, 'H'))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (1 - (waste_density - thresholdRestoration)
                              / (thresholdDepletion - thresholdRestoration)) \
                    * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get('H', 0)
        free_area = self.potential_waste_area - current_area
        return free_area
