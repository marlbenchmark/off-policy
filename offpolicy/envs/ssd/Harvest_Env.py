import numpy as np

from .agent import HarvestAgent  # HARVEST_VIEW_SIZE
from .constants import HARVEST_MAP
from .map import Map, ACTIONS

APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]


class HarvestEnv(Map):

    def __init__(self, args, ascii_map=HARVEST_MAP):
        super().__init__(args, ascii_map)
        self.num_agents = args.num_agents
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

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

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid)
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         HARVEST_VIEW_SIZE, HARVEST_VIEW_SIZE)
            # agent = HarvestAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        agent.fire += 1
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       ACTIONS['FIRE'], fire_char='F')
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j**2 + k**2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples
