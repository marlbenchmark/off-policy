"""Base map class that defines the rendering process
"""

import random

import matplotlib.pyplot as plt
import numpy as np


ACTIONS = {'MOVE_LEFT': [-1, 0],  # Move left
           'MOVE_RIGHT': [1, 0],  # Move right
           'MOVE_UP': [0, -1],  # Move up
           'MOVE_DOWN': [0, 1],  # Move down
           'STAY': [0, 0],  # don't move
           'TURN_CLOCKWISE': [[0, -1], [1, 0]],  # Rotate counter clockwise
           'TURN_COUNTERCLOCKWISE': [[0, 1], [-1, 0]]}  # Move right

ORIENTATIONS = {'LEFT': [-1, 0],
                'RIGHT': [1, 0],
                'UP': [0, -1],
                'DOWN': [0, 1]}

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

# the axes look like
# graphic is here to help me get my head in order
# WARNING: increasing array position in the direction of down
# so for example if you move_left when facing left
# your y position decreases.
#         ^
#         |
#         U
#         P
# <--LEFT*RIGHT---->
#         D
#         O
#         W
#         N
#         |


class Map(object):

    def __init__(self, args, ascii_map, color_map=None):
        """

        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        num_agents: int
            Number of agents to have in the system.
        render: bool
            Whether to render the environment
        color_map: dict
            Specifies how to convert between ascii chars and colors
        """
        self.num_agents = args.num_agents
        self.base_map = self.ascii_to_numpy(
            ascii_map)  # list to numpy actually
        # map without agents or beams
        self.world_map = np.full(
            (len(self.base_map), len(self.base_map[0])), ' ')
        self.beam_pos = []

        self.agents = {}

        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS
        self.spawn_points = []  # where agents can appear

        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
        self.setup_agents()

        self.env_name = args.env_name
        self.share_reward = args.share_reward
        self.shape_reward = args.shape_reward
        self.shape_beta = args.shape_beta

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples and build walls"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean

        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken

        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    # FIXME(ev) move this to a utils eventually
    def ascii_to_numpy(self, ascii_list):
        """converts a list of strings into a numpy array


        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    @property
    def agent_pos(self):
        return [agent.get_pos().tolist() for agent in self.agents.values()]

    # This method is just used for testing
    # FIXME(ev) move into the testing class
    @property
    def test_map(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            # If agent is not within map, skip.
            if not (agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                    agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = 'P'

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            char_id = str(int(agent_id[-1]) + 1)  # agent-i

            # If agent is not within map, skip.
            if not(agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and  # not in the map
                   agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = char_id

        for beam_pos in self.beam_pos:
            # TODOSSD: structure of self.beam_pos
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def check_agent_map(self, agent_map):
        """Checks the map to make sure agents aren't duplicated"""
        unique, counts = np.unique(agent_map, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # check for multiple agents
        for i in range(self.num_agents):
            if count_dict[str(i+1)] != 1:
                print('Error! Wrong number of agent', i, 'in map!')
                return False
        return True

    def map_to_colors(self, map=None, color_map=None):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        if map is None:
            map = self.get_map_with_agents()
        if color_map is None:
            color_map = self.color_map

        rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=int)
        for row_elem in range(map.shape[0]):
            for col_elem in range(map.shape[1]):
                rgb_arr[row_elem, col_elem,
                        :] = color_map[map[row_elem, col_elem]]

        return rgb_arr

    def render(self, filename=None):
        """ Creates an image of the map to plot or save.

        Args:
            path: If a string is passed, will save the image
                to disk at this location.
        """
        map_with_agents = self.get_map_with_agents()

        rgb_arr = self.map_to_colors(map_with_agents)
        plt.imshow(rgb_arr, interpolation='nearest')

        if filename is not None:
            plt.savefig(filename)

        return rgb_arr.astype(np.uint8)

    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions.
        Also resolves conflicts over multiple agents wanting a cell.

        This method works by finding all conflicts over a cell and randomly assigning them
       to one of the agents that desires the slot. It then sets all of the other agents
       that wanted the cell to have a move of staying. For moves that do not directly
       conflict with another agent for a cell, but may not be temporarily resolvable
       due to an agent currently being in the desired cell, we continually loop through
       the actions until all moves have been satisfied or deemed impossible.
       For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
       Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
       first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
       then be able to move into [2,2].

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        """

        reserved_slots = []
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            # TODO(ev) these two parts of the actions
            if 'MOVE' in action or 'STAY' in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(
                    selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                # allow the agents to confirm what position they can move to

                new_pos = agent.return_valid_pos(new_pos)
                reserved_slots.append((*new_pos, 'P', agent_id))
            elif 'TURN' in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_agent_rot(new_rot)

        # now do the conflict resolution part of the process

        # helpful for finding the agent in the conflicting slot
        agent_by_pos = {
            tuple(agent.get_pos()): agent.agent_id for agent in self.agents.values()}

        # agent moves keyed by ids
        agent_moves = {}

        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []

        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'P':
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)

        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:

            # first we will resolve all slots over which multiple agents
            # want the slot

            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where(
                            (search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i]
                                         for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            if move.tolist() in self.agent_pos:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[tuple(
                                    move)]
                                curr_pos = self.agents[agent_id].get_pos(
                                ).tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id]. \
                                    get_pos().tolist()
                                conflict_move = agent_moves.get(conflicting_agent_id,
                                                                curr_conflict_pos)
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif conflicting_agent_id not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if agent_moves[conflicting_agent_id] == curr_pos and \
                                            move.tolist() == self.agents[conflicting_agent_id] \
                                            .get_pos().tolist():
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]
                                        ].update_agent_pos(move)
                            agent_by_pos = {tuple(agent.get_pos()):
                                            agent.agent_id for agent in self.agents.values()}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where(
                            (search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i]
                                         for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].get_pos(
                            ).tolist()

            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.get_pos()):
                                agent.agent_id for agent in self.agents.values()}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    if move in self.agent_pos:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].get_pos().tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].get_pos(
                        ).tolist()
                        conflict_move = agent_moves.get(
                            conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent_id not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():
                            if agent_moves[conflicting_agent_id] == curr_pos and \
                                    move == self.agents[conflicting_agent_id].get_pos().tolist():
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        self.agents[agent_id].update_agent_pos(move)
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break

    def update_custom_moves(self, agent_actions):
        for agent_id, action in agent_actions.items():
            # check its not a move based action
            if 'MOVE' not in action and 'STAY' not in action and 'TURN' not in action:
                agent = self.agents[agent_id]
                updates = self.custom_action(agent, action)
                if len(updates) > 0:
                    self.update_map(updates)

    def update_map(self, new_points):
        """For points in new_points, place desired char on the map"""
        for i in range(len(new_points)):
            row, col, char = new_points[i]
            self.world_map[row, col] = char

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        # del self.world_map
        self.world_map = np.full(
            (len(self.base_map), len(self.base_map[0])), ' ')
        self.build_walls()
        self.custom_reset()

    def update_map_fire(self, firing_pos, firing_orientation, fire_len, fire_char, cell_types=[],
                        update_char=[], blocking_cells='P'):
        """From a firing position, fire a beam that may clean or hit agents

        Notes:
            (1) Beams are blocked by agents
            (2) A beam travels along until it hits a blocking cell at which beam the beam
                covers that cell and stops
            (3) If a beam hits a cell whose character is in cell_types, it replaces it with
                the corresponding index in update_char
            (4) As per the rules, the beams fire from in front of the agent and on its
                sides so the beam that starts in front of the agent travels out one
                cell further than it does along the sides.
            (5) This method updates the beam_pos, an internal representation of how
                which cells need to be rendered with fire_char in the agent view

        Parameters
        ----------
        firing_pos: (list)
            the row, col from which the beam is fired
        firing_orientation: (list)
            the direction the beam is to be fired in
        fire_len: (int)
            the number of cells forward to fire
        fire_char: (str)
            the cell that should be placed where the beam goes
        cell_types: (list of str)
            the cells that are affected by the beam
        update_char: (list of str)
            the character that should replace the affected cells.
        blocking_cells: (list of str)
            cells that block the firing beam
        Returns
        -------
        updates: (tuple (row, col, char))
            the cells that have been hit by the beam and what char will be placed there
        """
        agent_by_pos = {
            tuple(agent.get_pos()): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        firing_pos = [start_pos, start_pos + right_shift - firing_direction,
                      start_pos - right_shift - firing_direction]
        firing_points = []
        updates = []
        for pos in firing_pos:
            next_cell = pos + firing_direction
            for i in range(fire_len):
                if self.test_if_in_bounds(next_cell) and \
                        self.world_map[next_cell[0], next_cell[1]] != '@':

                    # FIXME(ev) code duplication
                    # agents absorb beams
                    # activate the agents hit function if needed
                    if [next_cell[0], next_cell[1]] in self.agent_pos:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        self.agents[agent_id].hit(fire_char)
                        firing_points.append(
                            (next_cell[0], next_cell[1], fire_char))
                        if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                            type_index = cell_types.index(self.world_map[next_cell[0],
                                                                         next_cell[1]])
                            updates.append(
                                (next_cell[0], next_cell[1], update_char[type_index]))
                        break

                    # update the cell if needed
                    if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                        self.waste_cleared += 1
                        type_index = cell_types.index(
                            self.world_map[next_cell[0], next_cell[1]])
                        updates.append(
                            (next_cell[0], next_cell[1], update_char[type_index]))

                    firing_points.append(
                        (next_cell[0], next_cell[1], fire_char))

                    # check if the cell blocks beams. For example, waste blocks beams.
                    if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                        break

                    # increment the beam position
                    next_cell += firing_direction

                else:
                    break

        self.beam_pos += firing_points
        return updates

    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        spawn_index = 0
        is_free_cell = False
        curr_agent_pos = [agent.get_pos().tolist()
                          for agent in self.agents.values()]
        random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                spawn_index = i
                is_free_cell = True
        assert is_free_cell, 'There are not enough spawn points! Check your map?'
        return np.array(self.spawn_points[spawn_index])

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    def rotate_view(self, orientation, view):
        """Takes a view of the map and rotates it the agent orientation
        Parameters
        ----------
        orientation: str
            str in {'UP', 'LEFT', 'DOWN', 'RIGHT'}
        view: np.ndarray (row, column, channel)
        Returns
        -------
        a rotated view
        """
        if orientation == 'UP':
            return view
        elif orientation == 'LEFT':
            return np.rot90(view, k=1, axes=(0, 1))
        elif orientation == 'DOWN':
            return np.rot90(view, k=2, axes=(0, 1))
        elif orientation == 'RIGHT':
            return np.rot90(view, k=3, axes=(0, 1))
        else:
            raise ValueError('Orientation {} is not valid'.format(orientation))

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.world_map[row, col] = '@'

    ########################################
    # Utility methods, move these eventually
    ########################################

    # TODO(ev) this can be a general property of map_env or a util
    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == 'UP':
            return action_vec
        elif orientation == 'LEFT':
            return self.rotate_left(action_vec)
        elif orientation == 'RIGHT':
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return np.dot(ACTIONS['TURN_COUNTERCLOCKWISE'], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(ACTIONS['TURN_CLOCKWISE'], action_vec)

    # TODO(ev) this should be an agent property
    def update_rotation(self, action, curr_orientation):
        if action == 'TURN_COUNTERCLOCKWISE':
            if curr_orientation == 'LEFT':
                return 'DOWN'
            elif curr_orientation == 'DOWN':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'UP'
            else:
                return 'LEFT'
        else:
            if curr_orientation == 'LEFT':
                return 'UP'
            elif curr_orientation == 'UP':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'DOWN'
            else:
                return 'LEFT'

    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        if pos[0] < 0 or pos[0] >= self.world_map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.world_map.shape[1]:
            return False
        else:
            return True

    def step(self, actions):  # action [1,2,4,3,7]
        """A single environment step. Returns reward, terminated, info."""
        
        actions_ssd = {
            'agent-{}'.format(i): actions[i][0] for i in range(self.num_agents)}

        self.beam_pos = []
        agent_actions = {}
        for agent_id, action in actions_ssd.items():
            agent_action = self.agents[agent_id].action_map(
                action)  # such as 'FIRE'
            agent_actions[agent_id] = agent_action

        # move
        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.get_pos()
            # whether comsume an apple
            new_char = agent.consume(self.world_map[pos[0], pos[1]])
            self.world_map[pos[0], pos[1]] = new_char

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)

        # execute spawning events
        self.custom_map_update()

        observations = []
        rewards = []
        dones = []
        infos = {'collective_return': [],
                 'waste_cleared': [],
                 'apple_consumption': [],
                 'sustainability': [],
                 'fire': []}
        for i in range(self.num_agents):
            # del self.agents['agent-' + str(i)].grid
            self.agents['agent-' + str(i)].grid = self.get_map_with_agents()
            observations.append(self.get_obs_agent(i))
            reward = self.agents['agent-' + str(i)].compute_reward()
            rewards.append([reward])
            if reward > 0:
                self.agents['agent-' + str(i)].sustainability += 1
            dones.append(self.agents['agent-' + str(i)].get_done())

        # update infos
        collective_return = 0
        sustainability = 0
        fire = 0
        apple_consumption = 0
        for i in range(self.num_agents):
            collective_return += self.agents['agent-' +
                                             str(i)].collective_return
            sustainability += self.agents['agent-' + str(i)].sustainability
            fire += self.agents['agent-' + str(i)].fire
            if self.env_name == "Harvest":
                apple_consumption += self.agents['agent-' +
                                                 str(i)].apple_consumption

        infos['collective_return'] = collective_return
        infos['sustainability'] = sustainability/self.num_agents
        infos['fire'] = fire

        if self.env_name == "Cleanup":
            infos['waste_cleared'] = self.waste_cleared
        if self.env_name == "Harvest":
            infos['apple_consumption'] = apple_consumption

        if self.share_reward:          
            global_reward = np.sum(rewards)
            rewards = [[global_reward]] * self.num_agents

        return observations, rewards, dones, infos

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        # TODOSSD: RGB values or precise info ?
        _agent_id = 'agent-' + str(agent_id)  # consistent with pymarl
        agent = self.agents[_agent_id]
        state = agent.get_state()
        rgb_arr = self.map_to_colors(state, self.color_map)
        # del state
        rgb_arr = self.rotate_view(agent.orientation, rgb_arr)
        return rgb_arr.transpose(2, 0, 1)

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()

        observations = []

        for i in range(self.num_agents):
            # del self.agents['agent-' + str(i)].grid
            self.agents['agent-' + str(i)].grid = self.get_map_with_agents()
            observations.append(self.get_obs_agent(i))

        return observations

    def close(self):
        return

    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
