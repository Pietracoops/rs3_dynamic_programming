import csv
import random
import math
import numpy as np
import copy
import time
from rs3_simulator import CombatSimulation, Weapon, Abilities, Player, Levels
import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset
import argparse

actions_int_str_dict = {1: "Kick", 2: "Stomp", 3: "Punish", 4: "Dismember", 5: "Fury", 6: "Quake", 7: "Cleave", 8: "Assault", 9: "Pulverize", 10: "Slice", 11: "Backhand", 12: "Slaughter", 13: "Overpower", 14: "Forceful Backhand", 15: "Smash", 16: "Barge", 17: "Sever", 18: "Hurricane", 19: "Meteor Strike", 20: "Piercing Shot", 21: "Binding Shot", 22: "Snap Shot", 23: "Deadshot", 24: "Tight Bindings", 25: "Snipe", 26: "Dazing Shot", 27: "Fragmentation Shot", 28: "Rapid Fire", 29: "Ricochet", 30: "Bombardment", 31: "Wrack", 32: "Impact", 33: "Asphyxiate", 34: "Omnipower", 35: "Deep Impact", 36: "Dragon Breath", 37: "Combust", 38: "Chain", 39: "Wild Magic", 40: "Tsunami", 41: "Anticipation", 42: "Debilitate", 43: "Freedom"}
actions_str_int_dict = {"Kick": 1, "Stomp": 2, "Punish": 3, "Dismember": 4, "Fury": 5, "Quake": 6, "Cleave": 7, "Assault": 8, "Pulverize": 9, "Slice": 10, "Backhand": 11, "Slaughter": 12, "Overpower": 13, "Forceful Backhand": 14, "Smash": 15, "Barge": 16, "Sever": 17, "Hurricane": 18, "Meteor Strike": 19, "Piercing Shot": 20, "Binding Shot": 21, "Snap Shot": 22, "Deadshot": 23, "Tight Bindings": 24, "Snipe": 25, "Dazing Shot": 26, "Fragmentation Shot": 27, "Rapid Fire": 28, "Ricochet": 29, "Bombardment": 30, "Wrack": 31, "Impact": 32, "Asphyxiate": 33, "Omnipower": 34, "Deep Impact": 35, "Dragon Breath": 36, "Combust": 37, "Chain": 38, "Wild Magic": 39, "Tsunami": 40, "Anticipation": 41, "Debilitate": 42, "Freedom": 43}

def heuristic_function_highest_damage(player):

    # Heuristic function needs to be constant - for a given rollout, needs to return same sequence of actions

    # Update player first
    player.update_available_abilities()

    # Get action that has highest damage output from list of available actions
    player_action = player.available_abilities


    ability_vect = [-1, ""]
        # Use the freedom ability if stunned or bleeding
    if player.is_stunned and "Freedom" in player.available_abilities:
        player_ability = player.abilities.get_ability("Freedom")
        ability_vect = [0, "Freedom"]
    else:
        for ability_name in player_action:
            # Get the ability first
            player_ability = player.abilities.get_ability(ability_name)
            if player_ability.damage_max > ability_vect[0]:
                ability_vect = [player_ability.damage_max, ability_name]

    return ability_vect[1]

def heuristic_function_random(player):
    # Update player first
    player.update_available_abilities()
    # Heuristic function choosing random action
    player_action = player.available_abilities
    # Pick random ability from list of available actions
    player_ability = random.choice(list(player_action))
    return player_ability

def reward_function(player1, player2):
    # Need to reward high damage differential and the adding and removing of statuses
    damage_delta = player1.ability_damage - player2.ability_damage
    stun_penalty = 0
    bleed_penalty = 0
    stun_bonus = 0
    bleed_bonus = 0

    if player1.is_stunned:
        stun_penalty = -1500
    if player1.is_bleeding:
        bleed_penalty = -500
    
    if player2.is_stunned:
        stun_bonus = 1500
    if player2.is_bleeding:
        bleed_bonus = 500

    return damage_delta + stun_penalty + bleed_penalty + stun_bonus + bleed_bonus

def generic_rollout(player1, player2, combat_simulator, num_iterations, num_rollouts, look_ahead=100):
    print(f"Starting Rollout with {num_iterations} iterations and {num_rollouts} rollouts")
    Q_state_action_dict = {}
    for k in range(num_iterations):
        # For all possible actions at the given state
        player1_state_vect, player1_state_enc = player1.get_state()
        player2_state_vect, player2_state_enc = player2.get_state()
        player1.update_available_abilities()
        player2.update_available_abilities()
        possible_actions = player1.available_abilities # set contains all possible actions at the given state
        print(f"State: {player1_state_enc}, {player2_state_enc}")
        Q_state_action_dict[(player1_state_enc, player2_state_enc)] = []
        for player1_action in possible_actions:
            Q_tilda = 0

            for s in range(num_rollouts):
                # Perform a deep copy of each player so that we don't lose the original state
                player1_copy = copy.deepcopy(player1)
                player2_copy = copy.deepcopy(player2)
                combat_simulator_copy = copy.deepcopy(combat_simulator)
                combat_simulator_copy.player = player1_copy
                combat_simulator_copy.opponent = player2_copy

                # Get Player2 Action using heuristic
                player2_action_roll = heuristic_function_highest_damage(player2_copy)

                # Throw actions into combat_simulator
                # Note: We are not taking into account armor or combat changes
                player1_copy, player2_copy, iter_time = combat_simulator_copy.simulate([player1_action, None, None], [player2_action_roll, None, None], player1_random_simulate=False, player2_random_simulate=False)
                Q_tilda_s = reward_function(player1_copy, player2_copy)

                for i in range(look_ahead): # Defaults to 100 iterations which equals to 180 seconds of combat - overestimated and will break when player is dead
                    player1_action_roll = heuristic_function_highest_damage(player1_copy)
                    player2_action_roll = heuristic_function_highest_damage(player2_copy)
                    player1_copy, player2_copy, iter_time = combat_simulator_copy.simulate([player1_action_roll, None, None], [player2_action_roll, None, None], player1_random_simulate=False, player2_random_simulate=False)
                    Q_tilda_s += reward_function(player1_copy, player2_copy)
                    if player1_copy.health == 0 or player2_copy.health == 0: # Terminate if any of the players have been killed
                        break

                Q_tilda += Q_tilda_s # Add the Q value of the rollout to the Q value of the state
            Q_tilda = Q_tilda / num_rollouts # Divide by number of rollouts to get average Q value for the state
            Q_state_action_dict[(player1_state_enc, player2_state_enc)].append([Q_tilda, player1_action])

        # Take the action that maximizes the Q value of the state
        best_action = sorted(Q_state_action_dict[(player1_state_enc, player2_state_enc)], key=lambda x: x[0])[-1][1]
        # Use heuristic function for player 2 action
        player2_action = heuristic_function_highest_damage(player2)
        # Take the next step of the simulation
        player1, player2, iter_time = combat_simulator.simulate([best_action, None, None], [player2_action, None, None], player1_random_simulate=False, player2_random_simulate=False)
        print(f"Iteration: {k}, Player1 Health: {player1.health}, Player2 Health: {player2.health}, Player1 Ability Used: {best_action}, Player2 Ability Used: {player2_action}")
        if player1.health == 0 or player2.health == 0: # Terminate if any of the players have been killed
            return player1, player2 # Return the player objects


class Node:
    def __init__(self, player1, player2, combat_simulator, action_name=None, parent=None, depth=0):
        self.player1 = player1
        self.player2 = player2
        self.combat_simulator = combat_simulator
        player1_state_vect, player1_state_enc = player1.get_state()
        player2_state_vect, player2_state_enc = player2.get_state()
        self.state = (player1_state_enc, player2_state_enc)
        self.action = action_name
        self.parent = parent
        self.depth = depth
        self.children = []
        self.reward = 0
        self.visits = 0
        self.is_terminal = False
        Node.update(self) # Update the depth of the node
    
    def update(self):
        self.player1.update_available_abilities()
        self.player2.update_available_abilities()
        if self.parent != None:
            self.depth = self.parent.depth + 1

    def expand(self):
        untried_actions = self.get_untried_actions() # List of untried actions
        action = random.choice(untried_actions) # Take a random action from the list of untried actions
        child = Node(copy.deepcopy(self.player1), copy.deepcopy(self.player2), copy.deepcopy(self.combat_simulator), action, self, self.depth) # Create a child node
        self.children.append(child) # Add the child node to the children list
        return child

    def simulate(self, num_rollouts, look_ahead=100):
        # Rollout from the child's action
        Q_tilda = rollout(copy.deepcopy(self.player1), copy.deepcopy(self.player2), copy.deepcopy(self.combat_simulator), heuristic_function_highest_damage, self.action, num_rollouts, look_ahead)
        return Q_tilda

    
    def get_untried_actions(self):
        # Get all abilities that the player can use
        all_abilities = self.player1.available_abilities # Initialize an empty list to store the untried actions
        untried_actions = [] # Iterate over all abilities and check if they have been tried already
        for ability in all_abilities:
            is_tried = False # Initialize a boolean flag to keep track of whether the ability has been tried or not
            for child in self.children: # Iterate over all child nodes and check if their action matches the current ability
                if child.action == ability:
                    is_tried = True
                    break
            if not is_tried: # If the ability has not been tried, add it to the list of untried actions
                untried_actions.append(ability)
    
        # Return the list of untried actions
        return untried_actions

    def is_fully_expanded(self):
        return len(self.children) == len(self.player1.available_abilities)
    
    def backpropagate(self, reward):
        reward = reward / 10000
        self.visits += 1
        self.reward += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def get_best_action(self):
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.action
    
    def select_child(self):
        exploration_factor = 1 / math.sqrt(2.0)
        max_uct = -float("inf")
        selected_child = None
        for child in self.children:
            uct = child.reward / child.visits + exploration_factor * math.sqrt(math.log(self.visits) / child.visits)
            if uct > max_uct:
                max_uct = uct
                selected_child = child
        return selected_child

def rollout(player1, player2, combat_simulator, heuristic_function, player_action, num_rollouts, look_ahead=100):
        # Rollout
        player1.update_available_abilities()
        player2.update_available_abilities()
        Q_tilda = 0
        for s in range(num_rollouts):
            player1_copy = copy.deepcopy(player1)
            player2_copy = copy.deepcopy(player2)
            combat_simulator_copy = copy.deepcopy(combat_simulator)
            combat_simulator_copy.player = player1_copy
            combat_simulator_copy.opponent = player2_copy
            # Get Player2 Action using heuristic
            player2_action_roll = heuristic_function(player2_copy)

            # Throw actions into combat_simulator
            # Note: We are not taking into account armor or combat changes
            player1_copy, player2_copy, iter_time = combat_simulator_copy.simulate([player_action, None, None], [player2_action_roll, None, None], player1_random_simulate=False, player2_random_simulate=False)
            Q_tilda_s = reward_function(player1_copy, player2_copy)

            for i in range(look_ahead): # Defaults to 100 iterations which equals to 180 seconds of combat - overestimated and will break when player is dead
                player1_action_roll = heuristic_function(player1_copy)
                player2_action_roll = heuristic_function(player2_copy)
                player1_copy, player2_copy, iter_time = combat_simulator_copy.simulate([player1_action_roll, None, None], [player2_action_roll, None, None], player1_random_simulate=False, player2_random_simulate=False)
                Q_tilda_s += reward_function(player1_copy, player2_copy)
                if player1_copy.health == 0 or player2_copy.health == 0: # Terminate if any of the players have been killed
                    break

            Q_tilda += Q_tilda_s # Add the Q value of the rollout to the Q value of the state
        Q_tilda = Q_tilda / num_rollouts # Divide by number of rollouts to get average Q value for the state
        return Q_tilda
            
def mcts(player1, player2, combat_simulator, num_iterations, num_branches, num_rollouts, look_ahead=100):
    print(f"Starting Monte Carlo Tree Search with {num_iterations} iterations and {num_rollouts} rollouts")
    for k in range(num_iterations):
        
        # For all possible actions at the given state
        root_node = Node(player1, player2, combat_simulator, None) # Initialize root node
        

        for i in range(num_branches):
            node = root_node
            # Selection
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.select_child()

            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()
                
            # Simulation
            reward = node.simulate(num_rollouts, look_ahead)

            # Backpropagation
            node.backpropagate(reward)

        # Return the best action from the root node
        best_action = root_node.get_best_action()

        # Use heuristic function for player 2 action
        player2_action = heuristic_function_highest_damage(player2)
        # Take the next step of the simulation
        player1, player2, iter_time = combat_simulator.simulate([best_action, None, None],
                                                                [player2_action, None, None],
                                                                 player1_random_simulate=False, player2_random_simulate=False)
        print(f"Iteration: {k}, Player1 Health: {player1.health},", end=" ")
        print(f"Player2 Health: {player2.health}, Player1 Ability Used: {best_action}, Player2 Ability Used: {player2_action}")
        if player1.health == 0 or player2.health == 0: # Terminate if any of the players have been killed
            return player1, player2 # Return the player objects

# Define the model class
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_parametric_approximator(player1, player2, combat_simulator):
    print("Starting Parametric Approximator Training")
    # Define the training parameters
    num_iterations = 10000
    num_rollouts = 100
    
    # Define the model parameters
    input_dim = 3
    hidden_dim = 20
    output_dim = 1
    
    # set batch size
    batch_size = 32
    num_epochs = 1000

    load_data = True

    # Instantiate the model
    model = MyModel(input_dim, hidden_dim, output_dim)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if load_data != True:
        # Perform Several combat simulations and rollouts to gather data
        print("Beginning Rollouts")
        states = []
        q_values = []
        for k in range(num_iterations):
            player1_state_vect, player1_state_enc = player1.get_state()
            player2_state_vect, player2_state_enc = player2.get_state()
            player1.update_available_abilities()
            player2.update_available_abilities()
            possible_actions = player1.available_abilities # set contains all possible actions at the given state
            for player1_action in possible_actions:
                Q_tilda = rollout(copy.deepcopy(player1), copy.deepcopy(player2),
                                  copy.deepcopy(combat_simulator), player1_action, heuristic_function_random, num_rollouts)
                action_int = actions_str_int_dict[player1_action]
                state = np.array([player1_state_enc, player2_state_enc, action_int])
                states.append(state)
                q_values.append(Q_tilda)
            
            print(f"Iteration: {k}, Samples Collected: {len(states)}")
        
        # Save the training data
        with open('rollouts.pkl', 'wb') as f:
            pickle.dump((states, q_values), f)
    else:
        # Load them up by doing the following:
        with open('rollouts.pkl', 'rb') as f:
            states, q_values = pickle.load(f)
    

    # Normalize the data
    states_scaled = states / np.max(states, axis=0)
    scaler = StandardScaler()
    q_values = np.array(q_values).reshape(-1, 1)
    scaler.fit(q_values)
    q_values_scaled = scaler.fit_transform(q_values)
    # max_value = scaler.scale_.max()
    # min_value = scaler.scale_.min()
    # og_qvalues = scaler.inverse_transform(q_values_scaled)

    # states = scaler.fit_transform(states)
    X = torch.FloatTensor(states_scaled)
    Y = torch.FloatTensor(q_values_scaled)


    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # create TensorDataset from X_train and y_train
    train_dataset = TensorDataset(X_train, Y_train)

    # create DataLoader for train data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # create TensorDataset from X_test and y_test
    test_dataset = TensorDataset(X_test, Y_test)

    # create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    for epoch in range(num_epochs):
        # Training
        print(f'Epoch {epoch + 1}')
        running_loss = 0.0
        i_counter = 0
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(torch.tensor(inputs, dtype=torch.float32))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # Print statistics
            running_loss += loss.item()
            if (i+1) % batch_size == 0:
                #print(f'Epoch {epoch+1}, Batch {i+1}: Loss: {running_loss/batch_size:.3f}')
                running_loss = 0.0
            i_counter += 1

        avg_loss = running_loss / (i_counter+1)

        # Validation
        running_vloss = 0.0
        j_counter = 0
        for j, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
            j_counter += 1
        avg_vloss = running_vloss / (j_counter + 1)
        print(f"LOSS train {avg_loss} valid {avg_vloss}")

    # Save the model weights
    torch.save(model.state_dict(), f"parametric_approximator_weights.pt")

def generic_rollout_para(player1, player2, combat_simulator, num_iterations, model=None, parametric_approximator=False):
    print(f"Starting Rollout with {num_iterations} iterations and parametric approximation")

    # Load the model
    print("Loading Parametric Approximator...")
    with open('rollouts.pkl', 'rb') as f:
        states, q_values = pickle.load(f)
    states_max = np.max(states, axis=0)
    scaler = StandardScaler()
    q_values = np.array(q_values).reshape(-1, 1)
    scaler.fit(q_values)
    input_dim = 3
    hidden_dim = 20
    output_dim = 1
    model = MyModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('parametric_approximator_weights.pt'))

    best_actions = []
    player2_actions = []
    Q_state_action_dict = {}
    for k in range(num_iterations):

        # For all possible actions at the given state
        player1_state_vect, player1_state_enc = player1.get_state()
        player2_state_vect, player2_state_enc = player2.get_state()
        player1.update_available_abilities()
        player2.update_available_abilities()
        possible_actions = player1.available_abilities # set contains all possible actions at the given state
        print(f"State: {player1_state_enc}, {player2_state_enc}")
        Q_state_action_dict[(player1_state_enc, player2_state_enc)] = []
        for player1_action in possible_actions:
            Q_tilda = 0

            # Convert action to int
            action_int = actions_str_int_dict[player1_action]
            # Build the input array
            state = np.array([player1_state_enc, player2_state_enc, action_int])
            # Normalize the input array
            state_scaled = state / states_max
            Q_tilda_unscaled = model(torch.tensor(state_scaled,  dtype=torch.float32))
            mass = Q_tilda_unscaled.detach().numpy()
            Q_tilda = scaler.inverse_transform(mass.reshape(1, -1))
            Q_state_action_dict[(player1_state_enc, player2_state_enc)].append([Q_tilda, player1_action])

        # Take the action that maximizes the Q value of the state
        best_action = sorted(Q_state_action_dict[(player1_state_enc, player2_state_enc)], key=lambda x: x[0])[-1][1]
        best_actions.append(best_action)
        # Use heuristic function for player 2 action
        player2_action = heuristic_function_highest_damage(player2)
        # Take the next step of the simulation
        player1, player2, iter_time = combat_simulator.simulate([best_action, None, None],
                                                                [player2_action, None, None],
                                                                 player1_random_simulate=False, player2_random_simulate=False)
        print(f"Iteration: {k}, Player1 Health: {player1.health}", end=" ")
        print(f"Player2 Health: {player2.health}, Player1 Ability Used: {best_action}, Player2 Ability Used: {player2_action}")
        if player1.health == 0 or player2.health == 0: # Terminate if any of the players have been killed
            return player1, player2, best_actions # Return the player objects


# Backwards Chaining Algorithm

def invert_dict(d):
    inverted_dict = {}
    for key, value in d.items():
        inverted_dict[tuple(value)] = key
    return inverted_dict

def exact_reward(player, opponent, player_action, opponent_action):
    # Get the player_action object from player
    player_action_object = player.abilities.get_ability(player_action)
    # Get the opponent_action object from opponent
    opponent_action_object = opponent.abilities.get_ability(opponent_action)
    # Get Average Damage
    player_average_damage = player_action_object.get_average_ability_damage()
    opponent_average_damage = opponent_action_object.get_average_ability_damage()
    # Calculate the reward
    reward = player_average_damage - opponent_average_damage
    return reward
    
def convert_adrenaline_to_state(adrenaline):
    player_adrenaline_category = 0
    if adrenaline < 25:
        player_adrenaline_category = 0
    elif adrenaline < 50 and adrenaline >= 25:
        player_adrenaline_category = 25
    elif adrenaline < 75 and adrenaline >= 50:
        player_adrenaline_category = 50
    elif adrenaline < 100 and adrenaline >= 75:
        player_adrenaline_category = 75
    elif adrenaline == 100:
        player_adrenaline_category = 100

    return player_adrenaline_category

def get_next_state(player, opponent, player_action, opponent_action, state_inverted_dict):
    player_adrenaline_category = 0
    opponent_adrenaline_category = 0
    # Get player adrenaline level
    player_adrenaline = player.adrenaline_level
    # Get opponent adrenaline level
    opponent_adrenaline = opponent.adrenaline_level

    # Get the player_action object from player
    player_action_object = player.abilities.get_ability(player_action)
    # Get the opponent_action object from opponent
    opponent_action_object = opponent.abilities.get_ability(opponent_action)

    player_adrenaline = player_action_object.energy_cost
    opponent_adrenaline = opponent_action_object.energy_cost

    player_adrenaline_category = convert_adrenaline_to_state(player_adrenaline)
    opponent_adrenaline_category = convert_adrenaline_to_state(opponent_adrenaline)

    state = state_inverted_dict[tuple([player_adrenaline_category, opponent_adrenaline_category])]
    return state

def exact_method(player, opponent, combat_simulator, num_iterations):

    # Get all possible states for player 1 melee, player 2 ranged

    # []

    player1 = copy.deepcopy(player)
    player2 = copy.deepcopy(opponent)

    player1.update_available_abilities()
    player_possible_actions = player1.available_abilities
    opponent_possible_actions = player2.available_abilities
    states_dict = {1:[0, 0], 2:[0, 25], 3:[0, 50], 4:[0, 75], 5:[0, 100], 6:[25, 0], 7:[25, 25],
                   8:[25, 50], 9:[25, 75], 10:[25, 100], 11:[50, 0], 12:[50, 25], 13:[50, 50],
                   14:[50, 75], 15:[50, 100], 16:[75, 0], 17:[75, 25], 18:[75, 50], 19:[75, 75],
                   20:[75, 100], 21:[100, 0], 22:[100, 25], 23:[100, 50], 24:[100, 75], 25:[100, 100]} 
    states_inverted_dict = invert_dict(states_dict)
    j_array = np.zeros((num_iterations+1, len(states_dict)+1)) # 2 dimensional array containing the stored rewards
    mu_array = np.zeros((num_iterations+1, len(states_dict)+1)) # 2 dimensional array containing the stored actions
    j_u_array_global = []
    for k in range(num_iterations-1, -1, -1):

        j_k_array = []
        mu_k_array = []
        for xi in range (1, len(states_dict) + 1): # Loop through the states
            # Update the adrenaline level of the players based on the state
            player1.adrenaline_level = states_dict[xi][0]
            player2.adrenaline_level = states_dict[xi][1]
            # Update the player abilities
            player1.update_available_abilities()
            player2.update_available_abilities()
            player_possible_actions = player1.available_abilities
            opponent_possible_actions = player2.available_abilities
            j_u_array = np.zeros(44) # Temporary array to store all reward values for a state (X)
            for ui in range (0, len(player_possible_actions)): # Loop through all possible action values (Abilities)
                player_action_int = actions_str_int_dict[list(player_possible_actions)[ui]] # Convert action to int
                j_val = 0
                j_val2 = 0
                for wi in range(0, len(opponent_possible_actions)): # Loop through all possible opponent abilities towards us
                    opponent_action_int = actions_str_int_dict[list(opponent_possible_actions)[wi]] # Convert action to int
                    ratio = (1 / len(opponent_possible_actions))
                    j_val2 += ratio * (exact_reward(player1, player2, list(player_possible_actions)[ui],
                                                    list(opponent_possible_actions)[wi]) + j_array[k + 1][get_next_state(player1, player2,
                                                    list(player_possible_actions)[ui], list(opponent_possible_actions)[wi], states_inverted_dict)])
                j_val = j_val2 # Perform the initial cost function part
                j_u_array[player_action_int] = j_val # Append our cost value to our temporary cost array
            best_action = np.argmin(j_u_array, axis=0) # Find the best action
            j_u_array_global.append(list(j_u_array))
            j_array[k][xi] = j_u_array[best_action] # Store the cost value associated to that best action in global reward array
            mu_array[k][xi] = best_action # Store the best action inside glboal mu array



    # Run the simulation
    for k in range(100): # Run until the combat is done
        # Update player and opponent abilities
        player.update_available_abilities()
        opponent.update_available_abilities()
        # Check player adrenaline level
        player_adrenaline_category = convert_adrenaline_to_state(player.adrenaline_level)
        opponent_adrenaline_category = convert_adrenaline_to_state(opponent.adrenaline_level)
        state = states_inverted_dict[tuple([player_adrenaline_category, opponent_adrenaline_category])]
        tmp_array = j_u_array_global[state - 1]
        tmp_array = np.array(tmp_array)
        sorted_indices = np.argsort(tmp_array)
        sorted_indices = sorted_indices[::-1]
        # Loop through all the abilities and see if they are available
        for i in range(len(sorted_indices)):
            if sorted_indices[i] == 0:
                continue
            # Get the ability name first
            ability_name = actions_int_str_dict[sorted_indices[i]]
            # Check if the ability is available
            if ability_name in player.available_abilities:
                # Get opponent ability using heuristic
                player2_action = heuristic_function_highest_damage(opponent)

                # Execute the ability
                player, opponent, iter_time = combat_simulator.simulate([ability_name, None, None],
                                                                        [player2_action, None, None],
                                                                         player1_random_simulate=False, player2_random_simulate=False)
                print(f"Iteration: {k}, Player1 Health: {player.health}", end=" ")
                print(f"Player2 Health: {opponent.health}, Player1 Ability Used: {ability_name}, Player2 Ability Used: {player2_action}")
                if player.health == 0 or opponent.health == 0: # Terminate if any of the players have been killed
                    return player, opponent # Return the player objects
                break


def Initialization():
   # ======================== Initialization =======================

    # Load abilities
    abilities = Abilities("abilities.csv")
    abilities.parse_abilities()

    # Create weapons
    # Melee
    rune_2hand_sword = Weapon()
    rune_2hand_sword.name = "Rune 2hand Sword"
    rune_2hand_sword.type = "Melee"
    rune_2hand_sword.damage = 1184
    rune_2hand_sword.accuracy = 928
    rune_2hand_sword.range = 1
    rune_2hand_sword.speed = 1
    rune_2hand_sword.level = 50
    rune_2hand_sword.additional_bonuses = 15
    
    # Ranged
    magic_shortbow = Weapon()
    magic_shortbow.name = "Magic Shortbow"
    magic_shortbow.type = "Ranged"
    magic_shortbow.damage = 438
    magic_shortbow.accuracy = 850
    magic_shortbow.range = 7
    magic_shortbow.speed = 2
    magic_shortbow.level = 50
    magic_shortbow.additional_bonuses = 15

    # Magic
    fire_blast_mainhand = Weapon()
    fire_blast_mainhand.name = "Mystic Wand - Fire Blast"
    fire_blast_mainhand.type = "Magic"
    fire_blast_mainhand.damage = 712
    fire_blast_mainhand.accuracy = 850
    fire_blast_mainhand.range = 8
    fire_blast_mainhand.speed = 3
    fire_blast_mainhand.level = 50
    fire_blast_mainhand.additional_bonuses = 15

    fire_blast_offhand = Weapon()
    fire_blast_offhand.name = "Mystic Orb - Fire Blast"
    fire_blast_offhand.type = "Magic"
    fire_blast_offhand.damage = 712
    fire_blast_offhand.accuracy = 850
    fire_blast_offhand.range = 8
    fire_blast_offhand.speed = 3
    fire_blast_offhand.level = 50
    fire_blast_offhand.additional_bonuses = 15

    # Create players
    player_levels = Levels()
    player_levels.hitpoints = 9700
    player_levels.attack = 95
    player_levels.defence = 95
    player_levels.strength = 90
    player_levels.magic = 90
    player_levels.ranged = 90
    player = Player("Its mass", player_levels)
    player.melee_weapon_mainhand = rune_2hand_sword
    player.ranged_weapon_mainhand = magic_shortbow
    player.magic_weapon_mainhand = fire_blast_mainhand
    player.magic_weapon_offhand = fire_blast_offhand
    player.abilities = copy.deepcopy(abilities)
    player.calculate_ability_damage_multiplier()
    player.abilities.bind_abilities_to_player(player.melee_ability_damage, player.ranged_ability_damage, player.magic_ability_damage)


    opponent_levels = Levels()
    opponent_levels.hitpoints = 9900
    opponent_levels.attack = 99
    opponent_levels.defence = 99
    opponent_levels.strength = 99
    opponent_levels.magic = 95
    opponent_levels.ranged = 99
    opponent = Player("Pietracoops", opponent_levels) # Pietracoops
    opponent.melee_weapon_mainhand = rune_2hand_sword
    opponent.ranged_weapon_mainhand = magic_shortbow
    opponent.magic_weapon_mainhand = fire_blast_mainhand
    opponent.magic_weapon_offhand = fire_blast_offhand
    opponent.abilities = copy.deepcopy(abilities)
    opponent.calculate_ability_damage_multiplier()
    opponent.abilities.bind_abilities_to_player(opponent.melee_ability_damage, opponent.ranged_ability_damage, opponent.magic_ability_damage)

    # Initialize their starting weapon and armor
    player.set_armor_style("Melee")
    player.set_combat_style("Melee")

    opponent.set_armor_style("Ranged")
    opponent.set_combat_style("Ranged")

    return player, opponent

class TestObject:
    def __init__(self, player, opponent):
        self.player = player
        self.opponent = opponent

        self.matches = []
        self.times = []
        self.player_victories = 0
        self.opponent_victories = 0
        self.ties = 0

    def compute_average_delta_hp(self):
        total_delta_hp = 0
        for match in self.matches:
            total_delta_hp += match[0] - match[1]
        self.average_delta_hp = total_delta_hp / len(self.matches)
        return self.average_delta_hp

    def compute_average_time(self):
        total_time = 0
        for time in self.times:
            total_time += time
        self.average_time = total_time / len(self.matches)
        return self.average_time
    
    def print_statistics(self):
        print(f"Player1: {self.player.name}, Win rate: {self.player_victories / len(self.matches)}")
        print(f"Player2: {self.opponent.name}, Win rate: {self.opponent_victories / len(self.matches)}")
        print(f"Average Delta HP: {TestObject.compute_average_delta_hp(self)}")
        print(f"Average Time: {TestObject.compute_average_time(self)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Algorithms entrypoint')
    parser.add_argument('--model', type=str, default='parametric_rollout', choices=['parametric_rollout', 'random', 'generic_rollout', 'mcts', 'backwards_chain'],
                        help='Available algorithms include: parametric_rollout, random, generic_rollout, mcts, backwards_chain')
    args = parser.parse_args()

    print("Initializing...")
    player, opponent = Initialization()

    player.print_status()
    opponent.print_status()

    CombatSimulator = CombatSimulation(player, opponent)

    print(f"Performing Tests for {args.model}")
    test_obj = TestObject(player, opponent)
    nanos_to_sec = (10 ** -9)
    RANDOM = 'random'
    GENERIC_ROLLOUT = 'generic_rollout'
    PARAMETRIC_ROLLOUT = 'parametric_rollout'
    MCTS = 'mcts'
    EXACT = 'backwards_chain'
    TEST = args.model # CHANGE THIS VALUE TO PERFORM THE TESTS FOR EACH ALGORITHM

    if TEST == 1:
        iterations = 100
    else:
        iterations = 1
    for j in range(100):
        player1 = copy.deepcopy(player)
        player2 = copy.deepcopy(opponent)
        combat_simulator = copy.deepcopy(CombatSimulator)
        combat_simulator.verbosity = 0
        print(f"Random Simulation Test: {j+1}")
        start_time_sec = time.time_ns() * nanos_to_sec
        for i in range(iterations):
            if TEST == RANDOM:
                player1, player2, iter_time = combat_simulator.simulate(None, None, player1_random_simulate=True, player2_random_simulate=True)
            if TEST == GENERIC_ROLLOUT:
                combat_simulator.verbosity = 0
                player1, player2 = generic_rollout(player1, player2, combat_simulator, 100, 4, 2)
            if TEST == PARAMETRIC_ROLLOUT:
                player1, player2, _ = generic_rollout_para(player1, player2, combat_simulator, 100, 20)
            if TEST == MCTS:
                player1, player2 = mcts(player1, player2, combat_simulator, 100, 15, 5, 4)
            if TEST == EXACT:
                player1, player2 = exact_method(player1, player2, combat_simulator, 40) # original 40

            if player1.health == 0 and player2.health == 0:
                print(f"it's a tie!")
                end_time_sec = time.time_ns() * nanos_to_sec
                test_obj.times.append(end_time_sec - start_time_sec)
                test_obj.ties += 1
                test_obj.matches.append([player1.health, player2.health])
                test_obj.print_statistics()
                break
            elif player1.health == 0:
                print(f"{opponent.name} wins!")
                end_time_sec = time.time_ns() * nanos_to_sec
                test_obj.times.append(end_time_sec - start_time_sec)
                test_obj.opponent_victories += 1
                test_obj.matches.append([player1.health, player2.health])
                test_obj.print_statistics()
                break
            elif player2.health == 0:
                print(f"{player.name} wins!")
                end_time_sec = time.time_ns() * nanos_to_sec
                test_obj.times.append(end_time_sec - start_time_sec)
                test_obj.player_victories += 1
                test_obj.matches.append([player1.health, player2.health])
                test_obj.print_statistics()
                break

    # Individual tests below for testing

    # # ==================== Parametric Approximation Deep Learning =======================
    # CombatSimulator.verbosity = 0
    # #train_parametric_approximator(player, opponent, CombatSimulator)
    # print("Performing Rollouts...")
    # CombatSimulator.verbosity = 0
    # player, opponent = generic_rollout_para(player, opponent, CombatSimulator, 100, 20)
    # if player.health == 0:
    #     print(f"{opponent.name} wins!")
    # elif opponent.health == 0:
    #     print(f"{player.name} wins!")


    # # ===================== MCTS Simulations =======================
    # print("Performing MCTS...")
    # CombatSimulator.verbosity = 0
    # player, opponent = mcts(player, opponent, CombatSimulator, 100, 100, 100)
    # if player.health == 0:
    #     print(f"{opponent.name} wins!")
    # elif opponent.health == 0:
    #     print(f"{player.name} wins!")


    # # ==================== Rollout Simulations =======================
    # print("Performing Rollouts...")
    # CombatSimulator.verbosity = 0
    # player, opponent = generic_rollout(player, opponent, CombatSimulator, 100, 20)
    # if player.health == 0:
    #     print(f"{opponent.name} wins!")
    # elif opponent.health == 0:
    #     print(f"{player.name} wins!")


    # # ==================== Random Simulations =======================
    # print("Randomly Simulating...")
    # for i in range(100):
    #     player, opponent, iter_time = CombatSimulator.simulate(None, None, player1_random_simulate=True, player2_random_simulate=True)
    #     print("")

    #     if player.health == 0:
    #         print(f"{opponent.name} wins!")
    #         break
    #     if opponent.health == 0:
    #         print(f"{player.name} wins!")
    #         break

    # print("done")


    # ==================== Exact Method =======================
    # print("Performing Exact Method...")
    # player, opponent = exact_method(player, opponent, CombatSimulator, 5)
    # if player.health == 0:
    #     print(f"{opponent.name} wins!")
    # elif opponent.health == 0:
    #     print(f"{player.name} wins!")


    print("done")
    