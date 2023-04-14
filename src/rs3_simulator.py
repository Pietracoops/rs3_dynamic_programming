import csv
import random
import math
import numpy as np
import copy
import time

game_tick_duration = 0.6 # seconds
MELEE = 0
MAGIC = 1
RANGED = 2
nanos_to_sec = (10 ** -9)

def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

class CombatSimulation:
    def __init__(self, player, opponent):
        # Each iteration is a game tick which is 0.6 seconds

        # When an ability is used, it cannot be used again for a specific amount of time (its cooldown),
        # and other abilities cannot be used for 3 ticks (1.8 seconds; the global cooldown). Constantly
        # using abilities seamlessly is generally the most effective form of combat.

        self.current_game_tick = 0
        self.player = player
        self.opponent = opponent
        self.iteration_time = 0.0
        self.verbosity = 1 # 0 = No output, 1 = Outputs player's actions


        # Adrenaline (energy) is gained by:
        # 2% - Auto attack
        # 8% - Basic Ability
        # -15% - Threshold Ability
        # -100% - Ultimate Ability

        # It is important to note that using an ability on the same game tick as an auto-attack would be fired on will prevent the auto-attack from automatically firing.
        
        # - Auto Attacks -
        # MELEE
        # Fastest: 4 per hit (Claws, daggers, defenders, maces, scimitars, whips)
        # Fast: 5 per hit (Hastae, hatchets, longswords, rapiers, swords)
        # Average: 6 per hit (Battleaxes, halberds, most mauls, pickaxes, warhammers, spears, two-handed swords)
        # MAGIC
        # Combat spells cast with wands or orbs: 4 per cast
        # Combat spells cast with Staves: 6 per cast
        # RANGED
        # Fastest: 4 per hit (Crossbows, darts, hexhunter bow, throwing knives, seercull)
        # Fast: 5 per hit (Most shortbows, Zaryte bow, Crystal bow, Chinchompas)
        # Average: 6 per hit (Javelins, shieldbows (except the dark bow), salamanders, thrown axes, two-handed crossbows)
        # Slowest: 12 per hit (Dark bow)

        # - Bleeds -
        # Bleeds hit once every 2 game ticks

        # Auto Attack Block Chances
        self.aa_block_chances = [ [0.23, 0.50, 0.23],  # Melee  vs. [Melee, Magic, Ranged]
                                  [0.16, 0.36, 0.36],  # Magic  vs. [Melee, Magic, Ranged]
                                  [0.33, 0.03, 0.33] ] # Ranged vs. [Melee, Magic, Ranged]
        self.ability_block_chances = [0.5, 0.15, 0.15] # 0 = disadvantage combat triangle, 1 = Equal combat triangle, 2 = advantage combat triangle

    def reset_for_next_cycle(self):
        self.player.reset_combat_statuses()
        self.opponent.reset_combat_statuses()

    def update_player(self, player_obj, player_action, random_simulate = False):
        
        # Update player's status and conditions
        if player_action is None:
            player_obj.update_available_abilities()
            if random_simulate:
                # Check available actions
                player_ability = player_obj.get_available_ability()
            else:
                player_ability = None
            # PLayer armor and combat style will remain the same
        else:
            if player_action[1] != None:
                player_obj.set_armor_style(player_action[1]) # Update player armor style
            if player_action[2] != None:
                player_obj.set_combat_style(player_action[2]) # Update player combat style
            player_obj.update_available_abilities() # Update after armor changes
            if player_action[0] != None:
                player_ability = player_obj.get_ability_to_execute(player_action[0]) # The ability selected by the player to execute
            else:
                player_ability = None

        return player_obj, player_ability

    def get_roll_boolean(ratio):
        # Return a boolean value based on a given ratio
        return random.random() < ratio

    def check_combat_triangle(self, attacker, defender):
        # Returns 0 if the defender has an advantage in combat triangle
        # Returns 1 if there are no advantages in combat triangle
        # Returns 2 if the attacker has an advantage in combat triangle

        # 0 = Melee
        # 1 = Magic
        # 2 = Ranged

        # Check if the combat triangle is valid
        if attacker.get_combat_style() == defender.get_armor_style():
            return 1
        elif attacker.get_combat_style_int() == MELEE:
            if defender.get_armor_style_int() == MAGIC:
                return 0
            elif defender.get_armor_style_int() == RANGED:
                return 2
        elif attacker.get_combat_style_int() == MAGIC:
            if defender.get_armor_style_int() == RANGED:
                return 0
            elif defender.get_armor_style_int() == MELEE:
                return 2
        elif attacker.get_combat_style_int() == RANGED:
            if defender.get_armor_style_int() == MELEE:
                return 0
            elif defender.get_armor_style_int() == MAGIC:
                return 2

    def get_block_chance(self, attacker, defender, auto_attack=False):
        if auto_attack:
            # Dealing with Auto-Attacks
            # Check the advantage in the combat triangle
            block_chance = self.aa_block_chances[attacker.get_combat_style_int()][defender.get_combat_style_int()]
            # Check if the armor matches because this will also reduce the accuracy of the attack
            combat_advantage = CombatSimulation.check_combat_triangle(self, attacker, attacker)
            if combat_advantage == 2:
                block_chance += 0.45
            elif combat_advantage == 0:
                block_chance += 0.30
            
        else:
            # Dealing with Abilities
            combat_advantage = CombatSimulation.check_combat_triangle(self, attacker, defender)
            block_chance = self.ability_block_chances[combat_advantage]

            # Check if the armor matches because this will also reduce the accuracy of the attack
            combat_advantage = CombatSimulation.check_combat_triangle(self, attacker, attacker)
            if combat_advantage == 2:
                block_chance += 0.25
            elif combat_advantage == 0:
                block_chance += 0.15
        
        return block_chance

    def get_adrenaline(self, combat_style, auto_attack=False, ability=None):
        # Adrenaline (energy) is gained by:
        # 2% - Auto attack
        # 8% - Basic Ability
        # -15% - Threshold Ability
        # -100% - Ultimate Ability

        # Get the amount of adrenaline generated from the attack
        if auto_attack:
            if combat_style == MELEE:
                return 3
            elif combat_style == MAGIC:
                if random.random() < 0.5:
                    return 2
                else:
                    return 3
            elif combat_style == RANGED:
                return 2
        else:
            if ability.energy_cost == 0:
                return 8
            elif ability.energy_cost == 50:
                return -15
            elif ability.energy_cost == 100:
                return -100

    def perform_stun_logic(player, current_game_tick):
        player.is_stun_immuned, player.stun_immune_ticks_remaining = player.apply_stun_immune_affects(current_game_tick)
        if not player.is_stun_immuned:
            player.is_stunned, player.stun_ticks_remaining = player.apply_stun_effects(current_game_tick)
            if player.is_stunned and player.activated_ability.name == "Freedom":
                player.freedom_pass = True
        return player

    def perform_auto_attack_logic(self, player, opponent, current_game_tick):
        if not player.is_stunned or player.freedom_pass:
            player.aa_performed, player.mainhand_aa_damage, player.offhand_aa_damage = player.auto_attack.perform_auto_attack(current_game_tick)
        # Check Auto-Attack Blocking
        player.aa_mainhand_blocked = CombatSimulation.get_roll_boolean(CombatSimulation.get_block_chance(self, player, opponent, auto_attack=True))
        if player.has_offhand_weapon:
            player.aa_offhand_blocked = CombatSimulation.get_roll_boolean(CombatSimulation.get_block_chance(self, player, opponent, auto_attack=True))
        else:
            player.aa_offhand_blocked = None

        return player, opponent

    def perform_ability_logic(self, player, opponent):
        # Get ability damage and Check blocking chance
        if not player.is_stunned or player.freedom_pass:
            if not player.activated_ability == None:
                player.ability_damage = player.activated_ability.get_ability_damage()
                if player.activated_ability.name == "Freedom" or player.activated_ability.name == "Anticipation":
                    player.ability_blocked = False
                else:
                    player.ability_blocked = CombatSimulation.get_roll_boolean(CombatSimulation.get_block_chance(self, player, opponent, auto_attack=False))
                player.add_used_abilities(player.activated_ability)

        return player, opponent
    
    def perform_auto_attack_damage_logic(self, player, opponent):
        if not player.is_stunned or player.freedom_pass:
            if player.aa_performed:
                if not player.aa_mainhand_blocked:
                    opponent.apply_damage(player.mainhand_aa_damage)
                    player.update_adrenaline(CombatSimulation.get_adrenaline(self, player.get_combat_style_int(), auto_attack=True))
                if player.has_offhand_weapon and not player.aa_offhand_blocked:
                    opponent.apply_damage(player.offhand_aa_damage)
                    player.update_adrenaline(CombatSimulation.get_adrenaline(self, player.get_combat_style_int(), auto_attack=True))

        return player, opponent

    def perform_ability_damage_logic(self, player, opponent, current_game_tick):
        if not player.is_stunned or player.freedom_pass:
            if not player.ability_blocked:
                if not player.activated_ability == None:
                    opponent.apply_damage(player.ability_damage)
                    player.update_adrenaline(CombatSimulation.get_adrenaline(self, player.get_combat_style_int(), auto_attack=False, ability=player.activated_ability))
                    # Apply statuses of ability to opponent
                    for status in player.activated_ability.adds_opponent_status:
                        opponent.apply_statuses(status, current_game_tick, player.ability_damage)
                    for status in player.activated_ability.adds_user_status:
                        player.apply_statuses(status, current_game_tick, player.ability_damage)
        
        return player, opponent

    def simulate(self, player_action = None, opponent_action = None, player1_random_simulate=False, player2_random_simulate=True): # Should be updated every time next action is available
        start_time_sec = time.time_ns() * nanos_to_sec

        # ======================== Reset Local Variables =========================
        CombatSimulation.reset_for_next_cycle(self)

        # ======================== Update Players ==========================         
        self.player, self.player.activated_ability     = CombatSimulation.update_player(self, self.player, player_action, random_simulate=player1_random_simulate)
        self.opponent, self.opponent.activated_ability = CombatSimulation.update_player(self, self.opponent, opponent_action, random_simulate=player2_random_simulate)

        self.game_tick_at_begginning = self.current_game_tick
        # ======================== Update Loop ==========================
        for current_game_tick in range(self.current_game_tick + 1, self.current_game_tick + 4):
            #print(f"Current game tick: {current_game_tick}")
            
            # Due to the fact that abilities cannot be used between 3 ticks of each other, simulate will process 3 ticks at a time
            # Update the current game tick
            self.current_game_tick = current_game_tick

            # ======================== Update Cooldowns =========================
            self.player.update_cooldowns()
            self.opponent.update_cooldowns()

            # ======================== Apply Bleeds ==========================
            self.player.is_bleeding, self.player.bleed_damage = self.player.apply_bleed_effects(current_game_tick)
            self.opponent.is_bleeding, self.opponent.bleed_damage = self.opponent.apply_bleed_effects(current_game_tick)

            # ======================== Apply Stuns =========================
            self.player = CombatSimulation.perform_stun_logic(self.player, current_game_tick)
            self.opponent = CombatSimulation.perform_stun_logic(self.opponent, current_game_tick)
            
            if current_game_tick == self.game_tick_at_begginning + 1:
                # ======================== Apply Auto Attacks =========================
                self.player, self.opponent = CombatSimulation.perform_auto_attack_logic(self, self.player, self.opponent, current_game_tick)
                self.opponent, self.player = CombatSimulation.perform_auto_attack_logic(self, self.opponent, self.player, current_game_tick)
                    
                # ======================== Apply Abilities =========================
                # Get ability damage and Check blocking chance
                self.player, self.opponent = CombatSimulation.perform_ability_logic(self, self.player, self.opponent)
                self.opponent, self.player = CombatSimulation.perform_ability_logic(self, self.opponent, self.player)

                # ======================== Damage Calculation =========================
                self.player, self.opponent = CombatSimulation.perform_auto_attack_damage_logic(self, self.player, self.opponent)
                self.opponent, self.player = CombatSimulation.perform_auto_attack_damage_logic(self, self.opponent, self.player)

                self.player, self.opponent = CombatSimulation.perform_ability_damage_logic(self, self.player, self.opponent, current_game_tick)
                self.opponent, self.player = CombatSimulation.perform_ability_damage_logic(self, self.opponent, self.player, current_game_tick)
        
        if self.verbosity == 2:
            CombatSimulation.print_status(self, self.player)
            CombatSimulation.print_status(self, self.opponent)
            print(f"Iteration Time: {self.iteration_time}")
        if self.verbosity == 1:
            print(f"Game tick: {current_game_tick}, {self.player.name} Health: {self.player.health}, {self.opponent.name} Health: {self.opponent.health},", end="")
            if self.player.activated_ability != None:
                print(f" Player1 Ability Used: {self.player.activated_ability.name},", end="")
            else:
                print(f" Player1 Ability Used: None,", end="")
            if self.opponent.activated_ability != None:
                print(f" Player2 Ability Used: {self.opponent.activated_ability.name}")
            else:
                print(f" Player2 Ability Used: None")
        end_time_sec = time.time_ns() * nanos_to_sec
        self.iteration_time = end_time_sec - start_time_sec

        return self.player, self.opponent, self.iteration_time

    def print_status(self, player):
        print(f"=============== {player.name} =============== (Gametick:{self.game_tick_at_begginning + 1})")
        print(f"-Health: {player.health}")
        if player.activated_ability != None:
            print(f"-Ability Used: {player.activated_ability.name} | Damage: {player.ability_damage} | Blocked: {player.ability_blocked}")
        else:
            print(f"-Ability Used: None ")
        print(f"-Auto Attack Performed: {player.aa_performed} | Mainhand Damage: {player.mainhand_aa_damage} | Offhand Damage: {player.offhand_aa_damage} | Blocked Mainhand: {player.aa_mainhand_blocked} | Blocked Offhand: {player.aa_offhand_blocked}")
        print(f"-Adrenaline Level: {player.adrenaline_level}")
        print(f"-Bleeding: {player.is_bleeding} | Bleeding Damage this tick: {player.bleed_damage}")
        print(f"-Stunned: {player.is_stunned} | Stun ticks remaining: {player.stun_ticks_remaining}")
        print(f"-Stun Immuned: {player.is_stun_immuned} | Stun immune ticks remaining: {player.stun_immune_ticks_remaining}")
        print(f"-Cooldowns: {player.cooldowns}")
        print(f"-Available Abilities: {player.available_abilities}")

class Abilities:
    def __init__(self, filename):
        self.filename = filename
        self.melee_abilities_basic = {}
        self.melee_abilities_threshold = {}
        self.melee_abilities_ultimate = {}
        self.ranged_abilities_basic = {}
        self.ranged_abilites_threshold = {}
        self.ranged_abilities_ultimate = {}
        self.magic_abilities_basic = {}
        self.magic_abilities_threshold = {}
        self.magic_abilities_ultimate = {}
        # Abilities that can be used for all styles of combat
        self.all_abilities_basic = {}
        self.all_abilities_threshold = {} 
        self.all_abilities_ultimate = {}
        self.include_all_abilities = True

        self.shared_cooldowns = {"Kick", "Backhand", "Impact", "Shock", "Binding Shot", "Demoralise"}

    def bind_abilities_to_player(self, melee_ability_damage, ranged_ability_damage, magic_ability_damage):
        for ability in self.melee_abilities_basic.keys():
            tmp_ability = self.melee_abilities_basic[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.melee_abilities_basic[ability] = tmp_ability
        for ability in self.melee_abilities_threshold.keys():
            tmp_ability = self.melee_abilities_threshold[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.melee_abilities_threshold[ability] = tmp_ability
        for ability in self.melee_abilities_ultimate.keys():
            tmp_ability = self.melee_abilities_ultimate[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.melee_abilities_ultimate[ability] = tmp_ability
        for ability in self.ranged_abilities_basic.keys():
            tmp_ability = self.ranged_abilities_basic[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.ranged_abilities_basic[ability] = tmp_ability
        for ability in self.ranged_abilites_threshold.keys():
            tmp_ability = self.ranged_abilites_threshold[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.ranged_abilites_threshold[ability] = tmp_ability
        for ability in self.ranged_abilities_ultimate.keys():
            tmp_ability = self.ranged_abilities_ultimate[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.ranged_abilities_ultimate[ability] = tmp_ability
        for ability in self.magic_abilities_basic.keys():
            tmp_ability = self.magic_abilities_basic[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.magic_abilities_basic[ability] = tmp_ability
        for ability in self.magic_abilities_threshold.keys():
            tmp_ability = self.magic_abilities_threshold[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.magic_abilities_threshold[ability] = tmp_ability
        for ability in self.magic_abilities_ultimate.keys():
            tmp_ability = self.magic_abilities_ultimate[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.magic_abilities_ultimate[ability] = tmp_ability
        for ability in self.all_abilities_basic.keys():
            tmp_ability = self.all_abilities_basic[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.all_abilities_basic[ability] = tmp_ability
        for ability in self.all_abilities_threshold.keys():
            tmp_ability = self.all_abilities_threshold[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.all_abilities_threshold[ability] = tmp_ability
        for ability in self.all_abilities_ultimate.keys():
            tmp_ability = self.all_abilities_ultimate[ability]
            tmp_ability.bind_ability_to_player(melee_ability_damage, ranged_ability_damage, magic_ability_damage)
            self.all_abilities_ultimate[ability] = tmp_ability

    def get_available_abilities(self, combat_style, adrenaline_level):
        if adrenaline_level > 100:
            adrenaline_level = 100
        # Return a list of all available abilities for the given combat style and adrenaline level
        if combat_style == "Melee":
            if adrenaline_level < 50:
                available_abilities = set(self.melee_abilities_basic.keys())
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                return available_abilities
            elif adrenaline_level < 100 and adrenaline_level >= 50:
                available_abilities = set(self.melee_abilities_basic.keys())
                available_abilities = available_abilities.union(set(self.melee_abilities_threshold.keys()))
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_threshold.keys()))
                return available_abilities
            elif adrenaline_level == 100:
                available_abilities = set(self.melee_abilities_basic.keys())
                available_abilities = available_abilities.union(set(self.melee_abilities_threshold.keys()))
                available_abilities = available_abilities.union(set(self.melee_abilities_ultimate.keys()))
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_threshold.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_ultimate.keys()))
                return available_abilities
        elif combat_style == "Ranged":
            if adrenaline_level < 50:
                available_abilities = set(self.ranged_abilities_basic.keys())
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                return available_abilities
            elif adrenaline_level < 100 and adrenaline_level >= 50:
                available_abilities = set(self.ranged_abilities_basic.keys())
                available_abilities = available_abilities.union(set(self.ranged_abilites_threshold.keys()))
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_threshold.keys()))
                return available_abilities
            elif adrenaline_level == 100:
                available_abilities = set(self.ranged_abilities_basic.keys())
                available_abilities = available_abilities.union(set(self.ranged_abilites_threshold.keys()))
                available_abilities = available_abilities.union(set(self.ranged_abilities_ultimate.keys()))
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_threshold.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_ultimate.keys()))
                return available_abilities
        elif combat_style == "Magic":
            if adrenaline_level < 50:
                available_abilities = set(self.magic_abilities_basic.keys())
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                return available_abilities
            elif adrenaline_level < 100 and adrenaline_level >= 50:
                available_abilities = set(self.magic_abilities_basic.keys())
                available_abilities = available_abilities.union(set(self.magic_abilities_threshold.keys()))
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_threshold.keys()))
                return available_abilities
            elif adrenaline_level == 100:
                available_abilities = set(self.magic_abilities_basic.keys())
                available_abilities = available_abilities.union(set(self.magic_abilities_threshold.keys()))
                available_abilities = available_abilities.union(set(self.magic_abilities_ultimate.keys()))
                if self.include_all_abilities:
                    available_abilities = available_abilities.union(set(self.all_abilities_basic.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_threshold.keys()))
                    available_abilities = available_abilities.union(set(self.all_abilities_ultimate.keys()))
                return available_abilities
        else:
            raise Exception("Invalid combat style")

    def get_ability(self, ability_name):
        # Return the ability with the given name
        if ability_name in self.melee_abilities_basic:
            return self.melee_abilities_basic[ability_name]
        elif ability_name in self.melee_abilities_threshold:
            return self.melee_abilities_threshold[ability_name]
        elif ability_name in self.melee_abilities_ultimate:
            return self.melee_abilities_ultimate[ability_name]
        elif ability_name in self.ranged_abilities_basic:
            return self.ranged_abilities_basic[ability_name]
        elif ability_name in self.ranged_abilites_threshold:
            return self.ranged_abilites_threshold[ability_name]
        elif ability_name in self.ranged_abilities_ultimate:
            return self.ranged_abilities_ultimate[ability_name]
        elif ability_name in self.magic_abilities_basic:
            return self.magic_abilities_basic[ability_name]
        elif ability_name in self.magic_abilities_threshold:
            return self.magic_abilities_threshold[ability_name]
        elif ability_name in self.magic_abilities_ultimate:
            return self.magic_abilities_ultimate[ability_name]
        elif ability_name in self.all_abilities_basic:
            return self.all_abilities_basic[ability_name]
        elif ability_name in self.all_abilities_threshold:
            return self.all_abilities_threshold[ability_name]
        elif ability_name in self.all_abilities_ultimate:
            return self.all_abilities_ultimate[ability_name]
        else:
            raise Exception("Invalid ability name")

    def parse_abilities(self):
        # Open the file to parse all ability parameters

        with open(self.filename, newline='') as csvfile:
            # Create a CSV reader object
            reader = csv.reader(csvfile, delimiter=',')          

            # Loop through each row in the CSV file
            for row in reader:
                # Append each value to the appropriate list

                if row[0] == "Class":
                    continue

                condition_strings = row[15].split()
                if condition_strings:
                    tmp_condition = Condition(condition_strings, int(row[16]), int(row[17]))
                else:
                    tmp_condition = Condition()

                # Statuses added to Opponent
                opponent_status_strings = row[14].split()
                opponent_statuses = []
                if opponent_status_strings:
                    for status in opponent_status_strings:
                        tmp_opponent_status = Status(status)
                        opponent_statuses.append(tmp_opponent_status)

                user_status_strings = row[12].split()
                user_statuses = []
                if user_status_strings:
                    for status in user_status_strings:
                        tmp_user_status = Status(status, float(row[13]))
                        user_statuses.append(tmp_user_status)

                tmp_ability = Ability(row[0],
                                    row[1],
                                    row[2],
                                    int(row[3]),
                                    int(row[4]),
                                    float(row[5]),
                                    float(row[6]),
                                    float(row[7]),
                                    int(row[8]),
                                    int(row[9]),
                                    int(row[10]),
                                    row[11],
                                    user_statuses,
                                    opponent_statuses,
                                    row[18],
                                    tmp_condition,
                                    row[19])
                
                if tmp_ability.class_type == "Melee":
                    if tmp_ability.energy_cost == 0:
                        self.melee_abilities_basic[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 50:
                        self.melee_abilities_threshold[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 100:
                        self.melee_abilities_ultimate[tmp_ability.name] = tmp_ability
                elif tmp_ability.class_type == "Ranged":
                    if tmp_ability.energy_cost == 0:
                        self.ranged_abilities_basic[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 50:
                        self.ranged_abilites_threshold[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 100:
                        self.ranged_abilities_ultimate[tmp_ability.name] = tmp_ability
                elif tmp_ability.class_type == "Magic":
                    if tmp_ability.energy_cost == 0:
                        self.magic_abilities_basic[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 50:
                        self.magic_abilities_threshold[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 100:
                        self.magic_abilities_ultimate[tmp_ability.name] = tmp_ability
                elif tmp_ability.class_type == "All":
                    if tmp_ability.energy_cost == 0:
                        self.all_abilities_basic[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 50:
                        self.all_abilities_threshold[tmp_ability.name] = tmp_ability
                    elif tmp_ability.energy_cost == 100:
                        self.all_abilities_ultimate[tmp_ability.name] = tmp_ability           

class Ability:
    def __init__(self,
                 class_type,
                 subclass_type,
                 name,
                 damage_min,
                 damage_max,
                 ability_duration,
                 stun_duration,
                 bind_duration,
                 bleed_hits,
                 number_of_hits,
                 energy_cost,
                 defence_boost,
                 adds_user_status,
                 adds_opponent_status,
                 cooldown_duration,
                 conditions,
                 keybind):

        self.class_type = class_type
        self.class_type_int = 0
        self.subclass_type = subclass_type
        self.name = name
        self.damage_min = damage_min
        self.damage_max = damage_max
        self.ability_duration = ability_duration
        self.stun_duration = stun_duration
        self.stun_duration_game_ticks = math.ceil(stun_duration / game_tick_duration)
        self.bind_duration = bind_duration
        self.bind_duration_game_ticks = math.ceil(bind_duration / game_tick_duration)
        self.bleed_hits = bleed_hits
        self.number_of_hits = number_of_hits
        self.energy_cost = energy_cost
        self.defence_boost = defence_boost
        self.adds_user_status = adds_user_status
        self.adds_opponent_status = adds_opponent_status
        self.cooldown_duration = cooldown_duration
        self.cooldown_duration_game_ticks = math.ceil(float(cooldown_duration) / game_tick_duration)
        self.conditions = conditions
        self.keybind = keybind

        # Status initialization
        if self.adds_opponent_status:
            for i in range(0, len(self.adds_opponent_status)):
                if self.adds_opponent_status[i].name == "Stun":
                    self.adds_opponent_status[i].duration = self.stun_duration_game_ticks
                elif self.adds_opponent_status[i].name == "Bind":
                    self.adds_opponent_status[i].duration = self.bind_duration_game_ticks
                elif self.adds_opponent_status[i].name == "Bleed":
                    self.adds_opponent_status[i].duration = self.bleed_hits


    def bind_ability_to_player(self, melee_ability_damage, ranged_ability_damage, magic_ability_damage):
        if self.class_type == "Melee":
            self.damage_min = (self.damage_min / 100) * melee_ability_damage
            self.damage_max = (self.damage_max / 100) * melee_ability_damage
            self.class_type_int = 0
        elif self.class_type == "Ranged":
            self.damage_min = (self.damage_min / 100) * ranged_ability_damage
            self.damage_max = (self.damage_max / 100) * ranged_ability_damage
            self.class_type_int = 1
        elif self.class_type == "Magic":
            self.damage_min = (self.damage_min / 100) * magic_ability_damage
            self.damage_max = (self.damage_max / 100) * magic_ability_damage
            self.class_type_int = 2

    def get_ability_damage(self):
        if self.damage_max == 0 and self.damage_min == 0: # Ability has no damage
            return 0
        # Generate a random damage value using a normal distribution with mean (max_damage + min_damage) / 2
        # and standard deviation (max_damage - min_damage) / 4
        damage = np.random.normal(loc=(self.damage_min + self.damage_max) / 2, scale=(self.damage_max - self.damage_min) / 4)
        # Ensure the damage value is within the valid range
        damage = max(self.damage_min, min(self.damage_max, damage))
        return damage
    
    def get_average_ability_damage(self):
        if self.damage_max == 0 and self.damage_min == 0: # Ability has no damage
            return 0
        damage = (self.damage_min + self.damage_max) / 2
        return damage
        
class Status:
    def __init__(self, name, duration=0):
        self.name = name
        self.duration = duration
        self.duration_game_ticks = math.ceil(float(duration) / game_tick_duration)
        self.damage = 0
        self.ticks_remaining = 0
        self.starting_game_tick = 0
        self.game_ticks_affected = []

class Condition:
    def __init__(self, names = None, condition_damage_min = None, condition_damage_max = None):
        self.name = names
        self.condition_damage_min = condition_damage_min
        self.condition_damage_max = condition_damage_max
    
    def check_condition(self, status):
        if status.name == self.name:
            return True
        else:
            return False

class Levels:
    def __init__(self):
        self.hitpoints = 0
        self.attack = 0
        self.strength = 0
        self.defence = 0
        self.magic = 0
        self.ranged = 0

class Weapon:
    def __init__(self):
        self.name = None
        self.type = None
        self.damage = 0
        self.spell_damage = 0
        self.accuracy = 0
        self.speed = 0 # 0 = Slowest, 1 = Average, 2 = Fast, 3 = Fastest
        self.range = 0
        self.level = 0
        self.additional_bonuses = 0
        self.weapon_type = 0 # 0 = mainhand, 1 = offhand, 2 = 2_handed
        
class Player:
    def __init__(self, player_name, levels):
        self.name = player_name
        self.health = levels.hitpoints
        self.levels = levels
        self.abilities = None
        self.combat_style = None
        self.combat_style_int = 0
        self.armor_style = None
        self.armor_style_int = 0
        self.adrenaline_level = 0
        self.statuses = set()
        self.cooldowns = {}
        self.used_abilities = set()
        self.available_abilities = set()
        self.bleed_list = [] # List of bleed objects
        self.stun_list = [] # List of stun objects
        self.bind_list = [] # List of blind objects
        self.stun_immune_list = [] # List of stun immune objects
        self.auto_attack = AutoAttack()

        self.has_offhand_weapon = False
        self.melee_weapon_mainhand = None
        self.melee_weapon_offhand = None
        self.magic_weapon_mainhand = None
        self.magic_weapon_offhand = None
        self.ranged_weapon_mainhand = None
        self.ranged_weapon_offhand = None

        # Need to call calculate_ability_damage_multiplier() to initialize these after
        # Weapons have been loaded
        self.melee_ability_damage_mainhand = 0
        self.melee_ability_damage_offhand = 0
        self.melee_ability_damage = 0
        self.magic_ability_damage_mainhand = 0
        self.magic_ability_damage_offhand = 0
        self.magic_ability_damage = 0
        self.ranged_ability_damage_mainhand = 0
        self.ranged_ability_damage_offhand = 0
        self.ranged_ability_damage = 0

        # Speed multipliers for melee weapons
        self.s_melee_multiplier = {1: 0.644295, 2: 0.783673, 3: 1}

        # Statuses for combat simulation
        self.activated_ability = None
        self.aa_performed = False
        self.mainhand_aa_damage = 0
        self.offhand_aa_damage = 0
        self.aa_mainhand_blocked = False
        self.aa_offhand_blocked = False
        self.ability_damage = 0
        self.ability_blocked = False
        self.is_stunned = False
        self.stun_ticks_remaining = 0
        self.is_bleeding = False
        self.bleed_damage = 0
        self.is_stun_immuned = False
        self.stun_immune_ticks_remaining = 0
        self.freedom_pass = False
        
    def print_status(self):
        print("Player: " + self.name)
        print("Health: " + str(self.health))
        print("Armor: " + str(self.armor_style))
        print("Combat Style: " + str(self.combat_style))
        print("Adrenaline: " + str(self.adrenaline_level))

    def get_state(self):

        if self.combat_style_int == 0: # Melee
            combat_vect = [1,0,0]
        elif self.combat_style_int == 1: # Magic
            combat_vect = [0,1,0]
        elif self.combat_style_int == 2: # Ranged
            combat_vect = [0,0,1]
        
        if self.armor_style_int == 0: # Melee
            armor_vect = [1,0,0]
        elif self.armor_style_int == 1: # Magic
            armor_vect = [0,1,0]
        elif self.armor_style_int == 2: # Ranged
            armor_vect = [0,0,1]

        stun = 0
        bleed = 0
        stun_immune = 0
        if "Stun" in self.statuses:
            stun = 1
        elif "Bleed" in self.statuses:
            bleed = 1
        elif "StunImmune" in self.statuses:
            stun_immune = 1

        statuses_vect = [stun, bleed, stun_immune]

        if self.adrenaline_level < 50:
            adrenaline_vect = [1,0,0]
        elif self.adrenaline_level < 100 and self.adrenaline_level >= 50:
            adrenaline_vect = [0,1,0]
        elif self.adrenaline_level == 100:
            adrenaline_vect = [0,0,1]

        combined = np.stack((combat_vect, armor_vect, statuses_vect, adrenaline_vect), axis=0)

        flattened = combined.flatten()

        binary = binatodeci(flattened)

        # Returns One-hot Vector [armor, combat, statuses, adrenaline]
        return flattened, binary   

    def reset_combat_statuses(self):
        self.activated_ability = None
        self.aa_performed = False
        self.mainhand_aa_damage = 0
        self.offhand_aa_damage = 0
        self.aa_mainhand_blocked = False
        self.aa_offhand_blocked = False
        self.ability_damage = 0
        self.ability_blocked = False
        self.is_stunned = False
        self.stun_ticks_remaining = 0
        self.is_bleeding = False
        self.bleed_damage = 0
        self.is_stun_immuned = False
        self.stun_immune_ticks_remaining = 0
        self.freedom_pass = False

    def reset_player(self):
        self.health = self.levels.hitpoints
        self.adrenaline_level = 0
        self.statuses = set()
        self.cooldowns = {}
        self.used_abilities = set()
        self.available_abilities = set()
        self.bleed_list = [] # List of bleed objects
        self.stun_list = [] # List of stun objects
        self.bind_list = [] # List of blind objects
        self.stun_immune_list = [] # List of stun immune objects
        self.auto_attack.previous_game_tick = 0
        self.auto_attack.next_game_tick = 1

    def bind_abilities(self, abilities):
        self.abilities = abilities
        self.abilities.bind_ability_to_player(self.melee_ability_damage_mainhand, self.melee_ability_damage_offhand, self.magic_ability_damage_mainhand)

    def apply_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.health = 0
        return self.health

    def calculate_ability_damage_multiplier(self):
        if self.melee_weapon_offhand is not None:
            self.melee_ability_damage_mainhand = (2.5 * self.levels.strength) + (self.melee_weapon_mainhand.damage * self.s_melee_multiplier[self.melee_weapon_mainhand.speed]) + self.melee_weapon_mainhand.additional_bonuses
            self.melee_ability_damage_offhand = (1.25 * self.levels.strength) + (self.melee_weapon_offhand.damage * self.s_melee_multiplier[self.melee_weapon_mainhand.speed]) + (0.5 * self.melee_weapon_offhand.additional_bonuses)
        else: # 2 handed weapon (should eventually take into account shields)
            self.melee_ability_damage_mainhand = (3.75 * self.levels.strength) + (self.melee_weapon_mainhand.damage * self.s_melee_multiplier[self.melee_weapon_mainhand.speed]) + (1.5 * self.melee_weapon_mainhand.additional_bonuses)
            self.melee_ability_damage_offhand = 0
        self.melee_ability_damage = self.melee_ability_damage_mainhand + self.melee_ability_damage_offhand

        if self.magic_weapon_offhand is not None:
            self.magic_ability_damage_mainhand = (2.5 * self.levels.magic) + min(9.6 * self.magic_weapon_mainhand.level, self.magic_weapon_mainhand.spell_damage) + self.magic_weapon_mainhand.additional_bonuses
            self.magic_ability_damage_offhand = (1.25 * self.levels.magic) + min(4.8 * self.magic_weapon_offhand.level, 0.5 * self.magic_weapon_offhand.spell_damage) + (0.5 * self.magic_weapon_offhand.additional_bonuses)
        else: # 2 handed weapon (should eventually take into account shields)
            self.magic_ability_damage_mainhand = (3.75 * self.levels.magic) + min(14.4 * self.magic_weapon_mainhand.level, 1.5 * self.magic_weapon_mainhand.spell_damage) + (1.5 * self.magic_weapon_mainhand.additional_bonuses)
            self.magic_ability_damage_offhand = 0
        self.magic_ability_damage = self.magic_ability_damage_mainhand + self.magic_ability_damage_offhand

        if self.ranged_weapon_offhand is not None:
            self.ranged_ability_damage_mainhand = (2.5 * self.levels.ranged) + min(9.6 * self.ranged_weapon_mainhand.level, self.ranged_weapon_mainhand.damage) + self.ranged_weapon_mainhand.additional_bonuses
            self.ranged_ability_damage_offhand = (1.25 * self.levels.ranged) + min(4.8 * self.ranged_weapon_offhand.level, 0.5 * self.ranged_weapon_offhand.damage) + self.ranged_weapon_offhand.additional_bonuses
        else: # 2 handed weapon (should eventually take into account shields)
            self.ranged_ability_damage_mainhand = (3.75 * self.levels.ranged) + min(14.4 * self.ranged_weapon_mainhand.level, 1.5 * self.ranged_weapon_mainhand.damage) + self.ranged_weapon_mainhand.additional_bonuses
            self.ranged_ability_damage_offhand = 0
        self.ranged_ability_damage = self.ranged_ability_damage_mainhand + self.ranged_ability_damage_offhand
    

        # Update the autoattack object as well
        self.auto_attack.melee_ability_damage_mainhand = self.melee_ability_damage_mainhand
        self.auto_attack.melee_ability_damage_offhand = self.melee_ability_damage_offhand
        self.auto_attack.magic_ability_damage_mainhand = self.magic_ability_damage_mainhand
        self.auto_attack.magic_ability_damage_offhand = self.magic_ability_damage_offhand
        self.auto_attack.ranged_ability_damage_mainhand = self.ranged_ability_damage_mainhand
        self.auto_attack.ranged_ability_damage_offhand = self.ranged_ability_damage_offhand

    def apply_statuses(self, statuses, current_game_tick, damage = None):
        if not isinstance(statuses, list):
            statuses = [statuses]
        for status in statuses:
                if status.name == "Bleed":
                    Player.add_bleed_effect(self, status, damage, current_game_tick)
                elif status.name == "Stun":
                    Player.add_stun_effect(self, status, current_game_tick)
                elif status.name == "Stun_Immune":
                    Player.add_stun_immune_affect(self, status, current_game_tick)
                # We are not including binds because players in project are stationary, and they are not affected by binds
                    
    def add_stun_immune_affect(self, status, starting_game_tick):
        # Stun immunity does not stack
        self.stun_immune_list.clear()
        self.statuses.add(status.name) # Add the status to the statuses
        status.ticks_remaining = math.floor((status.duration / game_tick_duration))
        status.damage = 0
        status.game_ticks_affected = range(starting_game_tick, starting_game_tick + (status.ticks_remaining), 1)
        self.stun_immune_list.append(status)
    
    def apply_stun_immune_affects(self, game_tick):
        stun_immune = False
        ticks_remaining = 0
        for status in self.stun_immune_list:
            if status.ticks_remaining > 0 and game_tick in status.game_ticks_affected:
                status.ticks_remaining -= 1 # decrease the ticks remaining
                stun_immune = True
                ticks_remaining = status.ticks_remaining
                if status.ticks_remaining == 0:
                    self.stun_immune_list.remove(status)
                    self.statuses.remove(status.name)
        return stun_immune, ticks_remaining

    def add_stun_effect(self, status, starting_game_tick):
        # Check first if Stun Immune is already in self.statuses
        if not "Stun_Immune" in self.statuses:
            self.statuses.add(status.name) # Add the status to the statuses
            status.ticks_remaining = status.duration
            status.damage = 0
            status.game_ticks_affected = range(starting_game_tick, starting_game_tick + (status.ticks_remaining), 1)
            self.stun_list.append(status)

    def apply_stun_effects(self, game_tick):
        stunned = False
        ticks_remaining = 0
        for status in self.stun_list:
            if status.ticks_remaining > 0 and game_tick in status.game_ticks_affected:
                stunned = True
                status.ticks_remaining -= 1 # decrease the ticks remaining
                ticks_remaining = status.ticks_remaining
                if status.ticks_remaining == 0:
                    self.stun_list.remove(status)
                    self.statuses.remove(status.name)

        return stunned, ticks_remaining

    def add_bleed_effect(self, status, damage, starting_game_tick):
        self.statuses.add(status.name) # Add the status to the statuses
        status.ticks_remaining = status.duration
        status.damage = damage
        status.game_ticks_affected = range(starting_game_tick, starting_game_tick + (status.ticks_remaining * 2), 2)
        self.bleed_list.append(status)

    def apply_bleed_effects(self, game_tick):
        bleeding = False
        bleed_damage = 0
        for status in self.bleed_list:
            bleeding = True
            if status.ticks_remaining > 0 and game_tick in status.game_ticks_affected:
                Player.apply_damage(self, status.damage)
                status.ticks_remaining -= 1 # decrease the ticks remaining
                bleed_damage = status.damage
                if status.ticks_remaining == 0:
                    self.bleed_list.remove(status)
                    self.statuses.remove(status.name)
                    bleeding = False

        return bleeding, bleed_damage
    
    def update_available_abilities(self):
        # Check adrenaline level
        available_abilities = self.abilities.get_available_abilities(self.combat_style, self.get_adrenaline())
        self.used_abilities = set(self.cooldowns.keys())
        self.available_abilities = available_abilities.difference(self.used_abilities)

    def add_used_abilities(self, ability):
        # Add the ability to the used_abilities and set the cooldown
        self.used_abilities.add(ability)
        self.cooldowns[ability.name] = ability.cooldown_duration_game_ticks
        if ability.name in self.abilities.shared_cooldowns:
            for ability_shared_cooldown in self.abilities.shared_cooldowns:
                self.cooldowns[ability_shared_cooldown] = ability.cooldown_duration_game_ticks
        
        # Special Ability Effects here
        if ability.name == "Freedom":
            self.stun_list.clear()
            self.bleed_list.clear()
            if "Stun" in self.statuses:
                self.statuses.remove("Stun")
            if "Bleed" in self.statuses:
                self.statuses.remove("Bleed")
            if "Bind" in self.statuses:
                self.statuses.remove("Bind")

    def update_cooldowns(self):
        # Update the cooldowns
        abilities_to_remove = []
        for ability in self.cooldowns:
            self.cooldowns[ability] -= 1 # decrease the cooldown
            if self.cooldowns[ability] == 0:
                abilities_to_remove.append(ability)

        for ability in abilities_to_remove:
            del self.cooldowns[ability]
    
    def get_ability_to_execute(self, ability_name):
        if self.available_abilities:
            if ability_name in self.available_abilities:
                ability = self.abilities.get_ability(ability_name)
        else:
            ability = None
        return ability

    def get_available_ability(self):
        if self.available_abilities:
            ability = self.abilities.get_ability(random.choice(list(self.available_abilities)))
        else:
            ability = None
        return ability

    def set_health(self, max_health):
        self.max_health = max_health
    
    def get_health(self):
        return self.max_health
    
    def set_combat_style(self, combat_style):
        self.combat_style = combat_style
        self.auto_attack.set_combat_style(combat_style) # Update the auto attacks

        if combat_style == "Melee":
            self.combat_style_int = MELEE
            self.has_offhand_weapon = False # These are solely for the project (to increase compuation speed)
        elif combat_style == "Magic":
            self.combat_style_int = MAGIC
            self.has_offhand_weapon = True  # These are solely for the project (to increase compuation speed)
        elif combat_style == "Ranged":
            self.combat_style_int = RANGED
            self.has_offhand_weapon = False # These are solely for the project (to increase compuation speed)

    def get_combat_style(self):
        return self.combat_style
    
    def get_combat_style_int(self):
        return self.combat_style_int
    
    def set_armor_style(self, armor_style):
        self.armor_style = armor_style
        if armor_style == "Melee":
            self.class_type_int = MELEE
        elif armor_style == "Magic":
            self.class_type_int = MAGIC
        elif armor_style == "Ranged":
            self.class_type_int = RANGED
    
    def get_armor_style(self):
        return self.armor_style
    
    def get_armor_style_int(self):
        return self.class_type_int

    def set_adrenaline(self, adrenaline_level):
        self.adrenaline_level = adrenaline_level
    
    def get_adrenaline(self):
        return self.adrenaline_level
    
    def update_adrenaline(self, adrenaline_to_add):
        self.adrenaline_level += adrenaline_to_add
        if self.adrenaline_level > 100:
            self.adrenaline_level == 100
    
    def clone(self):
        return Player(self.health)

class AutoAttack:
    def __init__(self, combat_style = None, starting_game_tick = 1):
        # Readjust auto attacks based on combat style
        # - Auto Attacks -
        # MELEE
        # Fastest: 4 per hit (Claws, daggers, defenders, maces, scimitars, whips)
        # Fast: 5 per hit (Hastae, hatchets, longswords, rapiers, swords)
        # Average: 6 per hit (Battleaxes, halberds, most mauls, pickaxes, warhammers, spears, two-handed swords)
        # MAGIC
        # Combat spells cast with wands or orbs: 4 per cast
        # Combat spells cast with Staves: 6 per cast
        # RANGED
        # Fastest: 4 per hit (Crossbows, darts, hexhunter bow, throwing knives, seercull)
        # Fast: 5 per hit (Most shortbows, Zaryte bow, Crystal bow, Chinchompas)
        # Average: 6 per hit (Javelins, shieldbows (except the dark bow), salamanders, thrown axes, two-handed crossbows)
        # Slowest: 12 per hit (Dark bow)
        self.previous_game_tick = starting_game_tick
        self.next_game_tick = 1
        self.combat_style = combat_style

        self.melee_ability_damage_mainhand = 0
        self.melee_ability_damage_offhand = 0
        self.magic_ability_damage_mainhand = 0
        self.magic_ability_damage_offhand = 0
        self.ranged_ability_damage_mainhand = 0
        self.ranged_ability_damage_offhand = 0

    def set_combat_style(self, combat_style):
        self.combat_style = combat_style

    def perform_auto_attack(self, current_game_tick):
        if current_game_tick == self.next_game_tick:
            mainhand_damage, offhand_damage = AutoAttack.get_auto_attack_damage(self)
            AutoAttack.update_next_auto_attack(self, current_game_tick)
            return 1, mainhand_damage, offhand_damage
        else:
            return 0, 0, 0

    def update_next_auto_attack(self, current_game_tick):
        self.previous_game_tick = current_game_tick
        if self.combat_style == "Melee":
            self.next_game_tick = self.previous_game_tick + 4
        elif self.combat_style == "Ranged":
            self.next_game_tick = self.previous_game_tick + 6
        elif self.combat_style == "Magic":
            self.next_game_tick = self.previous_game_tick + 4

    def get_auto_attack_damage(self):
        # return the autoattack damage
        if self.combat_style == "Melee":
            mainhand_ability_damage = self.melee_ability_damage_mainhand
            offhand_ability_damage = self.melee_ability_damage_offhand
        elif self.combat_style == "Ranged":
            mainhand_ability_damage = self.ranged_ability_damage_mainhand
            offhand_ability_damage = self.ranged_ability_damage_offhand
        elif self.combat_style == "Magic":
            mainhand_ability_damage = self.magic_ability_damage_mainhand
            offhand_ability_damage = self.magic_ability_damage_offhand

        # Generate a random damage value using a normal distribution with mean (max_damage + min_damage) / 2
        # and standard deviation (max_damage - min_damage) / 4
        mainhand_damage = np.random.normal(loc=(mainhand_ability_damage) / 2, scale=(mainhand_ability_damage - 1) / 4)
        # Ensure the damage value is within the valid range
        mainhand_damage = max(1, min(mainhand_ability_damage, mainhand_damage))

        if offhand_ability_damage != 0:
            offhand_damage = np.random.normal(loc=(offhand_ability_damage) / 2, scale=(offhand_ability_damage - 1) / 4)
            offhand_damage = max(1, min(offhand_ability_damage, offhand_damage))
        else:
            offhand_damage = 0

        return mainhand_damage, offhand_damage

