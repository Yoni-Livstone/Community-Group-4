import logging
import numpy as np
from itertools import combinations


# Global static variables
WEAKNESS_THRESHOLD = 5              # Percentile threshold for weakest player
PAIRING_ADVANTAGE = 3               # Advantage of pairing over individual work
TIRING_TASK_THRESHOLD = 20          # Energy threshold for tiring tasks
EXHAUSTING_TASK_THRESHOLD = 10      # Energy threshold for exhausting tasks
FULL_ENERGY = 10                    # Full energy level


# Set up logging
# Example use: logging.debug("This is a DEBUG message.")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log-results/team4_log.log", mode='w'),
        logging.StreamHandler()         # print to console too
    ]
)


def phaseIpreferences(player, community, global_random):
    '''
    Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id

    This function is part of Phase I where the player decides:
    - If they want to volunteer for a task with a partner (based on energy and task difficulty)
    - If they prefer to work alone on a task (again, based on their energy and the task difficulty)

    Args:
    - player: A player object representing the player making the decision.
    - community: A community object representing the community of players and tasks.
    - global_random: A random number generator.

    Returns:
    - list_choices: A list of task assignments for the player. Each element in the list is a list of two elements.
    '''

    # List to store task choices (task index and partner ID)
    list_choices = []

    # Calculate the cost matrices: individual costs and pair costs for all tasks
    cost_matrix_individual, cost_matrix_pairs = calculate_cost_matrix(community)

    # Rank task assignments (both individual and pairings) based on their cost
    list_of_ranked_assignments = get_ranked_assignments(community, cost_matrix_individual,
                                                        cost_matrix_pairs)
    
    # Iterate through each task in the ranked assignment list
    for t in range(len(list_of_ranked_assignments)):
        # If the task is "exhausting" the player may want to skip or go solo in Phase II
        if list_of_ranked_assignments[t][0][1] >= EXHAUSTING_TASK_THRESHOLD:

            # If the player is the weakest, return an empty list (indicating they won't volunteer)
            if is_weakest_player(player, community):
                return list_choices

        # Initialize the best decision for the current task (no partner, cost is infinity initially)
        best_decision = (None, np.inf)  # (partner_id, cost)

        # Iterate through all ranked assignments for this task (both individual and pair-based)
        for assignment in list_of_ranked_assignments[t]:
            # If the player is part of the current assignment
            if player.id in assignment[0]:
                # Check if the task is tiring
                if TIRING_TASK_THRESHOLD < assignment[1] < EXHAUSTING_TASK_THRESHOLD:
                    # Wait until energy is full to volunteer with a partner
                    if None not in assignment[0] and player.energy == FULL_ENERGY:
                        # Partner ID is the other player in the pair (the one not matching the current player)
                        partner_id = assignment[0][0] if player.id == assignment[0][1] else assignment[0][1]
                        list_choices.append([t, partner_id])
                        # Log the assignment decision
                        logging.info(f"Player {player.id} chose to pair with {partner_id} for task {t}.")
                else:
                    # If the task is easier, the player may volunteer to partner
                    # As long as their current energy is sufficient for the task cost
                    if player.energy >= assignment[1]:
                        if None not in assignment[0]:
                            # If it's a paired assignment, select the best partner based on cost
                            partner_id = assignment[0][0] if player.id == assignment[0][1] else assignment[0][1]
                            if best_decision[0] is not None and assignment[1] < best_decision[1]:
                                best_decision = (partner_id, assignment[1])
                            elif best_decision[0] is None and assignment[1] + PAIRING_ADVANTAGE < best_decision[1]:
                                best_decision = (partner_id, assignment[1])
                        else:
                            # If the assignment is solo, choose the best individual task
                            if assignment[1] < best_decision[1] + PAIRING_ADVANTAGE:
                                best_decision = (None, assignment[1])

        # If a best decision was made for this task (not solo or paired with the best partner), add it to the list
        if best_decision[0] is not None:
            list_choices.append([t, best_decision[0]])
            # Log the decision
            logging.info(f"Player {player.id} chose to pair with {best_decision[0]} for task {t}.")


    return list_choices


def phaseIIpreferences(player, community, global_random):
    '''
    Return a list of tasks for the particular player to do individually

    - Phase II is when players who are not paired with others will choose to take on tasks alone.
    - The player decides which tasks they are willing to volunteer for based on their energy levels and task difficulty.

    Args:
    - player: A player object representing the player making the decision.
    - community: A community object representing the community of players and tasks.
    - global_random: A random number generator.

    Returns:
    - list_tasks: A list of task indices that the player will volunteer to do individually.
    '''

    # List to store task indices the player is willing to do individually
    bids = []

    # Calculate cost matrices for individual and paired tasks
    cost_matrix_individual, cost_matrix_pairs = calculate_cost_matrix(community)

    # Rank the task assignments for all players, both individually and in pairs
    list_of_ranked_assignments = get_ranked_assignments(community, cost_matrix_individual,
                                                        cost_matrix_pairs)
    
    # Iterate through each task in the ranked assignment list
    for t in range(len(list_of_ranked_assignments)):
        # If the task is "exhausting" player may volunteer for this task if they are the weakest player (potentially sacrificing themselves)
        if list_of_ranked_assignments[t][0][1] >= EXHAUSTING_TASK_THRESHOLD:
            # If the player is the weakest in the community, they volunteer for the task (to sacrifice for the team)
            if is_weakest_player(player, community):
                bids.append(t)              # Add this task index to the list of bids
                # Log the decision
                logging.info(f"Player {player.id} volunteered for exhausting task {t} due to being the weakest.")
                return bids                        # Return early as the player has volunteered for a task

        # For easier tasks (energy required < 10), the player can volunteer to work alone if their energy is sufficient
        elif list_of_ranked_assignments[t][0][1] <= TIRING_TASK_THRESHOLD:
            # Iterate over the possible assignments for the task
            for assignment in list_of_ranked_assignments[t]:
                # Check if the player is part of the current assignment, and it is an individual (None represents solo work)
                if player.id in assignment[0] and None in assignment[0] and player.energy >= assignment[1]:
                    # If the player has enough energy for this solo task, they bid for it
                    bids.append(t)
                    # Log the decision
                    logging.info(f"Player {player.id} chose to do task {t} alone.")


    # Return the final list of task indices that the player is willing to do individually
    return bids

# @profile
def calculate_cost_matrix(community):
    """
    Calculates the cost matrices for individual members and pairs based on abilities and tasks.

    The cost is the difference between the task requirements and the player's abilities. The final 
    cost for an individual player or a pair is determined by summing these differences across all 
    task requirements (or abilities) for a given task. For pairs, the combined abilities of the two 
    players are considered, and the cost is halved to represent shared work.

    Args:
        community: An object representing the community, containing a list of members (players) and tasks.
    
    Returns:
        cost_matrix_individual: A dictionary where keys are tuples of the form (player_id, task_index), 
                                and values are the individual costs for that player-task combination.

        cost_matrix_pairs: A dictionary where keys are tuples of the form (player1_id, player2_id, task_index), 
                           and values are the costs for that pair-task combination.

    """
    tasks = community.tasks
    members = community.members

    # Individual costs
    cost_matrix_individual = {}
    for member in members:
        for t, task in enumerate(tasks):
            cost = sum([max(0, task[i] - member.abilities[i]) for i in range(len(member.abilities))])
            cost_matrix_individual[(member.id, t)] = cost

    # Pair costs
    cost_matrix_pairs = {}
    for member1, member2 in combinations(members, 2):
        for t, task in enumerate(tasks):
            combined_abilities = np.maximum(member1.abilities, member2.abilities)
            cost = sum([max(0, task[i] - combined_abilities[i])
                        for i in range(len(combined_abilities))]) / 2  # Half cost for shared work
            cost_matrix_pairs[(member1.id, member2.id, t)] = cost

    return cost_matrix_individual, cost_matrix_pairs

# @profile
def get_ranked_assignments(community, cost_matrix_individual, cost_matrix_pairs):
    """
    Rank assignments of paired and individual workers for each task based on the cost matrices.

    For each task, the function ranks all possible assignments (both individual and paired assignments) 
    based on their respective costs. The assignments are sorted in ascending order of cost, with the lowest 
    cost assignments (best choices) appearing first. This helps in making optimal assignments based on the 
    cost of completing each task, whether individually or in pairs.

    Args:
        community: The community object containing members (players) and tasks.
        cost_matrix_individual: A dictionary that holds the cost of individual assignments (player-task).
        cost_matrix_pairs: A dictionary that holds the cost of paired assignments (pair-task).
    
    Returns:
        list_of_ranked_assignments: A list where each element is a list of ranked assignments for a task. 
                                     Each ranked list contains tuples of (assignment, cost), where assignment 
                                     is either a (player_id, None) for individual assignments or 
                                     (player1_id, player2_id) for pair assignments, and cost is the 
                                     corresponding cost of the assignment.
    """
    tasks = community.tasks
    members = community.members
    
    list_of_ranked_assignments = []

    for t in range(len(tasks)):
        assignments = {}
        for member in members:
            assignments[(member.id, None)] = cost_matrix_individual[(member.id, t)]
        for member1, member2 in combinations(members, 2):
            assignments[(member1.id, member2.id)] = cost_matrix_pairs[(member1.id, member2.id, t)]

        ranked_assignments_dict = dict(sorted(assignments.items(), key=lambda item: item[1]))
        ranked_assignments = list(ranked_assignments_dict.items())
        list_of_ranked_assignments.append(ranked_assignments)

    return list_of_ranked_assignments

# @profile
def is_weakest_player(player, community):
    """
    Determines if a given player is one of the weakest in the community based on their abilities.

    The function compares the total ability score of the player against the abilities of all members in the community. 
    If the player's total abilities are below a specified percentile threshold (set by WEAKNESS_THRESHOLD), 
    they are considered the weakest player.

    Args:
        player: The player whose weakness is being evaluated. The player should have an attribute `abilities` which is a list or array of their abilities.
        community: The community object, which contains a list of all members (players). Each member in the community also has an `abilities` attribute.

    Returns:
        bool: Returns True if the player is the weakest, based on their total ability score compared to the community.
              Returns False otherwise.
    """

    sum_of_abilities = []
    
    for member in community.members:
        sum_of_abilities.append(sum(member.abilities))

    sorted_abilities = np.array(sorted(sum_of_abilities))

    # Calculate the threshold ability score based on the percentile specified by WEAKNESS_THRESHOLD
    threshold = np.percentile(sorted_abilities, WEAKNESS_THRESHOLD)
    
    # Return True if the player's total abilities are below the calculated threshold, indicating they are the weakest
    return sum(player.abilities) < threshold


