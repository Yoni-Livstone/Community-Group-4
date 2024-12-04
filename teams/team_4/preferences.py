import logging
import numpy as np
import pickle
from itertools import combinations
from heapq import heappush, heappop
from itertools import combinations


# Global static variables
WEAKNESS_THRESHOLD = 5          # Percentile threshold for weakest player
DIFFICULTY_ABILITY_RATIO = 2    # Ratio of average difficulty to average ability for strategy selection
PAIRING_ADVANTAGE = 3           # Advantage of pairing over individual work


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log-results/team4_log.log", mode='w'),
        logging.StreamHandler()
    ]
)


# @profile
def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    list_choices = []

    if community.completed_tasks % (len(community.members) * 2) == 0:
        # Calculate cost matrices
        cost_matrix_individual, cost_matrix_pairs = calculate_cost_matrix(community)
        # Rank assignment options by task
        list_of_ranked_assignments = get_ranked_assignments(community, cost_matrix_individual,
                                                            cost_matrix_pairs)
        with open("cost_matrix_individual.pkl", "wb") as f:
            pickle.dump(cost_matrix_individual, f)
        with open("cost_matrix_pairs.pkl", "wb") as f:
            pickle.dump(cost_matrix_pairs, f)
        with open("list_of_ranked_assignments.pkl", "wb") as f:
            pickle.dump(list_of_ranked_assignments, f)
    else:
        with open("cost_matrix_individual.pkl", "rb") as f:
            cost_matrix_individual = pickle.load(f)
        with open("cost_matrix_pairs.pkl", "rb") as f:
            cost_matrix_pairs = pickle.load(f)
        with open("list_of_ranked_assignments.pkl", "rb") as f:
            list_of_ranked_assignments = pickle.load(f)

    # If time is scarcer than energy, use simple strategy
    if average_difficulty(community) < average_ability(community) * DIFFICULTY_ABILITY_RATIO:
        # print("Simple strategy...")
        for t in range(len(list_of_ranked_assignments)):
            individual_cost = cost_matrix_individual[(player.id, t)]
            for assignment in list_of_ranked_assignments[t]:
                if player in assignment[0] and None not in assignment[0]:
                    partner = assignment[0][0] if player == assignment[0][1] else assignment[0][1]
                    if player.energy >= assignment[1] and partner.energy >= assignment[1]:
                        if individual_cost + PAIRING_ADVANTAGE > assignment[1]:
                            list_choices.append([t, partner.id])
        return list_choices

    for t in range(len(list_of_ranked_assignments)):
        # Exhausting tasks (energy required >= 20)
        # Return and volunteer alone in Phase II
        if list_of_ranked_assignments[t][0][1] >= 20:
            if is_weakest_player(player, community):
                return list_choices

        best_decision = (None, np.inf)  # (partner_id, cost)

        for assignment in list_of_ranked_assignments[t]:
            if player in assignment[0]:
                # Tiring tasks (10 <= energy required < 20)
                if 10 < assignment[1] < 20:
                    # Wait until energy is full to volunteer with a partner
                    if None not in assignment[0] and player.energy == 10:
                        partner_id = assignment[0][0].id if player == assignment[0][1] else assignment[0][1].id
                        list_choices.append([t, partner_id])
                # Easier tasks (energy required < 10)
                else:
                    # Volunteer to partner as long as energy >= task cost
                    if player.energy >= assignment[1]:
                        if None not in assignment[0]:
                            # Paired assignment
                            partner_id = assignment[0][0].id if player == assignment[0][1] else assignment[0][1].id
                            if best_decision[0] is not None and assignment[1] < best_decision[1]:
                                best_decision = (partner_id, assignment[1])
                            elif best_decision[0] is None and assignment[1] + PAIRING_ADVANTAGE < best_decision[1]:
                                best_decision = (partner_id, assignment[1])
                        else:
                            # Individual assignment
                            if assignment[1] < best_decision[1] + PAIRING_ADVANTAGE:
                                best_decision = (None, assignment[1])
        
        if best_decision[0] is not None:
            list_choices.append([t, best_decision[0]])

    return list_choices

# @profile
def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []

    # Calculate cost matrices
    cost_matrix_individual, cost_matrix_pairs = calculate_cost_matrix(community)

    # Rank assignment options by task
    list_of_ranked_assignments = get_ranked_assignments(community, cost_matrix_individual,
                                                        cost_matrix_pairs)

    for t in range(len(list_of_ranked_assignments)):
        # Exhausting tasks (energy required >= 20)
        # Volunteer to sacrifice
        if list_of_ranked_assignments[t][0][1] >= 20:
            if is_weakest_player(player, community):
                bids.append(t)
                return bids

        # Easier tasks (energy required < 10)
        # Volunteer to partner as long as energy >= task cost
        elif list_of_ranked_assignments[t][0][1] <= 10:
            for assignment in list_of_ranked_assignments[t]:
                if player in assignment[0] and None in assignment[0] and player.energy >= assignment[1]:
                    bids.append(t)

    return bids

@profile
def calculate_cost_matrix(community):
    """
    Calculates the cost matrices for individual members and pairs based on abilities and tasks.
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

@profile
def get_ranked_assignments(community, cost_matrix_individual, cost_matrix_pairs):
    """
    Optimized ranking of assignments based on cost matrices.
    """
    tasks = community.tasks
    members = community.members

    list_of_ranked_assignments = []

    for task in range(len(tasks)):
        assignments = {}
        for member in members:
            assignments[(member, None)] = cost_matrix_individual[(member.id, task)]

        for member1, member2 in combinations(members, 2):
            assignments[(member1, member2)] = cost_matrix_pairs[(member1.id, member2.id, task)]

        min_heap = []
        for assignment, cost in assignments.items():
            heappush(min_heap, (cost, assignment))

        ranked_assignments = []
        while min_heap:
            cost, assignment = heappop(min_heap)
            ranked_assignments.append((assignment, cost))

        list_of_ranked_assignments.append(ranked_assignments)
    return list_of_ranked_assignments


# @profile
# def get_ranked_assignments(community, cost_matrix_individual, cost_matrix_pairs):
    # """
    # Rank assignments of paired and individual workers for each task based on the cost matrices.
    # """
    # tasks = community.tasks
    # members = community.members
    
    # list_of_ranked_assignments = []

    # for t in range(len(tasks)):
    #     assignments = {}
    #     for member in members:
    #         assignments[(member, None)] = cost_matrix_individual[(member.id, t)]
    #     for member1, member2 in combinations(members, 2):
    #         assignments[(member1, member2)] = cost_matrix_pairs[(member1.id, member2.id, t)]

    #     ranked_assignments_dict = dict(sorted(assignments.items(), key=lambda item: item[1]))
    #     ranked_assignments = list(ranked_assignments_dict.items())
    #     list_of_ranked_assignments.append(ranked_assignments)

    # return list_of_ranked_assignments


def is_weakest_player(player, community):
    sum_of_abilities = []
    
    for member in community.members:
        sum_of_abilities.append(sum(member.abilities))

    sorted_abilities = np.array(sorted(sum_of_abilities))
    threshold = np.percentile(sorted_abilities, WEAKNESS_THRESHOLD)
    
    return sum(player.abilities) < threshold


def average_difficulty(community):
    return sum([sum(task) for task in community.tasks]) / len(community.tasks)


def average_ability(community):
    return sum([sum(member.abilities) for member in community.members]) / len(community.members)