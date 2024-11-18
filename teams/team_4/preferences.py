import logging
from itertools import combinations
from pulp import *


logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log-results/team4_log.log", mode='w'),
        logging.StreamHandler()
    ]
)


def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''
    list_choices = []

    num_members = len(community.members)

    personal_threshold = 0 # vary this in later stages

    for i, task in enumerate(community.tasks):
        for partner_id in range(num_members):
            if partner_id == player.id:
                continue
            partner = community.members[partner_id]
            energy_cost = sum([max(task[j] - max(player.abilities[j], partner.abilities[j]), 0) for j in range(len(player.abilities))])

            if (energy_cost / 2) <= player.energy - personal_threshold:
                list_choices.append([i, partner_id])

    # Evaluate need for sacrifice
    # [TODO]

    # Mixed-integer linear program
    # [TODO]

    return list_choices


def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    # logging.debug(f"Player energy: {player.energy}")
    bids = []
    pain = 2
    if player.energy < pain:
        return bids
    
    num_abilities = len(player.abilities)

    for i, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
        # logging.debug(f"Energy cost: {energy_cost}")
        if energy_cost <= pain:
            bids.append(i)
            # logging.debug(f"Task {i} is being bid on")
            # logging.debug(f"Player energy: {player.energy}")
            # logging.debug(f"Energy cost: " + str(energy_cost))

    # Execute sacrifice
    # [TODO]

    # MILP
    # [TODO]

    return bids


def optimize_task_assignments(community):
    """
    Optimizes task assignments for the given community using MILP.
    Args:
        community: An instance of the Community class.
    Returns:
        A list of assignments in the form [(member_id(s), task)].
    """
    # Get cost matrices
    cost_matrix_individual, cost_matrix_pairs = calculate_cost_matrix(community)

    # Create problem
    prob = LpProblem("Community_Task_Assignment", LpMinimize)

    # Decision variables
    tasks = community.tasks
    members = community.members
    pairs = list(combinations(members, 2))

    x_individual = LpVariable.dicts(
        "x_individual", [(m.id, t) for m in members for t in range(len(tasks))], 0, 1, LpBinary
    )
    x_pairs = LpVariable.dicts(
        "x_pairs",
        [(m1.id, m2.id, t) for m1, m2 in pairs for t in range(len(tasks))], 0, 1, LpBinary
    )

    # Minimize objective function:
    # 1) total energy expenditures in this round
    # 2) less optimal cost + 1 per assigned task to prevent empty assignments
    # 3) less remaining tasks as percentage of workers
    optimal_costs = lowest_cost_per_task(cost_matrix_individual, cost_matrix_pairs, tasks)
    assigned_tasks = get_assigned_tasks(x_individual, x_pairs)

    prob += (
        lpSum(
            cost_matrix_individual[m, t] * x_individual[m, t]
            for m, t in cost_matrix_individual
        )
        + lpSum(
            cost_matrix_pairs[m1, m2, t] * x_pairs[m1, m2, t]
            for m1, m2, t in cost_matrix_pairs
        )
        - lpSum(
            (optimal_costs[t] + 1) * x_individual[m, t]
            for m, t in x_individual
        )
        - lpSum(
            (optimal_costs[t] + 1) * x_pairs[m1, m2, t]
            for m1, m2, t in x_pairs
        )
        + lpSum(
            max(len(members) - (len(tasks) - len(assigned_tasks)), 0)
        )
    )

    # Constraint 1: Each task can be assigned to no more than one individual or one pair
    for t in range(len(tasks)):
        prob += (
            lpSum(x_individual[m.id, t] for m in members)
            + lpSum(x_pairs[m1.id, m2.id, t] for m1, m2 in pairs)
            <= 1, f'NoDuplicateTaskConstraint_Task_{t}'
        )

    # Constraint 2: Each member is assigned to at most one task (solo or pair)
    for m in members:
        prob += (
            lpSum(x_individual[m.id, t] for t in range(len(tasks)))
            + lpSum(
                x_pairs[m.id, p.id, t] for t in range(len(tasks)) for p in members if m.id < p.id
            )
            + lpSum(
                x_pairs[p.id, m.id, t] for t in range(len(tasks)) for p in members if p.id < m.id
            )
            <= 1, f'NoDuplicatePlayerConstraint_Player_{m.id}'
        )

    # Constraint 3: No one should go below 0 energy
    for m in members:
        prob += (
            m.energy
            - lpSum(
                cost_matrix_individual[m.id, t] * x_individual[m.id, t]
                for t in range(len(tasks))
            )
            - lpSum(
                cost_matrix_pairs[m.id, p.id, t] * x_pairs[m.id, p.id, t]
                for t in range(len(tasks)) for p in members if m.id < p.id
            )
            - lpSum(
                cost_matrix_pairs[p.id, m.id, t] * x_pairs[p.id, m.id, t]
                for t in range(len(tasks)) for p in members if p.id < m.id
            )
            >= 0, f'EnergyConstraint_Player_{m.id}'
        )

    # Solve the problem
    prob.solve()

    # Extract results
    assignments = []
    for t in tasks:
        for m in members:
            if x_individual[m.id, t].varValue == 1:
                assignments.append(([m.id], t))
        for m1, m2 in pairs:
            if x_pairs[m1.id, m2.id, t].varValue == 1:
                assignments.append(([m1.id, m2.id], t))

    return assignments


def calculate_cost_matrix(community):
    """
    Calculates the cost matrix for individual members and pairs based on their ability and tasks.
    Args:
        community: An instance of the Community class.
    Returns:
        A tuple containing:
            - cost_matrix_individual: Dictionary of costs for individuals.
            - cost_matrix_pairs: Dictionary of costs for pairs.
    """
    tasks = community.tasks
    members = community.members

    # Individual costs
    cost_matrix_individual = {}
    for member in members:
        for task in tasks:
            cost = max(0, task - member.ability)  # Energy cost based on ability
            cost_matrix_individual[(member.id, task)] = cost

    # Pair costs
    cost_matrix_pairs = {}
    for member1, member2 in combinations(members, 2):  # All possible pairs
        for task in tasks:
            combined_ability = max(member1.ability, member2.ability)
            cost = max(0, task - combined_ability) / 2  # Halve cost for shared work
            cost_matrix_pairs[(member1.id, member2.id, task)] = cost

    return cost_matrix_individual, cost_matrix_pairs


def lowest_cost_per_task(cost_matrix_individual, cost_matrix_pairs, tasks):
    """
    Returns the lowest cost for each task based on individual and pair costs.
    """
    lowest_costs = {}

    for t in range(len(tasks)):
        individual_costs = [
            cost_matrix_individual[m, t] for m, _ in cost_matrix_individual if _ == t
        ]
        pair_costs = [
            cost_matrix_pairs[m1, m2, t]
            for m1, m2, _ in cost_matrix_pairs
            if _ == t
        ]
        lowest_costs[t] = min(individual_costs + pair_costs)

    return lowest_costs


def get_assigned_tasks(x_individual, x_pairs):
    """
    Returns a set of assigned tasks based on the decision variables.
    """
    assigned_tasks = {}
    for m, t in x_individual:
        if x_individual[m, t].varValue == 1:
            assigned_tasks.add(t)
    for m1, m2, t in x_pairs:
        if x_pairs[m1, m2, t].varValue == 1:
            assigned_tasks.add(t)

    return assigned_tasks


# TODO -- sacrifice strategy
