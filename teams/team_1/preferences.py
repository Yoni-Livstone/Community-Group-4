global strong_players 
global remaining

def phaseIpreferences(player, community, global_random):
    '''Return a list of task index and the partner id for the particular player. The output format should be a list of lists such that each element
    in the list has the first index task [index in the community.tasks list] and the second index as the partner id'''

    strong_players = set()
    all_players = set(community.members)
    for p in community.members:
        for t in community.tasks:
            # check if the player has greater ability than the task in all dimensions
            if all(p.abilities[i] >= t[i] for i in range(len(t))):
                strong_players.add(p)

    # calculate the difference between all players and good players
    partnerships = all_players - strong_players

    # check possible partnerships and only assign the partnerships to bids if they can do tasks without losing energy
    # if they cannot, add them to remaining players (remaining = all players - good players - partnerships)

    list_choices = []
    if player.energy < 0:
        return list_choices
    num_members = len(community.members)
    partner_id = num_members - player.id - 1
    list_choices.append([0, partner_id])
    if len(community.tasks) > 1:
        list_choices.append([1, partner_id])
    list_choices.append([0, min(partner_id + 1, num_members - 1)])
    return list_choices

def phaseIIpreferences(player, community, global_random):
    '''Return a list of tasks for the particular player to do individually'''
    bids = []
    if player.energy < 0:
        return bids
    num_abilities = len(player.abilities)
    for i, task in enumerate(community.tasks):
        energy_cost = sum([max(task[j] - player.abilities[j], 0) for j in range(num_abilities)])
        if energy_cost >= 10:
            continue
        bids.append(i)
    return bids
