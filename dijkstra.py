#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def dijkstra(start, finish, occupancy_grid):
    # initialize 'visited' grid to zero to keep track of whether we
    # have visited this node before
    visited = np.zeros([height, width])

    # initialize a distance grid to inf, this is where we keep track
    # of all the distances to every 'node' or position in the grid
    distance = np.zeros([height, width])
    distance[:, :] = np.inf

    # set the position of the start node to zero
    distance[start] = 0

    # start by visiting neighbors of the first node, add the neighbors
    # of the start node to queue 'q'
    q = visit_neighbors(
        start, visited, distance, occupancy_grid
    )

    # start processing queue 'q'. Each time we pop off the position
    # in the queue with the shortest distance recorded
    # We visit the neighbors of the position and add them to the end
    # of the queue
    while q:

        # list all the distances for each of the positions in the queue
        distances = [distance[position] for position in q]

        # find the index of the shortest distance
        idx = distances.index(min(distances))

        # remove the position at index 'idx' from the queue
        position = q.pop(idx)

        # if the neighbor is the finish, we're done and we can exit early
        if position == finish:
            break

        # otherwise keep on processing the neighbors of node 'position'
        q = (
            q
            + visit_neighbors(
                position, visited, distance, occupancy_grid
            )
        )

    # get the path by starting at the finish node and working backwards until
    # we get to the start
    path = []
    get_path(finish, distance, path)

    # after the search, the path list will be in the reverse order.
    # reverse it so it goes from the start to the finish location
    path.reverse()
    return path, distance

def visit_neighbors(position, visited, distance, occupancy_grid):
    tmp = []

    # only compute distances to neighbors of node 'position' if
    # we haven't already visited it before.
    if visited[position] == 0:
        # mark it as visited for the next time, so this node
        # won't ever be processed again
        visited[position] = 1

        # get the minimum distance calculated to node 'position'
        current_distance = distance[position]

        # The eight neighbors of position are show below:
        #
        #   (-1, +1)     (0, +1)       (+1, +1)
        #             \     |     /
        #   (-1, 0) -   position    -  (+1, 0)
        #             /     |     \
        #   (-1, -1)     (0, -1)       (+1, -1)
        #
        neighbors = [
            (position[0] - 1, position[1] + 1),
            (position[0] + 0, position[1] + 1),
            (position[0] + 1, position[1] + 1),
            (position[0] + 1, position[1] + 0),
            (position[0] + 1, position[1] - 1),
            (position[0] + 0, position[1] - 1),
            (position[0] - 1, position[1] - 1),
            (position[0] - 1, position[1] + 0)
        ]

        # The distances to the neighbors can be found using
        # pythagorean theorem: distance = sqrt(x^2 + y^2)
        # Stores the distances in a list equal to the length
        # of the 'neighbors' list. We could compute the distances
        # using the following code, or just compute them ahead of time
        # and save them
        # distances = [
        #     np.sqrt(
        #         (neighbor[0] - position[0])**2
        #         + (neighbor[1] - position[1])**2
        #     ) for neighbor in neighbors
        # ]
        sqrt_2 = np.sqrt(2)
        distances = [sqrt_2, 1, sqrt_2, 1, sqrt_2, 1, sqrt_2, 1]

        # Check all the neighbors and update the distance only if the
        # new computed distance is LESS than the distance already recorded
        # at that location
        for d, neighbor in zip(distances, neighbors):
            i = neighbor[0]
            j = neighbor[1]
            if (i > -1 and i < visited.shape[0] and
                j > -1 and j < visited.shape[1] and
                occupancy_grid[i, j] == 0):
                distance[i, j] = np.fmin(current_distance + d, distance[i, j])
                tmp.append(neighbor)
    return tmp


def get_path(pos, distance, path):
    # to get the path, we work backwards starting from the 'finish' node
    # check all the neighbors of the finish node
    path.append(pos)
    neighbors = [
        (pos[0] - 1, pos[1] + 1),
        (pos[0] + 0, pos[1] + 1),
        (pos[0] + 1, pos[1] + 1),
        (pos[0] + 1, pos[1] + 0),
        (pos[0] + 1, pos[1] - 1),
        (pos[0] + 0, pos[1] - 1),
        (pos[0] - 1, pos[1] - 1),
        (pos[0] - 1, pos[1] + 0)
    ]

    neighbor_distances = [np.inf] * 8
    for k, neighbor in enumerate(neighbors):
        i = neighbor[0]
        j = neighbor[1]
        if (i > -1 and i < distance.shape[0] and
            j > -1 and j < distance.shape[1]):
            neighbor_distances[k] = distance[neighbor]

    # find the minimum distance and the position of the neighbor with the
    # minimum distance
    min_neighbor_distance = min(neighbor_distances)
    min_neighbor_position = neighbors[
        neighbor_distances.index(min_neighbor_distance)
    ]

    # append the new minimum neighbor position to the path list
    path.append(min_neighbor_position)

    # if the minimum distance we found is zero, we are back to the
    # start of the path and we are done.
    if min_neighbor_distance > 0:
        # otherwise continue searching the neighbors of the minimum
        # distance neighbor by calling this function again
        # When a function calls itself, it's considered to be 'recursive'
        get_path(min_neighbor_position, distance, path)


if __name__ == "__main__":
    height = 6
    width = 12
    occupancy_grid = np.zeros([height, width])
    occupancy_grid[2, 7] = 1
    occupancy_grid[3, 8] = 1
    occupancy_grid[2, 8] = 1
    occupancy_grid[3, 7] = 1
    occupancy_grid[2, 9] = 1
    occupancy_grid[1, 2] = 1
    occupancy_grid[2, 2] = 1
    occupancy_grid[2, 2] = 1
    occupancy_grid[2, 4] = 1

    start = (0, 0)
    finish = (5, width - 1)

    path, distance = dijkstra(start, finish, occupancy_grid)

    x, y = np.meshgrid(range(width), range(height))
    plt.pcolor(x, y, occupancy_grid)

    plt.scatter(start[1], start[0], marker='o')
    plt.scatter(finish[1], finish[0], marker='x')
    plt.plot([p[1] for p in path], [p[0] for p in path])
    plt.show()
