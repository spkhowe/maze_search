def update_decisions():
    tuple_decisions = { 
                'P1': {
                        'Maze1': {'path': [(6, 7), (5, 7), (0, 10), (2, 5), (4, 2), (1, 0)]},
                        'Maze2': {'path': [(2, 12), (5, 11)]},
                        'Maze3': {'path': [(0, 11), (1, 3)]},
                        'Maze4': {'path': [(6, 0), (2, 1), (1, 11), (4, 9)]},
                        'Maze5': {'path': [(0, 1), (5, 2), (5, 3), (4, 5)]},
                        'Maze7': {'path': [(0, 6), (1, 10), (4, 11), (5, 8)]},
                        'Maze7': {'path': [(6, 0), (2, 6)]},
                        'Maze8': {'path': [(6, 12), (4, 8), (3, 10), (0, 11)]},
                }
            }

    for maze in tuple_decisions['P1']:
        for elem in tuple_decisions['P1'][maze]['path']:
            elem = (elem[0]+1,elem[1]+1)

    return tuple_decisions

td = update_decisions()
print(td)