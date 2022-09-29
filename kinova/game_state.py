import networkx as nx

class GameState():
    def __init__(self):
        self.reset()

    def get_next_action(self, state=None):
        if state is None:
            state = self.state
        if state == 0:
            # the interaction hasn't started, so move to the goal location
            return "navigate"
        elif state == 1:
            # already navigated to the goal, so move to grasp it
            return "lower"
        elif state == 2:
            # already lowered to the goal, so grasp object
            return "grasp"
        elif state == 3:
            # currently grasping object, want to let go of it
            return "open"

    def is_legal(self, state, action):
        actions = [e[2]["action"] for e in list(self.G.edges(state, data=True))]
        return action in actions

    def complete_action(self, action, set_state=True):
        # check that this action is legal
        if not self.is_legal(self.state, action):
            return
        edges = list(self.G.edges(self.state, data=True))
        dest = max(edges, key=lambda e: e[2]["action"] == action)[1]
        
        if set_state:
            self.state = dest
        return dest

    def reset(self):
        # create state graph
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3])
        # 0: nothing grasped
        # 1: above object
        # 2: gripper around object
        # 3: grasping object

        # create connections between states via actions
        G.add_edge(0, 1, action="navigate")
        
        G.add_edge(1, 2, action="lower")
        G.add_edge(1, 0, action="navigate")

        G.add_edge(2, 3, action="grasp")
        G.add_edge(2, 0, action="navigate")

        G.add_edge(3, 0, action="open")

        self.G = G
        self.state = 0


if __name__ == "__main__":
    game = GameState()
    action = game.get_next_action()
    game.complete_action(action)
