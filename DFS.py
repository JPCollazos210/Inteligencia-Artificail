import pandas as pd
import numpy as np

# Definir la clase Node para representar cada posición en el laberinto
class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

# Clase StackFrontier para implementar DFS
class StackFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node

# Clase que representa el laberinto basado en la hoja de cálculo
class MazeFromExcel:
    def __init__(self, laberinto_matrix, start, goal):
        """
        Inicializa el laberinto con la matriz, el punto de inicio y el objetivo.
        """
        self.height = laberinto_matrix.shape[0]
        self.width = laberinto_matrix.shape[1]
        self.walls = laberinto_matrix
        self.start = start
        self.goal = goal
        self.solution = None

    def print(self):
        """
        Imprime el laberinto en la consola, mostrando el camino de la solución si existe.
        """
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        """
        Devuelve los vecinos válidos de un estado, evitando muros y bordes.
        """
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve(self):
        """
        Resuelve el laberinto utilizando DFS (StackFrontier) y devuelve la solución.
        """
        self.num_explored = 0

        # Inicializar la frontera con el nodo inicial
        start = Node(state=self.start, parent=None, action=None)
        frontier = StackFrontier()
        frontier.add(start)

        self.explored = set()

        # Loop hasta encontrar una solución
        while True:

            if frontier.empty():
                raise Exception("no solution")

            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)

            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)

# Cargar el archivo de Excel para revisar la estructura del laberinto
file_path = 'laberinto-vacio.xlsx'
laberinto_df = pd.read_excel(file_path, header=None)

# Convertir el DataFrame en una matriz de muros (True para muro, False para camino)
laberinto_matrix = laberinto_df.applymap(lambda x: True if x == "#" else False).to_numpy()

# Asignar las posiciones de inicio y meta (A y B) manualmente
start = (1, 1)  # Cambia estas coordenadas según el archivo
goal = (3, 30)  # Cambia estas coordenadas según el archivo

# Crear una instancia del laberinto a partir de la matriz extraída
maze = MazeFromExcel(laberinto_matrix, start, goal)

# Imprimir el laberinto original
print("Laberinto:")
maze.print()

# Resolver el laberinto
print("Resolviendo el laberinto...")
maze.solve()

# Imprimir el número de estados explorados y la solución
print(f"Estados explorados: {maze.num_explored}")
print("Solución:")
maze.print()
