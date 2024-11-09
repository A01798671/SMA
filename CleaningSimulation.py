"""
Programa que simula la limpieza de una habitación utilizando agentes móviles.
Autores: 
Melissa Mireles Rendón A01379736
Alberto Cebreros González A01798671
Fecha de creación/modificación: 08 de Noviembre del 2024
"""

import mesa
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CleaningAgent(mesa.Agent):
    """
    CleaningAgent es un agente que puede limpiar celdas sucias en una habitación.
    Se mueve aleatoriamente a celdas adyacentes y limpia si la celda está sucia.
    """

    def __init__(self, unique_id, model):
        """
        Inicializa el agente de limpieza.

        Parámetros:
        - uniqueId: Identificador único del agente.
        - model: Referencia al modelo al que pertenece el agente.
        """
        super().__init__(unique_id, model)
        self.cleaned_cells = 0
        self.movements = 0

    def step(self):
        """
        Realiza una acción por cada paso de tiempo.
        Si la celda actual está sucia, la limpia.
        De lo contrario, se mueve aleatoriamente a una celda adyacente.
        """
        current_cell = self.pos
        if self.model.grid.is_cell_dirty(current_cell):
            self.model.grid.clean_cell(current_cell)
            self.cleaned_cells += 1
        else:
            self.move_randomly()

    def move_randomly(self):
        """
        Se mueve a una celda vecina aleatoria si no está ocupada.
        """
        possible_moves = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_moves)
        if not self.model.grid.is_cell_occupied(new_position):
            self.model.grid.move_agent(self, new_position)
            self.movements += 1


class CleaningModel(mesa.Model):
    """
    CleaningModel simula el proceso de limpieza de una habitación con múltiples agentes en una grilla MxN.
    Los agentes se mueven limpiando celdas sucias.
    """

    def __init__(self, width, height, num_agents, dirty_percentage, max_time):
        """
        Inicializa el modelo con los parámetros especificados.

        Parámetros:
        - width: Ancho de la grilla.
        - height: Altura de la grilla.
        - numAgents: Número de agentes de limpieza.
        - dirtyPercentage: Porcentaje inicial de celdas sucias (valor entre 0 y 1).
        - maxTime: Tiempo máximo de simulación.
        """
        super().__init__()
        self.grid = CleaningGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.max_time = max_time
        self.total_cells = width * height
        self.dirty_cells = int(self.total_cells * dirty_percentage)
        self.time_steps = 0

        # Inicializar agentes
        for i in range(num_agents):
            agent = CleaningAgent(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (1, 1))

        # Inicializar celdas sucias
        self.init_dirty_cells()

    def init_dirty_cells(self):
        """
        Establece aleatoriamente las celdas sucias iniciales basadas en el porcentaje especificado.
        """
        available_positions = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        dirty_positions = random.sample(available_positions, self.dirty_cells)
        for pos in dirty_positions:
            self.grid.set_dirty(pos)

    def step(self):
        """
        Ejecuta un paso de tiempo en la simulación si el límite de tiempo y las celdas sucias lo permiten.
        """
        if self.time_steps < self.max_time and self.grid.count_dirty_cells() > 0:
            self.schedule.step()
            self.time_steps += 1

    def run_simulation(self):
        """
        Ejecuta la simulación, recolectando los porcentajes de limpieza a lo largo del tiempo.

        Retorna:
        - Un diccionario con los resultados:
            - timeSteps: Número total de pasos de tiempo ejecutados.
            - cleanPercentages: Lista de porcentajes de limpieza en cada paso.
            - totalMovements: Número total de movimientos realizados por los agentes.
            - cleanPercentage: Porcentaje final de limpieza alcanzado.
        """
        clean_percentages = []  # Lista para almacenar los porcentajes de limpieza
        while self.time_steps < self.max_time and self.grid.count_dirty_cells() > 0:
            self.step()
            clean_percentage = (
                (self.total_cells - self.grid.count_dirty_cells()) / self.total_cells * 100
            )
            clean_percentages.append(clean_percentage)  # Almacena el porcentaje de limpieza

        total_movements = sum(agent.movements for agent in self.schedule.agents)
        return {
            "time_steps": self.time_steps,
            "clean_percentages": clean_percentages,  # Lista de porcentajes a lo largo del tiempo
            "total_movements": total_movements,
            "clean_percentage": clean_percentages[-1] if clean_percentages else 0,  # Porcentaje final de limpieza
        }

    def run_simulation_with_visualization(self):
        """
        Ejecuta la simulación con visualización utilizando animación de matplotlib.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ims = []  # Lista para almacenar las imágenes de cada paso

        for _ in range(self.max_time):
            if self.grid.count_dirty_cells() > 0:
                self.schedule.step()
                self.time_steps += 1

                # Capturar la posición de los agentes y las celdas sucias
                grid_array = self.get_grid()
                im = ax.imshow(grid_array, animated=True)
                ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        plt.show()

    def get_grid(self):
        """
        Genera una representación de la grilla para visualización.

        Retorna:
        - gridArray: Matriz representando el estado de la grilla.
            - 0: Celda limpia.
            - 1: Celda sucia.
            - 2: Agente de limpieza.
        """
        grid_array = [[0 for _ in range(self.grid.height)] for _ in range(self.grid.width)]
        for pos in self.grid.dirty_cells:
            grid_array[pos[0]][pos[1]] = 1  # 1 para celdas sucias
        for agent in self.schedule.agents:
            grid_array[agent.pos[0]][agent.pos[1]] = 2  # 2 para agentes
        return grid_array


class CleaningGrid(mesa.space.MultiGrid):
    """
    CleaningGrid gestiona el estado de cada celda en la grilla, rastreando celdas sucias y limpias.
    """

    def __init__(self, width, height, torus):
        """
        Inicializa la grilla de limpieza.

        Parámetros:
        - width: Ancho de la grilla.
        - height: Altura de la grilla.
        - torus: Si la grilla es un toro (los bordes están conectados).
        """
        super().__init__(width, height, torus)
        self.dirty_cells = set()

    def set_dirty(self, pos):
        """
        Marca una celda como sucia.

        Parámetros:
        - pos: Tupla con las coordenadas de la celda.
        """
        self.dirty_cells.add(pos)

    def clean_cell(self, pos):
        """
        Limpia una celda específica.

        Parámetros:
        - pos: Tupla con las coordenadas de la celda.
        """
        if pos in self.dirty_cells:
            self.dirty_cells.remove(pos)

    def is_cell_dirty(self, pos):
        """
        Verifica si una celda está sucia.

        Parámetros:
        - pos: Tupla con las coordenadas de la celda.

        Retorna:
        - True si la celda está sucia, False en caso contrario.
        """
        return pos in self.dirty_cells

    def count_dirty_cells(self):
        """
        Cuenta el número total de celdas sucias.

        Retorna:
        - Número de celdas sucias.
        """
        return len(self.dirty_cells)

    def is_cell_occupied(self, pos):
        """
        Verifica si una celda está ocupada por un agente.

        Parámetros:
        - pos: Tupla con las coordenadas de la celda.

        Retorna:
        - True si la celda está ocupada, False en caso contrario.
        """
        return bool(self.get_cell_list_contents([pos]))


# Ejemplo de ejecución de la simulación
width = 10
height = 10
num_agents = 5
dirty_percentage = 0.3 # Porcentaje de celdas sucias (valor entre 0 y 1)
maxTime = 200
max_time = 200

model = CleaningModel(width, height, num_agents, dirty_percentage, max_time)
results = model.run_simulation()

# Visualización con seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(results['clean_percentages'])), y=results['clean_percentages'])
plt.title("Porcentaje de Limpieza a lo Largo del Tiempo")
plt.xlabel("Pasos de Tiempo")
plt.ylabel("Porcentaje de Limpieza")
plt.grid()
plt.show()

print("Simulation Results:")
print(f"Time Steps: {results['time_steps']}")
print(f"Total Movements: {results['total_movements']}")
print(f"Clean Percentage: {results['clean_percentage']:.2f}%")

# Ejemplo de ejecución de la simulación con visualización
model = CleaningModel(width, height, num_agents, dirty_percentage, max_time)
model.run_simulation_with_visualization()
