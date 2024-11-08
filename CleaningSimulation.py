import mesa
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CleaningAgent(mesa.Agent):
    """
    CleaningAgent is an agent that can clean dirty cells in a room. 
    It moves randomly to adjacent cells and cleans if it's dirty.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cleaned_cells = 0
        self.movements = 0

    def step(self):
        """
        Perform one action per time step.
        If the current cell is dirty, clean it.
        Otherwise, move randomly to an adjacent cell.
        """
        current_cell = self.pos
        if self.model.grid.is_cell_dirty(current_cell):
            self.model.grid.clean_cell(current_cell)
            self.cleaned_cells += 1
        else:
            self.move_randomly()

    def move_randomly(self):
        """
        Move to a random neighboring cell if it's not occupied.
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
    CleaningModel simulates a room cleaning process with multiple agents in a MxN grid.
    The agents move around cleaning dirty cells.
    """

    def __init__(self, width, height, num_agents, dirty_percentage, max_time):
        """
        Initialize the model with specified grid size, number of agents, 
        initial dirty cell percentage, and maximum simulation time.
        """
        super().__init__()
        self.grid = CleaningGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.max_time = max_time
        self.total_cells = width * height
        self.dirty_cells = int(self.total_cells * dirty_percentage)
        self.time_steps = 0

        # Initialize agents
        for i in range(num_agents):
            agent = CleaningAgent(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (1, 1))

        # Initialize dirty cells
        self.init_dirty_cells()

    def init_dirty_cells(self):
        """
        Randomly set initial dirty cells based on the specified dirty percentage.
        """
        available_positions = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        dirty_positions = random.sample(available_positions, self.dirty_cells)
        for pos in dirty_positions:
            self.grid.set_dirty(pos)

    def step(self):
        """
        Execute one time step in the simulation if time limit and dirty cells permit.
        """
        if self.time_steps < self.max_time and self.grid.count_dirty_cells() > 0:
            self.schedule.step()
            self.time_steps += 1

    def run_simulation(self):
        """
        Run the simulation, collecting cleaning percentages over time.
        """
        clean_percentages = []  # List to store the cleaning percentages
        while self.time_steps < self.max_time and self.grid.count_dirty_cells() > 0:
            self.step()
            clean_percentage = (
                (self.total_cells - self.grid.count_dirty_cells()) / self.total_cells * 100
            )
            clean_percentages.append(clean_percentage)  # Store the cleaning percentage

        total_movements = sum(agent.movements for agent in self.schedule.agents)
        return {
            "time_steps": self.time_steps,
            "clean_percentages": clean_percentages,  # List of percentages over time
            "total_movements": total_movements,
            "clean_percentage": clean_percentages[-1] if clean_percentages else 0,  # Final clean percentage
        }

    def run_simulation_with_visualization(self):
        """
        Run the simulation with visualization using matplotlib animation.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ims = []  # List to store the images for each step

        for _ in range(self.max_time):
            if self.grid.count_dirty_cells() > 0:
                self.schedule.step()
                self.time_steps += 1

                # Capture the position of the agents and dirt cells
                grid_array = self.get_grid()
                im = ax.imshow(grid_array, animated=True)
                ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        plt.show()

    def get_grid(self):
        """
        Generate a grid representation for visualization purposes.
        """
        grid_array = [[0 for _ in range(self.grid.height)] for _ in range(self.grid.width)]
        for pos in self.grid.dirty_cells:
            grid_array[pos[0]][pos[1]] = 1  # 1 for dirty cells
        for agent in self.schedule.agents:
            grid_array[agent.pos[0]][agent.pos[1]] = 2  # 2 for agents
        return grid_array


class CleaningGrid(mesa.space.MultiGrid):
    """
    CleaningGrid manages the state of each cell in the grid, tracking dirty and clean cells.
    """

    def __init__(self, width, height, torus):
        super().__init__(width, height, torus)
        self.dirty_cells = set()

    def set_dirty(self, pos):
        """
        Mark a cell as dirty.
        """
        self.dirty_cells.add(pos)

    def clean_cell(self, pos):
        """
        Clean a specified cell.
        """
        if pos in self.dirty_cells:
            self.dirty_cells.remove(pos)

    def is_cell_dirty(self, pos):
        """
        Check if a cell is dirty.
        """
        return pos in self.dirty_cells

    def count_dirty_cells(self):
        """
        Count the total number of dirty cells.
        """
        return len(self.dirty_cells)

    def is_cell_occupied(self, pos):
        """
        Check if a cell is occupied by an agent.
        """
        return bool(self.get_cell_list_contents([pos]))


# Example of running the simulation
width = 10
height = 10
num_agents = 5
dirty_percentage = 0.3
max_time = 200

model = CleaningModel(width, height, num_agents, dirty_percentage, max_time)
results = model.run_simulation()

# Visualization with seaborn
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
