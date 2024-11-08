import mesa
import random
import seaborn as sns
import matplotlib.pyplot as plt

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
        current_cell = self.pos
        if self.model.grid.is_cell_dirty(current_cell):
            self.model.grid.clean_cell(current_cell)
            self.cleaned_cells += 1
        else:
            self.move_randomly()

    def move_randomly(self):
        possible_moves = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_moves)
        if self.model.grid.is_cell_occupied(new_position) is False:
            self.model.grid.move_agent(self, new_position)
            self.movements += 1


class CleaningModel(mesa.Model):
    """
    CleaningModel simulates a room cleaning process with multiple agents in a MxN grid.
    The agents move around cleaning dirty cells.
    """

    def __init__(self, width, height, num_agents, dirty_percentage, max_time):
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
        available_positions = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        dirty_positions = random.sample(available_positions, self.dirty_cells)
        for pos in dirty_positions:
            self.grid.set_dirty(pos)

    def step(self):
        if self.time_steps < self.max_time and self.grid.count_dirty_cells() > 0:
            self.schedule.step()
            self.time_steps += 1

    def run_simulation(self):
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
            "clean_percentages": clean_percentages,  # Return the list of percentages
            "total_movements": total_movements,
            "clean_percentage": clean_percentages[-1] if clean_percentages else 0,  # Add the final clean percentage
        }

    def run_multiple_simulations(self, agent_counts, dirty_percentage, max_time):
        results = []
        for num_agents in agent_counts:
            model = CleaningModel(self.grid.width, self.grid.height, num_agents, dirty_percentage, max_time)
            simulation_results = model.run_simulation()
            results.append({
                "num_agents": num_agents,
                "time_steps": simulation_results["time_steps"],
                "total_movements": simulation_results["total_movements"],
                "clean_percentage": simulation_results["clean_percentage"],
            })
        return results


class CleaningGrid(mesa.space.MultiGrid):
    """
    CleaningGrid manages the state of each cell in the grid, tracking dirty and clean cells.
    """

    def __init__(self, width, height, torus):
        super().__init__(width, height, torus)
        self.dirty_cells = set()

    def set_dirty(self, pos):
        self.dirty_cells.add(pos)

    def clean_cell(self, pos):
        if pos in self.dirty_cells:
            self.dirty_cells.remove(pos)

    def is_cell_dirty(self, pos):
        return pos in self.dirty_cells

    def count_dirty_cells(self):
        return len(self.dirty_cells)

    def is_cell_occupied(self, pos):
        return self.get_cell_list_contents([pos]) != []


# Example of running the simulation
width = 10
height = 10
num_agents = 5
dirty_percentage = 0.3
max_time = 300

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

# Example of running multiple simulations
agent_counts = [5, 10, 15, 20, 25]  # Different agent counts
dirty_percentage = 0.3
max_time = 300

multiple_results = model.run_multiple_simulations(agent_counts, dirty_percentage, max_time)

for result in multiple_results:
    print(f"Agentes: {result['num_agents']}, Pasos de Tiempo: {result['time_steps']}, Movimientos Totales: {result['total_movements']}, Porcentaje Limpio: {result['clean_percentage']:.2f}%")
