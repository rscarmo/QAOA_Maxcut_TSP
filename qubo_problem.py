from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_algorithms import QAOA, SamplingVQE

# from qiskit.primitives import Sampler
from qiskit.primitives import BackendSampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKolkataV2
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer
from qiskit_aer import AerSimulator
from qiskit_optimization.problems.variable import VarType
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZGate, RXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile
from itertools import product
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.circuit.library import PhaseGate
from qiskit_algorithms import VQE
from qiskit.circuit.library import StatePreparation
from qiskit.synthesis.evolution import LieTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import QasmSimulator
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector


import numpy as np
import time
import networkx as nx

class QAOA_TSP_Maxcut:
    def __init__(
        self,
        G,
        config,
        seed=42,
        TSP=True,
        fake_backend = False
    ):
        """
        Initialize the QUBO for the Degree-Constrained Minimum Spanning Tree problem.

        Parameters:
        - G: A networkx.Graph object representing the input graph (weighted).
        - degree_constraints: A dictionary {v: max_degree} for each vertex.
        - root: The root vertex (v0) of the spanning tree.
        """
        self.qb = QuadraticProgram()
        self.G = G
        self.n = G.number_of_nodes()
        self.root = 0
        self.config = config
        self.seed = seed
        self.TSP = TSP
        if not self.TSP:
            self.Maxcut = True
        else:
            self.Maxcut = False
        self.fake_backend = fake_backend
        self.num_qubits = 0
        self.var_names = None

        self.objective_func_vals = []

        self._configure_variables()
        self._define_objective_function()
        if self.TSP:
            self._add_constraints()

    def _configure_variables(self):
        if self.Maxcut:
            for i in self.G.nodes:
                self.qb.binary_var(f'x{i}')  # Define binary variables
        elif self.TSP:
            self.num_cities = len(self.G.nodes)
            for i in range(1, self.num_cities):
                for p in range(1, self.num_cities):
                    self.qb.binary_var(f'x_{i}_{p}')

        self.var_names = self.qb.variables

    def _define_objective_function(self):
        linear_terms = {}
        quadratic_terms = {}

        if self.Maxcut:
            # Para cada aresta (i, j) com peso w:
            for i, j, w in self.G.edges(data='weight'):
                # Atualiza os termos lineares para x_i e x_j (soma w para cada ocorrência)
                linear_terms[f'x{i}'] = linear_terms.get(f'x{i}', 0) + w
                linear_terms[f'x{j}'] = linear_terms.get(f'x{j}', 0) + w

                # Atualiza o termo quadrático para a interação x_i * x_j: -2w
                key = (f'x{i}', f'x{j}')
                quadratic_terms[key] = quadratic_terms.get(key, 0) - 2 * w

        elif self.TSP:
            for i in range(1, self.num_cities):
                for j in range(1, self.num_cities):
                    if i != j :
                        for p in range(1, self.num_cities - 1):
                            quadratic_terms[(f'x_{i}_{p}', f'x_{j}_{p+1}')] = self.G[i][j]['weight']
            # Contribution of the fixed starting city (index 0)
            for i in range(1, self.num_cities):
                linear_terms[f'x_{i}_1'] = self.G[0][i]['weight']
                linear_terms[f'x_{i}_{self.num_cities - 1}'] = self.G[0][i]['weight']
                
                        

        # 3. Set the final objective function in the QUBO
        self.qb.minimize(linear=linear_terms, quadratic=quadratic_terms)

        if self.Maxcut:
            print(self.qb.prettyprint())

    def _add_constraints(self):
        if self.TSP:
            # Each city is visited exactly once
            for i in range(1, self.num_cities):
                coeffs = {f"x_{i}_{p}": 1 for p in range(1, self.num_cities)}
                # Create a linear constraint with sense='==', rhs=1
                self.qb.linear_constraint(
                    linear=coeffs,
                    sense='==',
                    rhs=1,
                    name=f"constraint_{i}_p"
                )
    
            # Each position in the tour is occupied by exactly one city
            for p in range(1, self.num_cities):
                # Build a dict of variable names -> coefficient 1
                coeffs = {f"x_{i}_{p}": 1 for i in range(1, self.num_cities)}
                # Create a linear constraint with sense='==', rhs=1
                self.qb.linear_constraint(
                    linear=coeffs,
                    sense='==',
                    rhs=1,
                    name=f"constraint_i_{p}"
                )
            print(self.qb.prettyprint())
    
    def configure_backend(self):
        if self.config.SIMULATION == "True":
            if not self.fake_backend:
                print("Proceeding with simulation...")
                # backend = AerSimulator(method="statevector", device="GPU")
                backend = AerSimulator(method="statevector")
            else:
                print("Proceeding with simulation in Fake IBM_Brisbane using AerSimulator...")
                service = QiskitRuntimeService(
                    channel="ibm_quantum", token=self.config.QXToken
                )                
                real_backend = service.backend("ibm_brisbane")
                
                # backend = AerSimulator.from_backend(real_backend, device='GPU')
                backend = AerSimulator.from_backend(real_backend)
            # backend = QasmSimulator()
            backend.set_options(seed_simulator=self.seed)
        else:
            print("Proceeding with IBM Quantum hardware...")
            service = QiskitRuntimeService(
                channel="ibm_quantum", token=self.config.QXToken
            )
            # backend = service.least_busy(min_num_qubits=127, operational=True, simulator=False)
            backend = service.backend("ibm_brisbane")
            print(f"Connected to {backend.name}!")
        return backend


    def cost_func_estimator(self, params, ansatz, hamiltonian, estimator, offset=0.0):

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        self.objective_func_vals.append(cost)

        if not self.Maxcut:
            return cost + offset
        else:
            return - (cost + offset)

    
    def evaluate_bitstring_cost(self, bitstring: str) -> float:
        """
        Given a candidate bitstring (e.g. '010101'),
        returns the objective value of that bitstring for this QUBO.
        
        Parameters:
            bitstring (str): A string of '0'/'1' characters whose length 
                            must match the total number of variables in self.qubo.

        Returns:
            float: The cost (objective value) evaluated at the bitstring.
        """
        # Convert each character in the bitstring to float 0.0 or 1.0
        x = [float(b) for b in bitstring]

        # Safety check: ensure bitstring length matches the number of QUBO variables
        if len(x) != len(self.qubo.variables):
            raise ValueError(
                f"Bitstring length ({str(x)}) does not match the expected number "
                f"of variables ({len(self.qubo.variables)}) in the QuadraticProgram."
            )

        # Evaluate the (unconstrained) objective function for this bitstring
        converter = QuadraticProgramToQubo()
        qubo_with_penalties = converter.convert(self.qb)

        cost = qubo_with_penalties.objective.evaluate(x)
        return cost

    def time_execution_feasibility(self, backend):
        # Retrieve backend properties
        properties = backend.properties()

        # Extract gate durations
        gate_durations = {}
        for gate in properties.gates:
            gate_name = gate.gate
            if gate.parameters:
                duration = gate.parameters[0].value  # Duration in seconds
                gate_durations[gate_name] = duration

        print("Gate durations (in seconds):")
        for gate, duration in gate_durations.items():
            print(f"{gate}: {duration * 1e9:.2f} ns")

        # Calculate total execution time
        total_time = 0
        for instruction, qargs, cargs in self.qaoa_circuit.data:
            gate_name = instruction.name
            gate_time = gate_durations.get(gate_name, 0)
            total_time += gate_time

        print(f"Total circuit execution time: {total_time * 1e6:.2f} µs")

        # Extract coherence times with qubit indices
        coherence_times = {}
        for qubit_index, qubit in enumerate(properties.qubits):
            T1 = None
            T2 = None
            for param in qubit:
                if param.name == 'T1':
                    T1 = param.value
                elif param.name == 'T2':
                    T2 = param.value
            coherence_times[qubit_index] = {'T1': T1, 'T2': T2}
            print(f"Qubit {qubit_index}: T1 = {T1*1e6:.2f} µs, T2 = {T2*1e6:.2f} µs")

        # Access the layout to map virtual qubits to physical qubits
        transpile_layout = self.qaoa_circuit._layout  # Note the underscore before 'layout'

        layout = transpile_layout.final_layout
        
        # Retrieve the virtual-to-physical qubit mapping
        virtual_to_physical = layout.get_virtual_bits()

        # Determine which physical qubits are used in the circuit
        used_physical_qubits = set(virtual_to_physical.values())

        # Now, get the minimum T1 and T2 among the used physical qubits
        min_T1 = min(coherence_times[q_index]['T1'] for q_index in used_physical_qubits)
        min_T2 = min(coherence_times[q_index]['T2'] for q_index in used_physical_qubits)

        # Compare execution time to thresholds
        threshold_T1 = 0.1 * min_T1
        threshold_T2 = 0.1 * min_T2

        print(f"Thresholds: 10% T1 = {threshold_T1*1e6:.2f} µs, 10% T2 = {threshold_T2*1e6:.2f} µs")
        print(f"Circuit execution time: {total_time*1e6:.2f} µs")

        if total_time < threshold_T1 and total_time < threshold_T2:
            print("Execution time is within acceptable limits.")
        else:
            print("Execution time may be too long; consider optimizing your circuit.")    
   

    def solve_problem(self, p=1, parameters=None):
        # Convert the problem with constraints into an unconstrained QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(self.qb)

        # Print the Ising model
        print(qubo.to_ising())

        qubo_ops, offset = qubo.to_ising()

        backend = self.configure_backend()

        if self.fake_backend or self.config.SIMULATION == False:
            self.time_execution_feasibility(backend)

        estimator = Estimator(backend)
        estimator.options.default_shots = 1000

        if self.config.SIMULATION == False:
            # Set simple error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"          

        # Define a callback function to track progress
        def callback(params):
            print(f"Current parameters: {params}")

        # Set up QAOA with the callback
        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process
        algorithm_globals.random_seed = self.seed

        # Generate initial parameters using the seed
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)       

        qaoa_mes = QAOAAnsatz(
            cost_operator=qubo_ops,
            reps=p          
        )
        qaoa_mes.measure_all()

        # Create a custom pass manager
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

        print(qaoa_mes.draw(output='mpl'))        

        # Transpile the circuit
        self.qaoa_circuit = pm.run(qaoa_mes)     

        print(self.qaoa_circuit.draw(output='mpl'))
            
        start_time = time.time()
        qaoa_result = minimize(
            self.cost_func_estimator,
            initial_params,
            args=(self.qaoa_circuit, qubo_ops, estimator, offset),
            method="COBYLA",
            tol=1e-2,
            callback=callback,
        )
        end_time = time.time()
        print(qaoa_result)

        self.execution_time = end_time - start_time
        # self.solution = qaoa_result.variables_dict
        print(f"Execution time: {self.execution_time} seconds")         
        
        if self.Maxcut:
            print("Best params:", qaoa_result.x)
            print("MaxCut value:", -qaoa_result.fun)  # Flip sign because Qiskit minimizes
        else:
            print("Best params:", qaoa_result.x)
            print("Best value:", qaoa_result.fun)  # Flip sign because Qiskit minimizes   

        return qaoa_result.x

    def qubo_sample(self, optimal_params):
        backend = self.configure_backend()
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = 1000
        optimized_circuit = self.qaoa_circuit.assign_parameters(optimal_params)

        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(1e4))
        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())

        # Reverse the bits of all keys in counts_bin
        reversed_distribution_bin = {
            key[::-1]: val / shots for key, val in counts_bin.items()
        }

        print(reversed_distribution_bin)

        return reversed_distribution_bin

    def print_number_of_qubits(self):
        """
        Calculate and print the number of qubits used in the problem.
        This is determined by the total number of binary variables in the QUBO.
        """
        self.num_qubits = len(self.qb.variables)
        print(f"Number of qubits required: {self.num_qubits}")