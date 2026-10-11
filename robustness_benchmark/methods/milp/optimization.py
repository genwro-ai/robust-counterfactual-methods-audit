"""MILP encoding of an (interval-abstracted) ReLU network for Gurobi.

With free inputs (MCE inside AP-Delta-S), the FeatureSchema is encoded exactly:
immutable inputs are fixed to the instance, each mutable one-hot group is a
set of binary variables summing to 1, and each integer feature is an integer
variable in raw units linked to its standardized input. Fixed inputs (RNCE's
robustness check of training points) are valid encodings already.
"""

import pandas as pd
from gurobipy import GRB, Model
from gurobipy.gurobipy import quicksum

from robustness_benchmark.core.task import ClassificationTask
from robustness_benchmark.methods.milp.interval_abstraction import (
    IntervalAbstractionPytorch,
)


class OptSolver:
    """
    A solver class that uses Gurobi to optimize a model based on a given task and instance.

    Attributes / Properties
    -------
    task: Task
        The task to be optimized.
    gurobiModel: Model
        The Gurobi optimization model.
    inputNodes: dict
        Dictionary to store Gurobi variables for input nodes.
    outputNode: Gurobi variable
        The Gurobi variable representing the output node.

    Methods
    -------
    setup(instance, desired_output=1, delta=0.5, bias_delta=0, M=1000000000, epsilon=0.0001, fix_inputs=True):
        Sets up the Gurobi model with constraints based on the provided instance and parameters.

    -------
    """

    def __init__(self, ct: ClassificationTask):
        """
        Initializes the OptSolver with a given Task.

        @param ct: Task, The task to be optimized.
        """
        self.task = ct
        self.gurobiModel = Model()
        self.inputNodes = None
        self.outputNode = None

    def setup(self, instance, delta=0, bias_delta=0, M=1000, fix_inputs=True):
        """
        Sets up the Gurobi model with constraints based on the provided instance and parameters.

        @param instance: pd.DataFrame or list, The input data instance for which to set up the model.
        @param desired_output: int, Optional, The desired output value (default is 1).
        @param delta: float, Optional, The delta value used in constraints (default is 0.5).
        @param bias_delta: float, Optional, The bias delta value used in constraints (default is 0).
        @param M: float, Optional, A large constant used in constraints (default is 1000000000).
        @param epsilon: float, Optional, The epsilon value used in constraints (default is 0.0001).
        @param fix_inputs: bool, Optional, Whether to fix input values or use variable bounds (default is True).

        @return: None
        """

        # Turn off the Gurobi output
        self.gurobiModel.setParam("OutputFlag", 0)

        # Convert instance to a list
        if isinstance(instance, pd.DataFrame):
            if instance.empty:
                raise ValueError("Empty instance provided")
            ilist = instance.iloc[0].tolist()
        else:
            ilist = instance.tolist()

        intabs = IntervalAbstractionPytorch(
            self.task.model, delta, bias_delta=bias_delta
        )

        self.inputNodes = {}
        all_nodes = {}
        activation_states = {}
        input_magnitudes = {}

        if fix_inputs:
            # Create the Gurobi variables for the inputs
            for i in range(len(self.task.training_data.X.columns)):
                key = f"v_0_{i}"
                self.inputNodes[key] = self.gurobiModel.addVar(
                    lb=-float("inf"), name=key
                )
                all_nodes[key] = self.inputNodes[key]

                self.gurobiModel.addConstr(
                    self.inputNodes[key] == ilist[i], name=f"constr_input_{i}"
                )

        else:
            schema = self.task.schema
            immutable = set(schema.immutable_columns())
            one_hot = {i for group in schema.one_hot_groups().values() for i in group}
            integer = dict(zip(schema.indices(schema.integer), schema.integer))
            # Create the Gurobi variables for the inputs
            for i, col in enumerate(self.task.training_data.X.columns):
                key = f"v_0_{i}"

                # Calculate the minimum and maximum values for the current column
                col_min = self.task.training_data.X[col].min()
                col_max = self.task.training_data.X[col].max()

                # Use the calculated min and max for the bounds of the variable
                if i in immutable:
                    col_min = col_max = ilist[i]
                self.inputNodes[key] = self.gurobiModel.addVar(
                    lb=col_min,
                    ub=col_max,
                    vtype=GRB.BINARY if i in one_hot - immutable else GRB.CONTINUOUS,
                    name=key,
                )
                all_nodes[key] = self.inputNodes[key]
                if i in integer and i not in immutable:
                    mean, scale = schema.numeric_scale[integer[i]]
                    raw = self.gurobiModel.addVar(
                        lb=round(float(col_min) * scale + mean),
                        ub=round(float(col_max) * scale + mean),
                        vtype=GRB.INTEGER,
                        name=f"raw_{i}",
                    )
                    self.gurobiModel.addConstr(raw == mean + scale * all_nodes[key])

                if delta > 0:
                    max_magnitude = max(abs(float(col_min)), abs(float(col_max)))
                    magnitude = self.gurobiModel.addVar(
                        lb=0,
                        ub=max_magnitude,
                        name=f"abs_{key}",
                    )
                    self.gurobiModel.addConstr(magnitude >= all_nodes[key])
                    self.gurobiModel.addConstr(magnitude >= -all_nodes[key])
                    input_magnitudes[key] = magnitude

            for feature, group in schema.one_hot_groups().items():
                if feature not in schema.immutable:
                    self.gurobiModel.addConstr(
                        quicksum(self.inputNodes[f"v_0_{i}"] for i in group) == 1,
                        name=f"one_hot_{feature}",
                    )

        self.gurobiModel.update()

        def interval_affine(layer, node):
            """Return upper and lower affine bounds of a pre-activation.

            First-layer inputs can be negative, so their weight intervals are
            sign-aware; later layers read nonnegative ReLU outputs.
            """

            keys = [
                f"weight_l{layer}_n{prev_node_index}_to_l{layer + 1}_n{node}"
                for prev_node_index in range(intabs.layers[layer])
            ]
            previous_nodes = [
                all_nodes[f"v_{layer}_{prev_node_index}"]
                for prev_node_index in range(intabs.layers[layer])
            ]
            if layer == 0 and delta > 0:
                upper_terms = []
                lower_terms = []
                for prev_node_index, (key, previous_node) in enumerate(
                    zip(keys, previous_nodes, strict=True)
                ):
                    weight_interval = intabs.weight_intervals[key]
                    nominal_weight = sum(weight_interval) / 2
                    weight_radius = (weight_interval[1] - weight_interval[0]) / 2
                    magnitude = (
                        abs(ilist[prev_node_index])
                        if fix_inputs
                        else input_magnitudes[f"v_0_{prev_node_index}"]
                    )
                    upper_terms.append(
                        nominal_weight * previous_node + weight_radius * magnitude
                    )
                    lower_terms.append(
                        nominal_weight * previous_node - weight_radius * magnitude
                    )
                return quicksum(upper_terms), quicksum(lower_terms)
            return (
                quicksum(
                    intabs.weight_intervals[key][1] * previous_node
                    for key, previous_node in zip(keys, previous_nodes, strict=True)
                ),
                quicksum(
                    intabs.weight_intervals[key][0] * previous_node
                    for key, previous_node in zip(keys, previous_nodes, strict=True)
                ),
            )

        num_layers = len(intabs.layers)

        # Iterate through all "hidden" layers
        for layer in range(num_layers - 2):
            # Go through each node in the current layer
            for node in range(intabs.layers[layer + 1]):
                var_name = f"v_{layer + 1}_{node}"
                activation_name = f"xi_{layer + 1}_{node}"

                all_nodes[var_name] = self.gurobiModel.addVar(
                    lb=-float("inf"), name=var_name
                )
                activation_states[activation_name] = self.gurobiModel.addVar(
                    vtype=GRB.BINARY, name=activation_name
                )

                self.gurobiModel.update()

                # 1) Add v_i_j >= 0 constraint
                self.gurobiModel.addConstr(
                    all_nodes[var_name] >= 0, name="constr1_" + var_name
                )

                # 2) Add v_i_j <= M ( 1 - xi_i_j )
                self.gurobiModel.addConstr(
                    M * (1 - activation_states[activation_name]) >= all_nodes[var_name],
                    name="constr2_" + var_name,
                )

                upper_affine, lower_affine = interval_affine(layer, node)

                # 3) Add the upper interval constraint and the inactive ReLU term.
                self.gurobiModel.addConstr(
                    upper_affine
                    + intabs.bias_intervals[f"bias_into_l{layer + 1}_n{node}"][1]
                    + M * activation_states[activation_name]
                    >= all_nodes[var_name],
                    name="constr3_" + var_name,
                )

                # 4) Add the lower interval constraint.
                self.gurobiModel.addConstr(
                    lower_affine
                    + intabs.bias_intervals[f"bias_into_l{layer + 1}_n{node}"][0]
                    <= all_nodes[var_name],
                    name="constr4_" + var_name,
                )

                self.gurobiModel.update()

        # Create a singular output node
        self.outputNode = self.gurobiModel.addVar(
            lb=-float("inf"), vtype=GRB.CONTINUOUS, name="output_node"
        )

        # Without hidden layers the output layer reads the inputs directly.
        output_upper, output_lower = interval_affine(num_layers - 2, 0)

        # Constraint 1: node <= ub(W)x + ub(B)
        self.gurobiModel.addConstr(
            output_upper + intabs.bias_intervals[f"bias_into_l{num_layers - 1}_n{0}"][1]
            >= self.outputNode,
            name="output_node_C1",
        )

        # Constraint 2: node => lb(W)x + lb(B)
        self.gurobiModel.addConstr(
            output_lower + intabs.bias_intervals[f"bias_into_l{num_layers - 1}_n{0}"][0]
            <= self.outputNode,
            name="output_node_C2",
        )
