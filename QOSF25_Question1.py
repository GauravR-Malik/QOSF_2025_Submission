#!/usr/bin/env python
# coding: utf-8
"""
Title: Parameter Estimation of an Unknown U3 Gate for Toffoli Circuit Equivalence

Description:
This script aims to determine the parameters (θ, φ, λ) of an unknown U3 gate 
such that the overall 3-qubit circuit becomes equivalent to a Toffoli (CCNOT) gate.

Given the known parts of circuit, it is clear that the first unknown gate could be taken as Hadamard.
This balances the hadamard operator specified at the end, and simplfies the task of finding the other unkown operator.
Taking other unknown operator is found by optimising the parameters such that the overall circuit is minimally away from the Toffoli gate.

Approach:
1. Define all standard gates (I, T, CNOT) manually as matrices, and the first unkown gate as the Hadamard matrix.
2. Construct the known portions of the circuit step-by-step using Kronecker products.
3. Represent the unknown single-qubit U3 gate symbolically with variable parameters.
4. Define the cost function as the Frobenius norm between the full circuit 
   (with U3 substituted) and the ideal Toffoli gate.
5. Use `scipy.optimize.minimize` to find optimal U3 parameters minimizing this cost.

Author: Gaurav Rudra Malik
Date: 17th October 2025
"""

import numpy as np
from scipy.optimize import minimize

# ==============================================================
# 1. Define Standard Quantum Gates
# ==============================================================

# Identity gate
I = np.eye(2)

# Hadamard gate
H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]])

# T gate (π/4 phase gate)
T = np.array([[1, 0],
              [0, np.exp(1j * np.pi / 4)]])

# CNOT gate (control = qubit 1, target = qubit 2)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

print("Defined basic gates: I, H, T, and CNOT.\n")


# ==============================================================
# 2. Construct the Known Circuit Portions
# ==============================================================

# ---- First Segment ----
# Combine gates as per given circuit description.
# This part acts on two qubits and is later extended to three qubits.

op1 = np.kron(T, I)
op2 = np.kron(I, T.T.conj())
op3 = np.kron(I, T)

ckt = op1 @ CNOT @ op2 @ CNOT @ op3
ckt = np.round(ckt, 2)

# Add Hadamard on the third qubit
# (Balancing the final Hadamard in the circuit)
ckt1 = np.kron(ckt, H)


# ==============================================================
# 3. Define CNOT Variants for 3-Qubit System
# ==============================================================

# CNOT13 : control qubit 1 → target qubit 3
CNOT13 = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0]
])

# CNOT23 : control qubit 2 → target qubit 3
CNOT23 = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0]
])

print("Defined 3-qubit CNOT gates: CNOT13 and CNOT23.\n")


# ==============================================================
# 4. Construct Remaining Circuit Segments
# ==============================================================

# Second segment: between the two unknown gates
ckt2 = (
    CNOT23
    @ np.kron(I, np.kron(I, T.conj().T))
    @ CNOT13
    @ np.kron(I, np.kron(I, T))
    @ CNOT23
)
ckt2 = np.round(ckt2, 2)

# Third segment: after the unknown U3 gate
ckt3 = (
    CNOT13
    @ np.kron(I, np.kron(I, T))
    @ np.kron(I, np.kron(I, H))
)


# ==============================================================
# 5. Define the Target: Toffoli (CCNOT) Gate
# ==============================================================

Toffoli = np.eye(8)
Toffoli[6, 6] = 0
Toffoli[6, 7] = 1
Toffoli[7, 6] = 1
Toffoli[7, 7] = 0

print("Constructed Toffoli gate matrix.\n")


# ==============================================================
# 6. Define the Unknown Gate: U3(θ, φ, λ)
# ==============================================================

def U3(theta, phi, lam):
    """
    Standard U3 gate definition:
        U3(θ, φ, λ) =
        [[cos(θ/2), -exp(iλ) sin(θ/2)],
         [exp(iφ) sin(θ/2), exp(i(φ+λ)) cos(θ/2)]]
    """
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
    ])


# ==============================================================
# 7. Build the Complete Circuit for Given Parameters
# ==============================================================

def circuit_matrix(params):
    """
    Given parameters [θ, φ, λ], construct the overall 3-qubit circuit matrix.
    """
    theta, phi, lam = params
    u3_gate = U3(theta, phi, lam)
    gate = np.kron(I, np.kron(I, u3_gate))  # U3 acts on 3rd qubit

    circuit = ckt1 @ ckt2 @ gate @ ckt3
    return circuit


# ==============================================================
# 8. Define Cost Function (Frobenius Norm)
# ==============================================================

def cost_function(params):
    """
    Cost = || U_circuit(params) - U_Toffoli ||_F
    where ||A||_F denotes the Frobenius norm.
    """
    circuit = circuit_matrix(params)
    diff = circuit - Toffoli
    return np.linalg.norm(diff, 'fro')


# ==============================================================
# 9. Run Optimization
# ==============================================================

initial_guess = [0.5, 0.5, 0.5]  # initial parameter values

result = minimize(
    cost_function,
    initial_guess,
    method='BFGS'
)

print("Optimization complete.")
print("Optimal parameters found:")
print(f"  θ = {result.x[0]:.4f}")
print(f"  φ = {result.x[1]:.4f}")
print(f"  λ = {result.x[2]:.4f}")
print(f"\nFinal cost = {result.fun:.6f}\n")


# ==============================================================
# 10. Verify the Result
# ==============================================================

U3_opt = U3(result.x[0], result.x[1], result.x[2])
final_gate = np.kron(I, np.kron(I, U3_opt))
final_circuit = ckt1 @ ckt2 @ final_gate @ ckt3

print("Final circuit matrix (rounded):")
print(np.round(final_circuit, 2))
print("\nAbsolute values of matrix elements:")
print(np.abs(np.round(final_circuit, 2)))


#***************************************************************
# ==============================================================
# Disclaimer:
# Parts of the comments and documentation were refined with the help of an AI assistant to enhance clarity and presentation. 
# The underlying approach, methodology, and circuit design were fully conceived and implemented by the author.
# ==============================================================
#***************************************************************


