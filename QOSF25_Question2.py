#!/usr/bin/env python
# coding: utf-8
"""
Quantum State Preparation Script
---------------------------------
This script constructs an N-qubit quantum state manually from a set of complex amplitudes,
without relying on high-level quantum libraries.

It performs the following tasks:
1. Takes user input for the number of qubits (N).
2. Prompts the user to either:
   - Enter custom complex amplitudes for each basis state, or
   - Use randomly generated complex amplitudes.
3. Normalizes the state vector if it is not already normalized.
4. Displays the resulting quantum state in Dirac notation.
5. Includes basic test routines to verify normalization and dimensional correctness.

Author: Gaurav Rudra Malik
Date: 17th October 2025
"""

import numpy as np


# ---------------------------------------------------------------
# Helper Function: Normalize a Quantum State
# ---------------------------------------------------------------
def normalise(state):
    """
    Normalizes the given complex state vector.

    Parameters
    ----------
    state : np.ndarray
        Complex-valued vector representing the quantum state amplitudes.

    Returns
    -------
    np.ndarray
        Normalized state vector such that ||state|| = 1.
    """
    norm = np.linalg.norm(state)
    return state / norm


# ---------------------------------------------------------------
# User Input: Number of Qubits
# ---------------------------------------------------------------
print('Enter the number of qubits:')
N = int(input())  # Read input and convert to integer

# The total dimension of the Hilbert space for N qubits is 2^N
d = 2**N


# ---------------------------------------------------------------
# Step 1: Generate Default Random Amplitudes
# ---------------------------------------------------------------
# Randomly generate real and imaginary parts to create complex amplitudes
default_coeff = []
for i in range(d):
    a = np.random.rand()  # Real part
    b = np.random.rand()  # Imaginary part
    default_coeff.append(a + 1j * b)


# ---------------------------------------------------------------
# Step 2: Take User Input or Use Default Coefficients
# ---------------------------------------------------------------
coeff = []

print('\nYou shall now be required to enter the coefficients of the state.')
print('To continue manually, press Enter.')
print('To use system-generated values, press X')

temp = input()

if (temp == 'X' or temp == 'x'):
    # If the user chooses auto mode, normalize the random coefficients
    print('Aborted! Using system generated values instead.\n')
    coeff = np.array(default_coeff)
    coeff = normalise(coeff)

else:
    # Manually enter coefficients for each basis state
    for i in range(d):
        binary = np.binary_repr(i, width=N)
        print(f'Enter the coefficient for basis state |{binary}⟩ as (a+bj): ')
        temp = complex(input())  # Convert string input to complex number
        coeff.append(temp)

    coeff = np.array(coeff)

    # Check if normalization condition is met; normalize if necessary
    if abs(np.linalg.norm(coeff) - 1) > 1e-8:
        print('\nEntered state values are not normalized!')
        print('Normalizing now...')
        coeff = normalise(coeff)
        print('Normalization carried out successfully!\n')


# ---------------------------------------------------------------
# Step 3: Display the Constructed Quantum State
# ---------------------------------------------------------------
print('\n The entered generated quantum state is:\n')

output_str = ''
for i in range(d):
    binary = np.binary_repr(i, width=N)
    coeff_str = str(complex(coeff[i]))
    output_str += coeff_str + '|' + binary + '>' + ' + '

# Remove the trailing '+' and print the final expression
output_str = output_str.rstrip(' + ')
print(output_str)


# ---------------------------------------------------------------
# Step 4: Display Summary Information
# ---------------------------------------------------------------
print('\n---------------------------------------------')
print('Norm of the state: ', np.linalg.norm(np.array(coeff)))
print('Number of entries: ', len(coeff))
print('---------------------------------------------')

if len(coeff) == 2**N:
    print(f'Created state is a valid quantum state for {N} qubits.')
else:
    print(f'Invalid number of coefficients for {N}-qubit system.')


# ---------------------------------------------------------------
# Step 5: Unit Tests
# ---------------------------------------------------------------
def test_normalisation():
    """Test that the normalize() function produces a unit-norm state."""
    state = np.array([1+1j, 1-1j, 0, 0])
    normed = normalise(state)
    assert abs(np.linalg.norm(normed) - 1) < 1e-10, "Normalization failed!"


def test_dimension():
    """Test that the state vector has the correct dimensionality for N=2."""
    N = 2
    coeff = np.array([1, 0, 0, 0])
    assert len(coeff) == 2**N, "State vector dimension incorrect!"


# Run tests when script is executed directly
if __name__ == "__main__":
    test_normalisation()
    test_dimension()
    print("\nAll tests passed successfully!")


#***************************************************************
# ==============================================================
# Disclaimer:
# Parts of the comments and documentation were refined with the help of an AI assistant to enhance clarity and presentation. 
# The underlying approach, methodology, and code were fully conceived and implemented by the author.
# ==============================================================
#***************************************************************

