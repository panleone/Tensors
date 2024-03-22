from trg_kernel import trg_algorithm
import sys

########## Main Program #################
print("This program simulates a 2D Ising model on a square lattice")
print("--------------------------------------------------")
print("Please insert in input the following:")
print("1) Temperature T (float and positive);")
print("2) The strength of the external field h (float);")
print("3) The maximum dimension of the t_tensor, Chi (positive integer), the bigger this is the more accurate result will be.")
print("4) Optional, default 10**(-6): The convergence threshold (positive float)")

print("--------------------------------------------------")
print("The program will output the free energy per unit particle f")

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print("Wrong number of inputs!")
    exit()
args = sys.argv
temp = int(args[1])
if temp < 0:
    print("Temperature must be positive!")
    exit()

h = int(args[2])
chi = int(args[3])
if chi < 0:
    print("chi must be positive!")
    exit()
convergence_treshold = 10**(-6)
if len(sys.argv) == 5:
    convergence_treshold = int(args[4])
if convergence_treshold < 0:
    print("the convergence threshold must be positive!")
    exit()
print("The free energy per unit particle is f = ", trg_algorithm(temp, h, chi, convergence_treshold))

