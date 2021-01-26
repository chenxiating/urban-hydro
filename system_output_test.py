import sys
import os

file_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_directory)
terminal_output = open("output.txt", 'w')
print("test words before stdout")
print(os.getcwd())
path = os.path.abspath('test.txt')
directory = os.path.dirname(path)

sys.stdout = terminal_output
print("test words after stdout")
print("--linebreak--")
print(directory)

#terminal_output.close() 
