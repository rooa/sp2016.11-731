import sys

if len(sys.argv) != 3:
    raise NotImplementedError('The program takes two arguments: the German corpus with the compound split and the original parallel corpus')

with open(sys.argv[1], 'r') as input_file:
    split_lines = input_file.readlines()

with open(sys.argv[2], 'r') as input_file:
    parallel_lines = input_file.readlines()

assert len(split_lines) == len(parallel_lines)

for split_line, parallel_line in zip(split_lines, parallel_lines):
    german_sent = split_line.split('|||')[1].strip()
    english_sent = parallel_line.split('|||')[1].strip()
    print german_sent + ' ' + '|||' + ' ' + english_sent
    

