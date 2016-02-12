import sys

if len(sys.argv) != 3:
    raise NotImplementedError('Program only takes two arguments: the output file with the split compound words and the indices mapping')

with open(sys.argv[1], 'r') as output_file:
    output_lines = output_file.readlines() 

with open(sys.argv[2], 'r') as indices_mapping_file:
    mapping_lines = indices_mapping_file.readlines()

assert len(output_lines) == len(mapping_lines)

for output_line, mapping_line in zip(output_lines, mapping_lines):
    mapping_dict = {}
    for mapping in mapping_line.strip().split():
        assert '-' in mapping and mapping.count('-') == 1
        original = mapping.split('-')[0]
        mapped = mapping.split('-')[1]        
        mapping_dict[mapped] = original
    to_print = []
    for pair in output_line.strip().split():
        assert '-' in pair and pair.count('-') == 1
        head = pair.split('-')[0]
        dep = pair.split('-')[1]
        to_print.append(mapping_dict[head] + '-' + dep)    
    print ' '.join(to_print)
