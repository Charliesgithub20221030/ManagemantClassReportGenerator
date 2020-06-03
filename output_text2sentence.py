
with open("generated/output.txt", 'r') as f:
    data = f.readlines()

data = [i for i in data[0].split(' ') if len(i) < 128]

with open("generated/output_sentence.txt", 'w') as f:
    for line in data:
        if line == ' ' or line == '':
            continue
        print('line length %d' % len(line))
        f.write(line+'\n')
print('done')
